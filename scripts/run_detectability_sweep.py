#!/usr/bin/env python
"""Controlled detectability sweep over nside × density × f_sky × amplitude.

STAGE 2 — PREPARED, NOT YET RUN. This traces the analytic detectability-law
curves (``docs/detectability_law.rst``) empirically over the full design grid
and pins the clustering floor. It is a *heavy remote job*; run it under
nohup/tmux on the compute host (see ``bash/run_remote_full.sh``). On a laptop use
only ``--check`` (validates inputs + prints the plan, zero compute).

It adds the ``--fsky`` footprint-fraction knob that no existing sweep CLI has
(realised as a polar cap of the requested area on the GLASS full-sky field).

Reuses the library estimators end to end:
``glass_mocks.generate_glass_delta_map`` → ``sample_positions_from_delta`` (with a
``vis`` cap) → ``contamination.apply_contamination`` → ``regression.run_decontamination``
→ ``covariance.mock_sandwich_covariance`` for the calibrated per-template sigma.

Example (remote)::

    python scripts/run_detectability_sweep.py \
        --nsides 32 64 128 256 --n-means 8 30 127 490 \
        --fskys 0.1 0.25 0.44 --amps 0.005 0.01 0.03 0.05 0.1 \
        --n-sims 10 --methods OLS ISD-1 --out results/detectability_sweep.csv

Continue after a timeout/crash (appends, skips cells already in the CSV)::

    python scripts/run_detectability_sweep.py ... --resume --out results/detectability_sweep.csv

Check only (laptop)::

    python scripts/run_detectability_sweep.py --check
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results" / "detectability_sweep.csv"

# Fast methods only for the sweep. ElasticNet is EXCLUDED from low-amplitude
# cells (documented CV-alpha->0 ~40 min/fit blow-up); include it only at amp>=0.05.
FAST_METHODS = ("OLS", "ISD-1")
ELASTICNET_MIN_AMP = 0.05


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nsides", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--n-means", type=float, nargs="+", default=[8, 30, 127, 490],
                   help="mean galaxies per pixel (density knob)")
    p.add_argument("--fskys", type=float, nargs="+", default=[0.1, 0.25, 0.44],
                   help="sky fraction (polar-cap area); the knob missing from every other sweep CLI")
    p.add_argument("--amps", type=float, nargs="+", default=[0.005, 0.01, 0.03, 0.05, 0.1],
                   help="injected systematic field RMS")
    p.add_argument("--n-sims", type=int, default=30)
    p.add_argument("--n-sys", type=int, default=8, help="number of synthetic templates")
    p.add_argument("--methods", nargs="+", default=list(FAST_METHODS))
    p.add_argument("--cl-amplitude", type=float, default=5e-4)
    p.add_argument("--n-mock", type=int, default=100, help="uncontaminated mocks for the sandwich sigma")
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--resume", action="store_true",
                   help="append to an existing --out, skipping cells already in it "
                        "(use after a timeout/crash); without it --out is truncated")
    p.add_argument("--check", action="store_true", help="validate inputs + print the plan; run NOTHING")
    p.add_argument("--dry-run", action="store_true", help="alias of --check")
    return p.parse_args(argv)


def fsky_cap(nside: int, fsky: float) -> np.ndarray:
    """Boolean visibility map: a polar cap of area f_sky (contiguous footprint)."""
    import healpy as hp

    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return (np.cos(theta) > (1.0 - 2.0 * fsky)).astype(float)


def make_templates(nside: int, n_sys: int, seed: int) -> np.ndarray:
    """``n_sys`` smooth, mean-zero, unit-variance templates (low-ell GLASS fields)."""
    from sys_mapping import glass_mocks

    cols = []
    for i in range(n_sys):
        t = glass_mocks.generate_glass_delta_map(nside, z_max=0.5, cl_amplitude=1e-3, seed=seed + 101 * i)
        t = t - t.mean()
        s = t.std()
        cols.append(t / s if s > 0 else t)
    return np.array(cols).T  # (n_pix, n_sys)


def n_configs(a) -> int:
    return len(a.nsides) * len(a.n_means) * len(a.fskys) * len(a.amps)


# Per-cell cost is set by the ``n_mock`` Poisson mocks drawn over the visible
# pixels and the sandwich covariance built from them — NOT by the number of fits.
# So it scales as npix_vis * n_mock, with a small per-method term on top.
# Calibrated 2026-08-17 (1 core, n_mock=100, methods=OLS+ISD-1):
#     nside   fsky   npix_vis    measured
#        64   0.25      12.3k       1.5 s
#       128   0.25      49.2k       0.6 s
#       256   0.44     346k         3.6 s
#       512   0.05     157k         3.2 s
#      1024   0.05     629k        14.2 s
# Accurate at nside >= 512 (where the time actually goes), conservative by ~2x at
# coarse nside. Re-measure and update this table if the estimate drifts.
SEC_PER_PIX_MOCK = 2.0e-7


# Resident-set model, calibrated against the same runs (measured 1.3 / 1.3 / 3.0 GB
# at nside 256/512/1024): a fixed interpreter+JAX+healpy baseline, plus ~2x the two
# big arrays — the FULL-SKY template block (make_templates keeps all n_sys maps at
# npix_full, before the f_sky cut) and the (n_mock, npix_vis) mock ensemble.
RSS_BASELINE_B = 0.8e9
RSS_ARRAY_FACTOR = 2.0


def cell_cost(a, nside: int, fsky: float) -> tuple[float, int]:
    """(estimated seconds, peak bytes) for one (config, sim) cell."""
    npix_full = 12 * nside * nside
    npix_vis = npix_full * fsky
    secs = SEC_PER_PIX_MOCK * npix_vis * (a.n_mock + 5 * len(a.methods))
    arrays = (npix_full * a.n_sys + npix_vis * a.n_mock) * 8
    peak = int(RSS_BASELINE_B + RSS_ARRAY_FACTOR * arrays)
    return secs, peak


def estimate_cost(a) -> tuple[float, int]:
    """(total seconds, peak bytes of the largest cell) over the whole grid."""
    total, peak = 0.0, 0
    for nside in a.nsides:
        for fsky in a.fskys:
            secs, cell_peak = cell_cost(a, nside, fsky)
            total += secs * len(a.n_means) * len(a.amps) * a.n_sims
            peak = max(peak, cell_peak)
    return total, peak


COLS = ["nside", "n_mean", "fsky", "npix", "amp", "sim", "method",
        "field_snr", "max_snr_cal", "a_field_corr"]


def n_methods_at(a, amp: float) -> int:
    """How many rows one cell at this amplitude should produce (see run_cell)."""
    return sum(1 for m in a.methods
               if not (m == "ElasticNet" and amp < ELASTICNET_MIN_AMP))


def cell_key(nside, n_mean, fsky, amp, sim) -> str:
    """Identity of a (config, sim) cell, as it appears in the CSV.

    Built from ``str()`` of each field — the same conversion ``csv.DictWriter``
    applies on write — so a key built in memory compares equal to one read back
    out of the file. Do NOT round-trip through float(): 0.005 and 5e-3 are the
    same number but different CSV text, and the done-set is matched on text.
    """
    return "|".join(str(v) for v in (nside, n_mean, fsky, amp, sim))


def load_done(a, path: Path) -> tuple[set, list]:
    """``(done cell keys, rows to keep)`` from an existing sweep CSV, for --resume.

    A cell counts as done only when *every* row it should have is present and
    complete. A hard kill leaves a torn final line, and can leave a cell whose
    method rows were only partly written; treating either as done would silently
    drop that cell from the sweep — it would never be recomputed and never appear
    in the results. So short/long lines are discarded, cells are required to carry
    their full complement of method rows, and the rows belonging to incomplete
    cells are dropped too (they would otherwise duplicate when the cell re-runs).

    The caller rewrites the file from the returned rows, which is what removes the
    torn tail.
    """
    if not path.exists():
        return set(), []
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if None in row or any(row.get(c) in (None, "") for c in COLS):
                continue  # torn or malformed line
            rows.append(row)

    keyed = [(cell_key(r["nside"], r["n_mean"], r["fsky"], r["amp"], r["sim"]), r)
             for r in rows]
    counts = Counter(k for k, _ in keyed)
    done = set()
    for key, n in counts.items():
        try:
            amp = float(key.split("|")[3])
        except ValueError:
            continue
        if n >= n_methods_at(a, amp):
            done.add(key)
    return done, [r for k, r in keyed if k in done]


def check(a) -> int:
    """Validate inputs, imports, and output path; print the plan. Zero compute."""
    ok = True
    try:
        import healpy  # noqa: F401
        from sys_mapping import contamination, covariance, glass_mocks, regression  # noqa: F401
        print("[check] imports OK (healpy, glass_mocks, contamination, regression, covariance)")
    except Exception as e:  # pragma: no cover
        print(f"[check] IMPORT FAILURE: {e}"); ok = False
    try:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[check] output writable: {a.out}")
    except Exception as e:
        print(f"[check] OUTPUT NOT WRITABLE: {e}"); ok = False
    ncfg = n_configs(a)
    nfit = ncfg * a.n_sims * len(a.methods)
    est_s, peak_b = estimate_cost(a)
    print(f"[check] grid: nsides={a.nsides} n_means={a.n_means} fskys={a.fskys} amps={a.amps}")
    print(f"[check] configs={ncfg}  sims/config={a.n_sims}  methods={a.methods}  -> fits={nfit}")
    print(f"[check] sandwich mocks/cell={a.n_mock}  cells={ncfg * a.n_sims}")
    print(f"[check] estimated {est_s / 3600.0:.1f} core-h  ({est_s / 60.0:.0f} core-min), "
          f"peak RSS ~{peak_b / 1e9:.1f} GB (largest cell) -> size SLURM --mem above this")
    if a.resume and a.out.exists():
        n_done = len(load_done(a, a.out)[0])
        print(f"[check] --resume: {n_done} complete cells already in {a.out.name}, "
              f"{max(ncfg * a.n_sims - n_done, 0)} to go")
    if any(m == "ElasticNet" for m in a.methods):
        print(f"[check] NOTE: ElasticNet skipped at amp<{ELASTICNET_MIN_AMP} (CV-alpha blow-up)")
    print(f"[check] {'READY' if ok else 'NOT READY'} -> {a.out}")
    return 0 if ok else 1


def run_cell(a, nside, n_mean, fsky, amp, sim):
    """One (config, sim) cell -> list of per-method result rows. HEAVY (remote)."""
    import healpy as hp
    from sys_mapping import contamination, covariance, glass_mocks, regression

    seed = a.seed + sim
    vis = fsky_cap(nside, fsky)
    npix = int(vis.sum())
    T = make_templates(nside, a.n_sys, a.seed)  # (n_pix_full, n_sys)
    Tv = T[vis > 0]
    Tv = (Tv - Tv.mean(0)) / Tv.std(0)

    # ground truth: unit-variance field of RMS = amp on the templates
    rng = np.random.default_rng(seed + 777)
    avec = rng.normal(size=a.n_sys)
    f = Tv @ avec
    avec *= amp / (f.std() + 1e-12)
    bvec = np.zeros(a.n_sys)

    delta = glass_mocks.generate_glass_delta_map(nside, z_max=0.5, cl_amplitude=a.cl_amplitude, seed=seed)
    dv = delta[vis > 0]
    dcont = np.asarray(contamination.apply_contamination(dv, Tv.T, avec, bvec))

    # Poisson-sample counts at n_mean/pixel from (1+dcont); build observed overdensity
    lam = n_mean * np.clip(1.0 + dcont, 0.0, None)
    counts = rng.poisson(lam)
    dg_obs = counts / max(counts.mean(), 1e-9) - 1.0

    # uncontaminated mock ensemble for the sandwich sigma
    mocks = []
    for k in range(a.n_mock):
        c = rng.poisson(n_mean * np.clip(1.0 + dv, 0.0, None))
        mocks.append(c / max(c.mean(), 1e-9) - 1.0)
    cov_sw = covariance.mock_sandwich_covariance(Tv.T, np.array(mocks))
    sig_sw = np.sqrt(np.clip(np.diag(cov_sw), 0, None))

    rows = []
    for method in a.methods:
        if method == "ElasticNet" and amp < ELASTICNET_MIN_AMP:
            continue
        res = regression.run_decontamination(method, dg_obs, Tv.T, seed=seed)
        ahat = np.asarray(res.get("a_hat", res.get("alpha_hat", np.zeros(a.n_sys))))[: a.n_sys]
        field_snr = np.linalg.norm(Tv @ ahat) / (np.std(dg_obs - Tv @ ahat) + 1e-12)
        snr_cal = np.abs(ahat) / np.clip(sig_sw, 1e-12, None)
        rows.append(dict(nside=nside, n_mean=n_mean, fsky=fsky, npix=npix, amp=amp,
                         sim=sim, method=method, field_snr=field_snr,
                         max_snr_cal=float(np.max(snr_cal)),
                         a_field_corr=float(np.corrcoef(Tv @ ahat, f)[0, 1])))
    return rows


def main(argv=None):
    a = parse_args(argv)
    if a.check or a.dry_run:
        return check(a)
    # HEAVY path (remote only)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # --resume keeps the cells already in the CSV and recomputes only the rest, so
    # a SLURM timeout or a kill costs one cell rather than the whole run. Without
    # --resume an existing file is overwritten, as before — say so rather than
    # doing it quietly.
    resuming = a.resume and a.out.exists()
    done, keep = load_done(a, a.out) if resuming else (set(), [])
    if resuming:
        # Rewrite the sanitised rows through a temp file + atomic rename before
        # appending anything: truncating the real file in place would lose the
        # whole run if we died in the gap between truncate and write.
        print(f"[sweep] resuming {a.out}: {len(done)} complete cells kept "
              f"({len(keep)} rows)", flush=True)
        tmp = a.out.with_name(a.out.name + ".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader(); w.writerows(keep)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, a.out)
    elif a.out.exists():
        print(f"[sweep] WARNING: overwriting {a.out} (pass --resume to continue it)", flush=True)

    n_new, n_skip = 0, 0
    with open(a.out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if not resuming:
            w.writeheader()
        for nside in a.nsides:
            for n_mean in a.n_means:
                for fsky in a.fskys:
                    for amp in a.amps:
                        for sim in range(a.n_sims):
                            if cell_key(nside, n_mean, fsky, amp, sim) in done:
                                n_skip += 1
                                continue
                            for row in run_cell(a, nside, n_mean, fsky, amp, sim):
                                w.writerow(row); n_new += 1
                            fh.flush()  # a kill now costs at most one cell
                        print(f"[sweep] nside={nside} n_mean={n_mean} fsky={fsky} amp={amp} done", flush=True)
    print(f"[sweep] wrote {n_new} rows ({n_skip} cells skipped as done) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
