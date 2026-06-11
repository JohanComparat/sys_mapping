"""Full sys_mapping pipeline on Legacy Survey DR10 BGS galaxy catalogs.

Loads DATA+RAND FITS files, builds the overdensity field, runs MCMC for the
additive and combined models, applies debiasing + 2PCF correction, writes
per-galaxy systematic weights and diagnostic plots.

This script extends compute_sys_weights.py by adding w(θ) measurement and
systematic-correction plots for each sample.

Usage
-----
# All BGS VLIM samples in a catalog directory
python scripts/run_ls10_analysis.py --catalog-dir /path/to/BGS_VLIM_Mstar

# Single sample
python scripts/run_ls10_analysis.py \\
    --catalog-dir /path/to/BGS_VLIM_Mstar \\
    --sample LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228

# With external template maps
python scripts/run_ls10_analysis.py \\
    --catalog-dir /path/to/BGS_VLIM_Mstar \\
    --template-dir /path/to/systematics/ \\
    --nside 64 --output-dir results/ls10/
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import healpy as hp
import numpy as np
import yaml
from astropy.io import fits

import sys_mapping as sm
from sys_mapping.plotting import METHOD_COLORS, METHOD_LINESTYLES, METHOD_LABELS
from sys_mapping.correction import correct_two_point_function

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _parse_z_range(sample_id: str) -> tuple[float, float]:
    """Extract z_min, z_max from a sample_id like …_0.05_z_0.22_…."""
    m = re.search(r'_(\d+\.\d+)_z_(\d+\.\d+)_', sample_id)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.05, 0.35  # BGS fallback


# ── Template loading ───────────────────────────────────────────────────────

def load_templates_from_dir(template_dir, nside):
    """Return (n_sys, n_pix) template array from all *.fits files in template_dir."""
    fits_files = sorted(Path(template_dir).glob("*.fits"))
    if not fits_files:
        return None, []
    maps, names = [], []
    for f in fits_files:
        m = hp.read_map(str(f))
        if len(m) != hp.nside2npix(nside):
            m = hp.ud_grade(m, nside)
        good = m != hp.UNSEEN
        if good.any():
            m[~good] = 0.0
            m -= np.mean(m[good])
            std = np.std(m[good])
            if std > 0:
                m /= std
        maps.append(m)
        names.append(f.stem)
    return np.array(maps), names


def synthetic_templates(nside, n_families=5, seed=0):
    families = list(range(n_families))
    tmpl = sm.generate_systematic_maps(nside, families=families, seed=seed)
    names = [f"synth_family_{i}" for i in families]
    return tmpl, names


# ── Per-sample pipeline ────────────────────────────────────────────────────

_DOCS_STATIC_LS10 = Path(__file__).resolve().parent.parent / "docs" / "_static" / "results_ls10"


def _regen_weight_figures(sample_id, nside, templates, good_pix,
                          all_method_results, n_sys, n_pix, outdir, docs_dir):
    """Regenerate weight-map and weight-histogram figures from saved a_hat/b_hat."""
    import matplotlib
    import matplotlib.pyplot as plt
    import shutil as _shutil

    _METHOD_ORDER_FIG = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
    _wmap_full = {}
    for _mf in _METHOD_ORDER_FIG:
        _rf = all_method_results.get(_mf, {})
        _params = np.asarray(_rf.get("b_hat" if _mf == "MCMC-comb" else "a_hat",
                                     np.zeros(n_sys)))
        if len(_params) != n_sys:
            _params = np.zeros(n_sys)
        _t_full = np.einsum("i,ij->j", _params, templates)
        _wm = np.full(n_pix, np.nan)
        _wm[good_pix] = 1.0 / np.maximum(1.0 + _t_full[good_pix], 0.01)
        _wmap_full[_mf] = _wm

    _all_weights_good = np.concatenate([
        _wmap_full[m][good_pix] for m in _METHOD_ORDER_FIG
        if np.any(np.isfinite(_wmap_full[m][good_pix]))
    ])
    _vabs = max(abs(np.nanpercentile(_all_weights_good, 2) - 1),
                abs(np.nanpercentile(_all_weights_good, 98) - 1), 0.05)
    _vmin_map, _vmax_map = 1.0 - _vabs, 1.0 + _vabs

    # Weight map 2×3 mollview
    fig12, axes12 = plt.subplots(2, 3, figsize=(18, 8))
    for _ax12, _mf in zip(axes12.flat, _METHOD_ORDER_FIG):
        _wm = _wmap_full[_mf].copy()
        _wm[~good_pix] = hp.UNSEEN
        plt.sca(_ax12)
        hp.mollview(_wm, title=METHOD_LABELS.get(_mf, _mf),
                    min=_vmin_map, max=_vmax_map,
                    cmap="RdBu_r", hold=True, notext=True, unit="weight")
    plt.suptitle(f"Systematic weight maps — {sample_id}\nNSIDE={nside}", fontsize=11)
    wmap_path = outdir / f"{sample_id}_NSIDE{nside:04d}_weight_map.png"
    plt.savefig(str(wmap_path), dpi=110, bbox_inches="tight")
    _shutil.copy(wmap_path, docs_dir / wmap_path.name)
    plt.close()
    print(f"Plot saved: {wmap_path}")

    # Weight histogram
    fig13, ax13 = plt.subplots(figsize=(9, 5))
    for _mf in _METHOD_ORDER_FIG:
        _wg = _wmap_full[_mf][good_pix]
        _wg = _wg[np.isfinite(_wg)]
        ax13.hist(_wg, bins=60, range=(_vmin_map, _vmax_map),
                  color=METHOD_COLORS.get(_mf, "grey"),
                  ls=METHOD_LINESTYLES.get(_mf, "-"),
                  histtype="step", lw=1.5, alpha=0.85,
                  label=METHOD_LABELS.get(_mf, _mf))
    ax13.axvline(1.0, color="k", lw=0.8, ls="--")
    ax13.set_xlabel("Pixel weight")
    ax13.set_ylabel("Number of pixels")
    ax13.set_yscale("log")
    ax13.set_title(f"Weight distributions — {sample_id}  NSIDE={nside}")
    ax13.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    whist_path = outdir / f"{sample_id}_NSIDE{nside:04d}_weight_hist.png"
    plt.savefig(str(whist_path), dpi=130, bbox_inches="tight")
    _shutil.copy(whist_path, docs_dir / whist_path.name)
    plt.close()
    print(f"Plot saved: {whist_path}")


def _plot_wtheta_figure(theta_arcmin, w_obs, all_w_corr, sample_id, nside, n_gal,
                        outdir, docs_dir):
    """Log-log w(θ) comparison for all 6 methods.

    Top panel: log-log scale, positive values only.
    Bottom panel: fractional correction δw/w = (w_corr - w_obs)/w_obs on linear scale.
    """
    import matplotlib.pyplot as plt
    import shutil as _shutil

    _METHOD_ORDER_FIG = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
    w_sig = np.abs(w_obs) > 1e-5 * np.abs(w_obs).max()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5]})
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    _wobs_pos = np.where(w_obs > 0, w_obs, np.nan)
    ax.plot(theta_arcmin, _wobs_pos,
            color=METHOD_COLORS['observed'], ls='-.', lw=2.5, label='Observed', zorder=10)
    for _m in _METHOD_ORDER_FIG:
        _wc = all_w_corr.get(_m)
        if _wc is None:
            continue
        ax.plot(theta_arcmin, np.where(_wc > 0, _wc, np.nan),
                color=METHOD_COLORS.get(_m, 'grey'),
                ls=METHOD_LINESTYLES.get(_m, '-'), lw=1.5,
                label=METHOD_LABELS.get(_m, _m))
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(f"{sample_id}\nNSIDE={nside}  n_gal={n_gal:,}")
    ax.legend(fontsize=8, ncol=2)

    ax2 = axes[1]
    ax2.set_xscale("log")
    ax2.axhline(0, color='k', lw=0.8, ls='--', zorder=1)
    _w_ref = np.where(np.abs(w_obs) > 1e-8, w_obs, np.nan)
    for _m in _METHOD_ORDER_FIG:
        _wc = all_w_corr.get(_m)
        if _wc is None:
            continue
        _frac = (_wc - w_obs) / _w_ref
        _frac[~w_sig] = np.nan
        ax2.plot(theta_arcmin, _frac,
                 color=METHOD_COLORS.get(_m, 'grey'),
                 ls=METHOD_LINESTYLES.get(_m, '-'), lw=1.5,
                 label=METHOD_LABELS.get(_m, _m))
    ax2.set_xlabel(r"$\theta$ [arcmin]")
    ax2.set_ylabel(r"$\delta w / w_{\rm obs}$")

    plt.tight_layout()
    fig_path = Path(outdir) / f"{sample_id}_NSIDE{nside:04d}_wtheta.png"
    plt.savefig(str(fig_path), dpi=130, bbox_inches="tight")
    _shutil.copy(fig_path, Path(docs_dir) / fig_path.name)
    plt.close()
    print(f"Plot saved: {fig_path}")


def run_sample(sample_id, data_file, rand_file, templates, template_names,
               nside, n_walkers, n_steps, n_burn, output_dir, args,
               only_methods=None, force=False, figures_only=False,
               preselect=False, preselect_method="isd", preselect_n_top=None,
               preselect_p_threshold=0.05, preselect_n_mocks=100):
    import matplotlib.pyplot as plt

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    docs_dir = _DOCS_STATIC_LS10
    docs_dir.mkdir(parents=True, exist_ok=True)
    n_sys = templates.shape[0]
    n_pix = hp.nside2npix(nside)

    print(f"\n{'='*60}")
    print(f"Sample: {sample_id}")
    print(f"{'='*60}")

    # ── figures-only: reload saved results and skip MCMC ─────────────────────
    if figures_only:
        import json as _json_fo
        _all_mr_fo: dict = {}
        _params_path_fo = outdir / f"{sample_id}_NSIDE{nside:04d}_params.json"
        if _params_path_fo.exists():
            _pd = _json_fo.loads(_params_path_fo.read_text())
            _all_mr_fo["MCMC-add"] = {
                "a_hat": np.array(_pd.get("a_hat_add", np.zeros(n_sys))),
                "b_hat": np.zeros(n_sys),
                "sigma_hat": _pd.get("sigma_hat_add"),
                "acceptance_fraction": _pd.get("acceptance_fraction_add"),
            }
            _all_mr_fo["MCMC-comb"] = {
                "a_hat": np.array(_pd.get("a_hat_add", np.zeros(n_sys))),
                "b_hat": np.array(_pd.get("b_hat_comb", np.zeros(n_sys))),
                "R": None,
                "sigma_hat": _pd.get("sigma_hat_comb"),
                "acceptance_fraction": _pd.get("acceptance_fraction_comb"),
            }
        for _pf in sorted(outdir.glob(f"{sample_id}_NSIDE{nside:04d}_partial_*.json")):
            try:
                _sub_data = _json_fo.loads(_pf.read_text())
                for _m in _sub_data.get("methods_run", []):
                    if _m not in _all_mr_fo and _m in _sub_data:
                        _ah = _sub_data[_m].get("a_hat", [])
                        _all_mr_fo[_m] = {
                            "a_hat": np.array(_ah if _ah else np.zeros(n_sys)),
                            "b_hat": np.zeros(n_sys),
                            "sigma_hat": _sub_data[_m].get("sigma_hat"),
                            "acceptance_fraction": _sub_data[_m].get("acceptance_fraction"),
                        }
            except Exception:
                pass
        # Rebuild good_pix from galaxy catalog (needed for weight map footprint)
        with fits.open(data_file, memmap=True) as _hdul:
            _cat = _hdul[1].data
            _ra_gal_fo = np.asarray(_cat["RA"], dtype=np.float64)
            _dec_gal_fo = np.asarray(_cat["DEC"], dtype=np.float64)
        with fits.open(rand_file, memmap=True) as _hdul:
            _rand_fo = _hdul[1].data
            _ra_rand_fo = np.asarray(_rand_fo["RA"], dtype=np.float64)
            _dec_rand_fo = np.asarray(_rand_fo["DEC"], dtype=np.float64)
        _gc_fo = sm.pixelize_catalog(_ra_gal_fo, _dec_gal_fo, nside)
        _rc_fo = sm.pixelize_catalog(_ra_rand_fo, _dec_rand_fo, nside)
        _delta_g_fo, _good_pix_fo = sm.compute_overdensity(_gc_fo, _rc_fo)
        _delta_t_fo = sm.assign_template_values(templates, _good_pix_fo)
        for _msh, _rsh in _all_mr_fo.items():
            if _rsh.get("sigma_hat") is None and _rsh.get("a_hat") is not None:
                _a_sh = np.asarray(_rsh["a_hat"])
                if _a_sh.shape == (n_sys,):
                    _rsh["sigma_hat"] = float(np.sqrt(
                        np.mean((_delta_g_fo - _a_sh @ _delta_t_fo) ** 2)
                    ))
        # Persist newly-computed sigma_hat to partial JSON files for non-MCMC methods
        import json as _json_sh
        for _msh_up, _rsh_up in _all_mr_fo.items():
            if _msh_up in ("MCMC-add", "MCMC-comb"):
                continue
            _sh_up = _rsh_up.get("sigma_hat")
            if _sh_up is None:
                continue
            _pjf_up = outdir / f"{sample_id}_NSIDE{nside:04d}_partial_{_msh_up}.json"
            if _pjf_up.exists():
                _pjd_up = _json_sh.loads(_pjf_up.read_text())
                if _pjd_up.get(_msh_up, {}).get("sigma_hat") is None:
                    _pjd_up.setdefault(_msh_up, {})["sigma_hat"] = _sh_up
                    _pjf_up.write_text(_json_sh.dumps(_pjd_up, indent=2))
        _regen_weight_figures(sample_id, nside, templates, _good_pix_fo,
                              _all_mr_fo, n_sys, n_pix, outdir, docs_dir)
        # Regenerate all-method FITS weight file
        _pix_gal_fo = hp.ang2pix(nside, np.radians(90 - _dec_gal_fo),
                                  np.radians(_ra_gal_fo))
        _fits_cols_fo = []
        for _mw_fo, _pk_fo, _cn_fo in [
            ("OLS",        "a_hat", "WEIGHT_OLS"),
            ("ElasticNet", "a_hat", "WEIGHT_ENET"),
            ("ISD-1",      "a_hat", "WEIGHT_ISD1"),
            ("ISD-3",      "a_hat", "WEIGHT_ISD3"),
            ("MCMC-add",   "a_hat", "WEIGHT_ADD"),
            ("MCMC-comb",  "b_hat", "WEIGHT_COMB"),
        ]:
            _rw_fo = _all_mr_fo.get(_mw_fo, {})
            _pw_fo = np.asarray(_rw_fo.get(_pk_fo, np.zeros(n_sys)))
            if _pw_fo.shape != (n_sys,):
                _pw_fo = np.zeros(n_sys)
            _cont_fo = np.einsum("i,ij->j", _pw_fo, templates)
            _wm_fo = np.ones(n_pix)
            _wm_fo[_good_pix_fo] = 1.0 / np.maximum(1.0 + _cont_fo[_good_pix_fo], 0.01)
            _wg_fo = np.where(_good_pix_fo[_pix_gal_fo], _wm_fo[_pix_gal_fo],
                               1.0).astype(np.float32)
            _fits_cols_fo.append(fits.Column(name=_cn_fo, format="E", array=_wg_fo))
        _fits_cols_fo.append(fits.Column(name="WEIGHT_SYS", format="E",
                                          array=_fits_cols_fo[-1].array.copy()))
        _fp_fo = outdir / f"{sample_id}_NSIDE{nside:04d}_WEIGHTS.fits"
        fits.BinTableHDU.from_columns(_fits_cols_fo).writeto(str(_fp_fo), overwrite=True)
        print(f"All-method weights (re)generated: {_fp_fo}")
        # Regenerate wtheta figure using TreeCorr (approximate: zero covariance)
        try:
            print(f"  Measuring w(θ) for {sample_id} …")
            _theta_fo, _w_obs_fo = sm.measure_two_point_function(
                _ra_gal_fo, _dec_gal_fo, _ra_rand_fo, _dec_rand_fo,
                min_sep=0.5, max_sep=300.0, nbins=30, sep_units="arcmin",
            )
            _METHOD_FO = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
            _delta_t_rot_fo, _R_fo, _ = sm.rotate_templates(_delta_t_fo)
            _lon_fo, _lat_fo = hp.pix2ang(nside, np.where(_good_pix_fo)[0], lonlat=True)
            import treecorr as _tc_fo
            _ct_fo = np.zeros((n_sys, len(_theta_fo)))
            for _i in range(n_sys):
                _cat_ti = _tc_fo.Catalog(ra=_lon_fo, dec=_lat_fo,
                                          ra_units="degrees", dec_units="degrees",
                                          k=_delta_t_rot_fo[_i])
                _kk_ti = _tc_fo.KKCorrelation(min_sep=0.5, max_sep=300.0, nbins=30,
                                               sep_units="arcmin", bin_slop=0.01)
                _kk_ti.process(_cat_ti)
                _ct_fo[_i] = _kk_ti.xi
            _all_wc_fo = {}
            for _mfo in _METHOD_FO:
                _rfo = _all_mr_fo.get(_mfo, {})
                _afo = np.asarray(_rfo.get("a_hat", np.zeros(n_sys)))
                _bfo = np.asarray(_rfo.get("b_hat", np.zeros(n_sys)))
                if len(_afo) != n_sys:
                    _afo = np.zeros(n_sys)
                    _bfo = np.zeros(n_sys)
                _ar_fo = _R_fo @ _afo
                _br_fo = _R_fo @ _bfo
                try:
                    _all_wc_fo[_mfo] = correct_two_point_function(
                        _w_obs_fo, _ar_fo, _br_fo,
                        np.zeros(n_sys), np.zeros(n_sys), _ct_fo)
                except Exception:
                    _all_wc_fo[_mfo] = np.full_like(_w_obs_fo, np.nan)
            _plot_wtheta_figure(_theta_fo, _w_obs_fo, _all_wc_fo,
                                sample_id, nside, len(_ra_gal_fo), outdir, docs_dir)
            import json as _json_wd
            _wdata_path = outdir / f"{sample_id}_NSIDE{nside:04d}_wtheta_data.json"
            _wdata_path.write_text(_json_wd.dumps({
                "theta_arcmin": _theta_fo.tolist(),
                "w_obs": _w_obs_fo.tolist(),
                "all_w_corr": {m: v.tolist() for m, v in _all_wc_fo.items()},
            }))
            print(f"  Saved w(θ) data → {_wdata_path.name}")
        except Exception as _exc_fo:
            print(f"  WARNING: wtheta figure failed ({_exc_fo}); skipping.")
        return {"sample_id": sample_id, "n_galaxies": len(_ra_gal_fo)}

    # ── Skip if already computed ──────────────────────────────────────────────
    if not force:
        import json as _json_skip
        _is_partial = "MCMC-comb" not in (set(only_methods) if only_methods else set())
        if _is_partial and only_methods:
            _slug_chk = "_".join(sorted(only_methods)).replace("/", "-")
            _pchk = outdir / f"{sample_id}_NSIDE{nside:04d}_partial_{_slug_chk}.json"
            if _pchk.exists():
                print(f"  {sample_id}: {_pchk.name} already exists — skipping (use --force to rerun).")
                return _json_skip.loads(_pchk.read_text())
        elif not _is_partial:
            _fchk = outdir / f"{sample_id}_NSIDE{nside:04d}_params.json"
            if _fchk.exists():
                print(f"  {sample_id}: {_fchk.name} already exists — skipping (use --force to rerun).")
                return _json_skip.loads(_fchk.read_text())

    # Load catalogs
    with fits.open(data_file, memmap=True) as hdul:
        cat = hdul[1].data
        ra_gal = np.asarray(cat["RA"], dtype=np.float64)
        dec_gal = np.asarray(cat["DEC"], dtype=np.float64)
        weight_col = None
        if "WEIGHT_COMP" in cat.names:
            weight_col = np.asarray(cat["WEIGHT_COMP"], dtype=np.float64)

    with fits.open(rand_file, memmap=True) as hdul:
        rand = hdul[1].data
        ra_rand = np.asarray(rand["RA"], dtype=np.float64)
        dec_rand = np.asarray(rand["DEC"], dtype=np.float64)

    print(f"Galaxies: {len(ra_gal):,}  Randoms: {len(ra_rand):,}")

    # Pixelise
    gal_counts = sm.pixelize_catalog(ra_gal, dec_gal, nside, weights=weight_col)
    rand_counts = sm.pixelize_catalog(ra_rand, dec_rand, nside)

    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    n_good = good_pix.sum()
    print(f"Unmasked pixels: {n_good:,} / {n_pix:,}  ({100*n_good/n_pix:.1f}%)")
    print(f"δ_g  mean={delta_g.mean():.4f}  std={delta_g.std():.4f}")

    # Template values at good pixels
    delta_t = sm.assign_template_values(templates, good_pix)

    # ── All-methods inference (fastest → slowest) ─────────────────────────
    _METHOD_ORDER = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
    _methods_to_run = set(_METHOD_ORDER) if only_methods is None else set(only_methods)
    _partial_mode   = "MCMC-comb" not in _methods_to_run

    # Pre-selection: run once, then pass pre-filtered delta_t to each method
    _presel_result = None
    _presel_isd    = None
    _presel_indices = None
    delta_t_decontam = delta_t  # may be replaced by filtered subset below

    if preselect:
        from sys_mapping.diagnostics import isd_template_significance
        from sys_mapping.model_selection import snr_preselect

        _z_min, _z_max = _parse_z_range(sample_id)
        _z_edges = np.array([_z_min, _z_max])
        _nz = np.array([float(len(ra_gal))])
        print(f"  Pre-selection: method={preselect_method}  "
              f"z=[{_z_min},{_z_max}]  n_mocks={preselect_n_mocks}")

        _pre = snr_preselect(delta_g, delta_t,
                             method=preselect_method,
                             n_top=preselect_n_top)
        _selected = list(_pre.selected_indices)

        if preselect_method == "isd" and preselect_p_threshold is not None:
            _isd_sig = isd_template_significance(
                delta_g, delta_t[_selected], good_pix, nside,
                n_total=0,
                n_total_footprint=int(len(ra_gal)),
                z_edges=_z_edges, nz=_nz,
                n_mocks=preselect_n_mocks, seed=0, rand_factor=2,
            )
            _keep = [s for s, p in zip(_selected, _isd_sig["p_values"])
                     if p <= preselect_p_threshold]
            _selected = _keep if _keep else _selected[:1]
            _presel_isd = _isd_sig
            print(f"    Selected {len(_selected)}/{n_sys} templates "
                  f"(p ≤ {preselect_p_threshold}): {_selected}")
        else:
            print(f"    Selected {len(_selected)}/{n_sys} templates: {_selected}")

        _presel_result   = _pre
        _presel_indices  = _selected
        delta_t_decontam = delta_t[_selected]

    all_method_results = {}
    for _meth in _METHOD_ORDER:
        if _meth not in _methods_to_run:
            continue
        print(f"  Running {_meth} ...")
        try:
            all_method_results[_meth] = sm.run_decontamination(
                _meth, delta_g, delta_t_decontam,
                n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
                seed=42, progress=_meth.startswith("MCMC"),
            )
            # Store pre-selection metadata in each method result
            if preselect:
                all_method_results[_meth]["preselect_indices"] = _presel_indices
                all_method_results[_meth]["preselect_result"]  = _presel_result
                all_method_results[_meth]["preselect_isd"]     = _presel_isd
            print(f"    {_meth}: {all_method_results[_meth]['elapsed_s']:.1f}s")
        except Exception as _exc:
            import warnings
            warnings.warn(f"{_meth} failed: {_exc}")
        # Compute sigma_hat from residuals for non-MCMC methods
        _n_decontam = delta_t_decontam.shape[0]
        _res_sh = all_method_results.get(_meth, {})
        if (_res_sh.get("sigma_hat") is None
                and _res_sh.get("a_hat") is not None):
            _a_sh = np.asarray(_res_sh["a_hat"])
            if _a_sh.shape == (_n_decontam,):
                _resid_sh = delta_g - _a_sh @ delta_t_decontam
                _res_sh["sigma_hat"] = float(np.sqrt(np.mean(_resid_sh ** 2)))

    # ── Partial-mode early return ─────────────────────────────────────────
    if _partial_mode:
        import datetime
        _slug = "_".join(sorted(_methods_to_run)).replace("/", "-")
        _partial: dict = {
            "schema_version": 1,
            "sample_id": sample_id,
            "methods_run": sorted(_methods_to_run),
            "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_sys": n_sys,
            "n_good": int(n_good),
            "n_galaxies": int(len(ra_gal)),
            "template_names": template_names,
        }
        for meth in sorted(_methods_to_run):
            res = all_method_results.get(meth, {})
            sub: dict = {"elapsed_s": res.get("elapsed_s")}
            a = res.get("a_hat")
            if a is not None:
                sub["a_hat"] = (a.tolist() if hasattr(a, "tolist") else list(a))
            cov_a = res.get("cov_a")
            if cov_a is not None:
                sub["var_a"] = np.diag(cov_a).tolist()
            if meth == "ElasticNet":
                sub["cv_alpha"]    = res.get("alpha")
                sub["cv_l1_ratio"] = res.get("l1_ratio")
            sub["sigma_hat"] = res.get("sigma_hat")
            if meth in ("MCMC-add", "MCMC-comb"):
                sub["acceptance_fraction"] = res.get("acceptance_fraction")
            _partial[meth] = sub
        _ppath = outdir / f"{sample_id}_NSIDE{nside:04d}_partial_{_slug}.json"
        import json as _json
        _ppath.write_text(_json.dumps(_partial, indent=2))
        print(f"  Partial results saved → {_ppath}")
        return _partial

    # ── Merge partial results from prior phases ───────────────────────────
    for _pf in sorted(outdir.glob(f"{sample_id}_NSIDE{nside:04d}_partial_*.json")):
        try:
            import json as _json2
            _pdata = _json2.loads(_pf.read_text())
            for _meth in _pdata.get("methods_run", []):
                if _meth not in all_method_results and _meth in _pdata:
                    _sub = _pdata[_meth]
                    _ah  = _sub.get("a_hat", [])
                    all_method_results[_meth] = {
                        "a_hat":     np.array(_ah) if _ah and len(_ah) == n_sys else np.zeros(n_sys),
                        "b_hat":     np.zeros(n_sys),
                        "elapsed_s": _sub.get("elapsed_s"),
                    }
        except Exception as _exc:
            import warnings
            warnings.warn(f"Could not load partial file {_pf}: {_exc}")

    res_add  = all_method_results.get("MCMC-add", {})
    res_comb = all_method_results.get("MCMC-comb", {})

    a_hat_add  = np.asarray(res_add.get("a_hat", np.zeros(n_sys)))
    b_hat_comb = np.asarray(res_comb.get("b_hat", np.zeros(n_sys)))
    var_a_add  = np.diag(res_add.get("cov_a", np.eye(n_sys)))
    var_b_comb = np.diag(res_comb.get("cov_b", np.eye(n_sys)))
    acc_add    = float(res_add.get("acceptance_fraction", float("nan")))
    acc_comb   = float(res_comb.get("acceptance_fraction", float("nan")))
    print(f"Acceptance fraction — add: {acc_add:.3f}  comb: {acc_comb:.3f}")

    # PCA-rotated templates for 2PCF correction (use R from MCMC-comb, deterministic)
    _R = res_comb.get("R")
    if _R is not None:
        delta_t_rot = _R @ delta_t
    else:
        from sys_mapping.correction import rotate_templates as _rt
        delta_t_rot, _R, _ = _rt(delta_t)

    # ── LRT ───────────────────────────────────────────────────────────────
    if res_add.get("flat_chain") is not None and res_comb.get("flat_chain") is not None:
        theta_add  = sm.get_mle_params(res_add["flat_chain"])
        theta_comb = sm.get_mle_params(res_comb["flat_chain"])
        lrt = sm.likelihood_ratio_test(
            delta_g, delta_t_rot, theta_add, theta_comb,
            null_model="additive", alt_model="combined", significance=0.05,
        )
        print(f"\nLRT  λ_LR={lrt.lambda_lr:.2f}  p={lrt.p_value:.4f}  reject_null={lrt.reject_null}")
    else:
        from collections import namedtuple
        _LRTResult = namedtuple("LRTResult", ["lambda_lr", "p_value", "n_dof", "reject_null"])
        lrt = _LRTResult(lambda_lr=float("nan"), p_value=float("nan"), n_dof=0, reject_null=False)

    # ── w(θ) ──────────────────────────────────────────────────────────────
    print("\nMeasuring w(θ) …")
    theta_arcmin, w_obs = sm.measure_two_point_function(
        ra_gal, dec_gal, ra_rand, dec_rand,
        min_sep=0.5, max_sep=300.0, nbins=30, sep_units="arcmin",
    )

    # Template 2PCFs in rotated basis
    lon_obs, lat_obs = hp.pix2ang(nside, np.where(good_pix)[0], lonlat=True)
    try:
        import treecorr
        ct_rot = np.zeros((n_sys, len(theta_arcmin)))
        for i in range(n_sys):
            cat_t = treecorr.Catalog(ra=lon_obs, dec=lat_obs,
                                      ra_units="degrees", dec_units="degrees",
                                      k=delta_t_rot[i])
            kk = treecorr.KKCorrelation(min_sep=0.5, max_sep=300.0, nbins=30,
                                         sep_units="arcmin", bin_slop=0.01)
            kk.process(cat_t)
            ct_rot[i] = kk.xi
    except Exception as e:
        print(f"WARNING: TreeCorr template 2PCF failed ({e}); using zeros.")
        ct_rot = np.zeros((n_sys, len(theta_arcmin)))

    # Two-point correction (combined model, rotated basis)
    w_corr_comb = correct_two_point_function(
        w_obs, np.asarray(res_comb.get("a_rot", np.zeros(n_sys))),
        np.asarray(res_comb.get("b_rot", np.zeros(n_sys))),
        np.diag(res_comb.get("cov_a_rot", np.eye(n_sys))),
        np.diag(res_comb.get("cov_b_rot", np.eye(n_sys))), ct_rot,
    )
    # Additive model correction
    w_corr_add = correct_two_point_function(
        w_obs, np.asarray(res_add.get("a_rot", np.zeros(n_sys))),
        np.zeros(n_sys),
        np.diag(res_add.get("cov_a_rot", np.eye(n_sys))),
        np.zeros(n_sys), ct_rot,
    )

    # ── Per-galaxy weights: all 6 methods ─────────────────────────────────
    _METHOD_COL_W = [
        ("OLS",        "a_hat", "WEIGHT_OLS"),
        ("ElasticNet", "a_hat", "WEIGHT_ENET"),
        ("ISD-1",      "a_hat", "WEIGHT_ISD1"),
        ("ISD-3",      "a_hat", "WEIGHT_ISD3"),
        ("MCMC-add",   "a_hat", "WEIGHT_ADD"),
        ("MCMC-comb",  "b_hat", "WEIGHT_COMB"),
    ]
    pix_gal = hp.ang2pix(nside, np.radians(90 - dec_gal), np.radians(ra_gal))
    fits_cols = []
    for _mw, _pk, _cn in _METHOD_COL_W:
        _rw = all_method_results.get(_mw, {})
        _pw = np.asarray(_rw.get(_pk, np.zeros(n_sys)))
        if _pw.shape != (n_sys,):
            _pw = np.zeros(n_sys)
        _cont = np.einsum("i,ij->j", _pw, templates)
        _wm = np.ones(n_pix)
        _wm[good_pix] = 1.0 / np.maximum(1.0 + _cont[good_pix], 0.01)
        _wg = np.where(good_pix[pix_gal], _wm[pix_gal], 1.0).astype(np.float32)
        fits_cols.append(fits.Column(name=_cn, format="E", array=_wg))
    fits_cols.append(fits.Column(name="WEIGHT_SYS", format="E",
                                 array=fits_cols[-1].array.copy()))

    def _safe_float(v, fallback=-999.0):
        f = float(v)
        return fallback if not np.isfinite(f) else f

    fits_path = outdir / f"{sample_id}_NSIDE{nside:04d}_WEIGHTS.fits"
    _hdr = fits.Header()
    _hdr["SAMPLE"]  = sample_id[:68]
    _hdr["NSIDE"]   = nside
    _hdr["N_SYS"]   = n_sys
    _hdr["LRT_LAM"] = _safe_float(lrt.lambda_lr)
    _hdr["LRT_P"]   = _safe_float(lrt.p_value)
    _hdr["LRT_REJ"] = bool(lrt.reject_null)
    _hdr["W_SYS"]   = "alias for WEIGHT_COMB - recommended default"
    fits.BinTableHDU.from_columns(fits_cols, header=_hdr).writeto(str(fits_path), overwrite=True)
    print(f"All-method weights saved: {fits_path}")

    # ── Save JSON params ──────────────────────────────────────────────────
    json_path = outdir / f"{sample_id}_NSIDE{nside:04d}_params.json"
    params_dict = {
        "sample_id": sample_id,
        "nside": nside,
        "n_sys": n_sys,
        "template_names": template_names,
        "a_hat_add": a_hat_add.tolist(),
        "b_hat_comb": b_hat_comb.tolist(),
        "var_a_add": var_a_add.tolist(),
        "var_b_comb": var_b_comb.tolist(),
        "lrt": {
            "lambda_lr": float(lrt.lambda_lr),
            "p_value": float(lrt.p_value),
            "n_dof": int(lrt.n_dof),
            "reject_null": bool(lrt.reject_null),
        },
        "acceptance_fraction_add": float(acc_add),
        "acceptance_fraction_comb": float(acc_comb),
        "sigma_hat_add": _safe_float(res_add.get("sigma_hat", float("nan"))),
        "sigma_hat_comb": _safe_float(res_comb.get("sigma_hat", float("nan"))),
        "n_galaxies": int(len(ra_gal)),
        "n_good_pix": int(n_good),
    }
    json_path.write_text(json.dumps(params_dict, indent=2))
    print(f"Params saved: {json_path}")

    # ── Compute corrected 2PCF for all 6 methods ─────────────────────────
    _METHOD_ORDER_LS10 = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
    _R_ls10 = np.asarray(res_comb.get("R", np.eye(n_sys)))
    _all_w_corr_ls10 = {}
    for _mls in _METHOD_ORDER_LS10:
        _rls = all_method_results.get(_mls, {})
        _a_ls = np.asarray(_rls.get("a_hat", np.zeros(n_sys)))
        if _a_ls.shape != (n_sys,):
            _a_ls = np.zeros(n_sys)
        _b_ls = np.asarray(_rls.get("b_hat", np.zeros(n_sys)))
        if _b_ls.shape != (n_sys,):
            _b_ls = np.zeros(n_sys)
        if _mls in ("MCMC-add", "MCMC-comb") and _rls.get("a_rot") is not None:
            _ar_ls = np.asarray(_rls["a_rot"])
            _b_rot = _rls.get("b_rot")
            _br_ls = np.asarray(_b_rot if _b_rot is not None else np.zeros(n_sys))
            _va_ls = np.diag(np.asarray(_rls["cov_a_rot"])) if _rls.get("cov_a_rot") is not None else np.zeros(n_sys)
            _vb_ls = np.diag(np.asarray(_rls["cov_b_rot"])) if _rls.get("cov_b_rot") is not None else np.zeros(n_sys)
        else:
            _ar_ls = _R_ls10 @ _a_ls
            _br_ls = _R_ls10 @ _b_ls
            _va_ls = np.zeros(n_sys)
            _vb_ls = np.zeros(n_sys)
        try:
            _all_w_corr_ls10[_mls] = correct_two_point_function(
                w_obs, _ar_ls, _br_ls, _va_ls, _vb_ls, ct_rot)
        except Exception:
            _all_w_corr_ls10[_mls] = np.full_like(w_obs, np.nan)

    # ── Diagnostic plot: all 6 methods (shared helper) ────────────────────
    _plot_wtheta_figure(theta_arcmin, w_obs, _all_w_corr_ls10,
                        sample_id, nside, len(ra_gal), outdir, docs_dir)

    # ── Weight maps and histograms: all 6 methods ────────────────────────
    _regen_weight_figures(sample_id, nside, templates, good_pix,
                          all_method_results, n_sys, n_pix, outdir, docs_dir)

    return {
        "sample_id": sample_id,
        "data_file": str(data_file),
        "rand_file": str(rand_file),
        "weights_file": str(fits_path),
        "params_file": str(json_path),
    }


# ── Summary YAML ───────────────────────────────────────────────────────────

def write_summary(entries, nside, output_dir):
    summary_path = Path(output_dir) / f"summary_NSIDE{nside:04d}.yaml"
    with open(summary_path, "w") as f:
        yaml.dump({"nside": nside, "samples": entries}, f, default_flow_style=False)
    print(f"\nSummary written: {summary_path}")


# ── Incremental RST documentation update ──────────────────────────────────────

def _format_method_stats_table(sample_id, nside, output_dir):
    """Return an RST csv-table block with per-method goodness-of-fit statistics."""
    import math as _math
    out = Path(output_dir)
    slug = f"{sample_id}_NSIDE{nside:04d}"
    METHOD_ORDER = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
    rows = []
    pfile = out / f"{slug}_params.json"
    pdata = json.loads(pfile.read_text()) if pfile.exists() else {}
    n_good = pdata.get("n_good_pix")
    if not n_good:
        # Older params.json files lack n_good_pix; read from any partial JSON
        for _mf in METHOD_ORDER:
            _pjf = out / f"{slug}_partial_{_mf}.json"
            if _pjf.exists():
                _ng = json.loads(_pjf.read_text()).get("n_good")
                if _ng:
                    n_good = int(_ng)
                    break
    for meth in METHOD_ORDER:
        sub = {}
        pjson = out / f"{slug}_partial_{meth}.json"
        if pjson.exists():
            sub = json.loads(pjson.read_text()).get(meth, {})
        if meth == "MCMC-comb" and not sub.get("sigma_hat") and pdata:
            _cmb = pdata.get("combined_model", {})
            sub = {
                "sigma_hat": (pdata.get("sigma_hat_comb") or _cmb.get("sigma_hat")),
                "acceptance_fraction": (pdata.get("acceptance_fraction_comb")
                                        or _cmb.get("acceptance_fraction")),
            }
        elif meth == "MCMC-add" and not sub.get("sigma_hat") and pdata:
            if not sub:
                _add = pdata.get("additive_model", {})
                sub = {
                    "sigma_hat": (pdata.get("sigma_hat_add") or _add.get("sigma_hat")),
                    "acceptance_fraction": (pdata.get("acceptance_fraction_add")
                                            or _add.get("acceptance_fraction")),
                }
        sigma = sub.get("sigma_hat")
        acc   = sub.get("acceptance_fraction")
        elaps = sub.get("elapsed_s")
        if sigma is not None and n_good and sigma > 0:
            ln_L = -n_good / 2.0 * (1.0 + _math.log(2.0 * _math.pi) + 2.0 * _math.log(sigma))
            ln_L_str = f"{ln_L:.1f}"
        else:
            ln_L_str = "—"
        rows.append((
            meth,
            f"{sigma:.4f}" if sigma is not None else "—",
            ln_L_str,
            f"{acc:.3f}"   if acc is not None else "—",
            f"{elaps:.0f}" if elaps is not None else "—",
        ))
    lrt_str = "—"
    lrt = pdata.get("lrt", {})
    lam = lrt.get("lambda_lr")
    pv  = lrt.get("p_value")
    dof = lrt.get("n_dof")
    rej = lrt.get("reject_null") or lrt.get("reject_additive")
    if lam is not None and lam == lam and np.isfinite(float(lam)):
        lrt_str = (f"λ={float(lam):.1f}  p={float(pv):.2e}"
                   f"  dof={dof}  reject={'yes' if rej else 'no'}")
    lines = [
        ".. csv-table:: Goodness-of-fit statistics",
        '   :header: "Method", "σ_hat", "ln L", "acc_frac", "elapsed (s)"',
        "   :widths: 14, 12, 14, 12, 12",
        "",
    ]
    for row in rows:
        lines.append(f'   "{row[0]}", "{row[1]}", "{row[2]}", "{row[3]}", "{row[4]}"')
    lines.append(f'   "LRT (add vs. comb)", "{lrt_str}", "", "", ""')
    lines.append("")
    return "\n".join(lines)


def update_ls10_rst(method_tag: str, entries: list,
                    output_dir: str = None, nside: int = 64) -> None:
    """Append or replace a sentinel-bounded section in results_ls10.rst.

    Called after each method phase to make fast results immediately browsable.
    """
    import re
    import datetime

    rst_path = Path(__file__).resolve().parent.parent / "docs" / "results_ls10.rst"
    if not rst_path.exists():
        return
    if not entries:
        return

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    tag_safe = method_tag.replace("-", "_").replace("+", "_")
    start_sentinel = f".. _auto-ls10-{tag_safe}-start:"
    end_sentinel   = f".. _auto-ls10-{tag_safe}-end:"

    # Build rows
    rows: list[str] = []
    for e in entries:
        sid  = e.get("sample_id", "?")
        sub  = e.get(method_tag, {})
        a_hat = sub.get("a_hat", [])
        names = e.get("template_names", [])
        elapsed = sub.get("elapsed_s") or 0.0

        if a_hat and names:
            best = int(max(range(len(a_hat)), key=lambda i: abs(a_hat[i])))
            top  = names[best] if best < len(names) else "—"
            ah   = float(a_hat[best])
        else:
            top, ah = "—", float("nan")

        if method_tag == "OLS":
            var_a = sub.get("var_a", [])
            sig = sub.get("sigma", float("nan"))
            sa  = float(var_a[best]**0.5) if var_a and best < len(var_a) else float("nan")
            rows.append(f'   "{sid[:40]}", "{top[:30]}", "{ah:+.4f}", "{sa:.4f}", "{sig:.4f}", "{elapsed:.1f}"')
        elif method_tag in ("ElasticNet", "ISD-1", "ISD-3"):
            alpha = sub.get("cv_alpha", float("nan"))
            l1r   = sub.get("cv_l1_ratio", float("nan"))
            n_nz  = sum(1 for a in a_hat if abs(a) > 1e-8) if a_hat else 0
            rows.append(f'   "{sid[:40]}", "{top[:30]}", "{ah:+.4f}", "{alpha or float("nan"):.4f}", "{l1r or float("nan"):.2f}", "{n_nz}", "{elapsed:.1f}"')
        elif method_tag == "MCMC-add":
            acc = sub.get("acceptance_fraction", float("nan"))
            sig = sub.get("sigma_hat", float("nan"))
            el_h = elapsed / 3600
            rows.append(f'   "{sid[:40]}", "{acc:.3f}", "{sig:.4f}", "{top[:30]}", "{ah:+.4f}", "{el_h:.2f}"')
        else:  # MCMC-comb (full results from run_sample)
            acc_a = e.get("acceptance_fraction_add", float("nan"))
            acc_c = e.get("acceptance_fraction_comb", float("nan"))
            lrt_l = e.get("lrt", {}).get("lambda_lr", float("nan"))
            lrt_p = e.get("lrt", {}).get("p_value", float("nan"))
            rej   = e.get("lrt", {}).get("reject_null", False)
            n_gal = e.get("n_galaxies", 0)
            el_h  = elapsed / 3600
            # In figures-only mode, entries carry no MCMC data — read from params.json
            if np.isnan(float(acc_a)) and output_dir is not None:
                _pf_mc = Path(output_dir) / f"{sid}_NSIDE{nside:04d}_params.json"
                if _pf_mc.exists():
                    _pd_mc = json.loads(_pf_mc.read_text())
                    # Support both new schema (top-level keys) and old schema (sub-dicts)
                    _add_mc = _pd_mc.get("additive_model", {})
                    _cmb_mc = _pd_mc.get("combined_model", {})
                    acc_a = (_pd_mc.get("acceptance_fraction_add")
                             or _add_mc.get("acceptance_fraction") or acc_a)
                    acc_c = (_pd_mc.get("acceptance_fraction_comb")
                             or _cmb_mc.get("acceptance_fraction") or acc_c)
                    _lrt_mc = _pd_mc.get("lrt", {})
                    lrt_l = _lrt_mc.get("lambda_lr", lrt_l)
                    lrt_p = _lrt_mc.get("p_value", lrt_p)
                    rej   = (_lrt_mc.get("reject_null")
                             or _lrt_mc.get("reject_additive") or rej)
                    if not n_gal:
                        n_gal = _pd_mc.get("n_galaxies", 0)
            rows.append(
                f'   "{sid[:40]}", "{n_gal:,}", "{acc_a:.3f}", "{acc_c:.3f}", '
                f'"{lrt_l:.1f}", "{lrt_p:.4f}", "{rej}", "{el_h:.2f}"'
            )

    # Build header
    if method_tag == "OLS":
        header = '"Sample", "Top template", "a_hat", "sigma_a", "sigma_OLS", "t (s)"'
        widths = "40, 30, 10, 10, 12, 8"
    elif method_tag in ("ElasticNet", "ISD-1", "ISD-3"):
        header = '"Sample", "Top template", "a_hat", "CV alpha", "CV l1_ratio", "N non-zero", "t (s)"'
        widths = "40, 30, 10, 12, 12, 12, 8"
    elif method_tag == "MCMC-add":
        header = '"Sample", "acc", "sigma_hat", "Top template", "a_hat", "t (h)"'
        widths = "40, 10, 12, 30, 10, 10"
    else:
        header = '"Sample", "N_gal", "acc_add", "acc_comb", "lambda_LR", "p-val", "reject", "t (h)"'
        widths = "40, 12, 10, 10, 10, 8, 8, 8"

    lines_in_section = [
        "",
        f"{method_tag} Results — LS10 BGS (auto-generated {ts})",
        "-" * (len(f"{method_tag} Results — LS10 BGS (auto-generated {ts})") + 2),
        "",
        f"Samples processed: {len(entries)}",
        "",
        f".. csv-table:: LS10 {method_tag} estimates per sample",
        f"   :header: {header}",
        f"   :widths: {widths}",
        "",
    ] + rows + [""]

    new_section = (
        "\n" + start_sentinel + "\n"
        + "\n".join(lines_in_section) + "\n"
        + end_sentinel + "\n"
    )

    text = rst_path.read_text()
    pattern = re.compile(
        re.escape(start_sentinel) + r".*?" + re.escape(end_sentinel),
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(new_section.strip(), text)
    else:
        text = text.rstrip("\n") + "\n" + new_section

    rst_path.write_text(text)
    print(f"  Updated {rst_path.name} — section [{method_tag}]")

    # When MCMC-comb finishes, also write per-sample figure blocks
    if method_tag == "MCMC-comb":
        _update_ls10_comparison_figures(entries, rst_path, output_dir, nside)


def _update_ls10_comparison_figures(entries: list, rst_path,
                                    output_dir=None, nside: int = 64) -> None:
    """Write per-sample w(θ), weight-map, weight-histogram image blocks + stats table."""
    import re
    import datetime

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    start_sentinel = ".. _auto-ls10-figures-start:"
    end_sentinel   = ".. _auto-ls10-figures-end:"

    def _fig_block(sample_id, nside_val, suffix, caption):
        fname = f"{sample_id}_NSIDE{nside_val:04d}_{suffix}.png"
        return (
            f"\n.. raw:: html\n\n"
            f"   <figure style=\"text-align:center;margin:1em 0;\">\n"
            f"     <a href=\"_static/results_ls10/{fname}\" target=\"_blank\">\n"
            f"       <img src=\"_static/results_ls10/{fname}\"\n"
            f"            style=\"width:95%;max-width:1100px;\"\n"
            f"            alt=\"{caption}\">\n"
            f"     </a>\n"
            f"     <figcaption style=\"font-size:0.87em;color:#555;margin-top:0.3em;\">\n"
            f"       {caption}  Click to enlarge.\n"
            f"     </figcaption>\n"
            f"   </figure>\n"
        )

    lines = [
        "",
        f"Per-sample comparison figures (auto-generated {ts})",
        "-" * 56,
        "",
        "Each sample shows a goodness-of-fit table followed by three figures:",
        "",
        "* **w(θ)**: log-log angular correlation before/after correction (all 6 methods).",
        "* **Weight maps**: 2×3 Mollweide projection of per-pixel weights for all 6 methods.",
        "* **Weight histograms**: log-scale distribution of per-pixel weights.",
        "",
        ".. note::",
        "",
        "   ISD-3 and (for some samples) ISD-1 weights may be unreliable — see method notes above.",
        "",
    ]

    for e in entries:
        sid = e.get("sample_id", "?")
        n_gal = e.get("n_galaxies", 0)
        label = sid.replace("LS10_VLIM_ANY_", "log M* ≥ ").replace("_Mstar_12.0", "").split("_N_")[0]
        lines.append(f"**{label}**  (N_gal = {n_gal:,})")
        lines.append("")
        if output_dir is not None and sid != "?":
            try:
                lines.append(_format_method_stats_table(sid, nside, output_dir))
            except Exception:
                pass
        lines.append(_fig_block(sid, nside, "wtheta",
            f"w(θ) before/after correction — {label}"))
        lines.append(_fig_block(sid, nside, "weight_map",
            f"Weight maps (6 methods) — {label}"))
        lines.append(_fig_block(sid, nside, "weight_hist",
            f"Weight histograms (6 methods) — {label}"))
        lines.append("")

    new_section = (
        "\n" + start_sentinel + "\n"
        + "\n".join(lines) + "\n"
        + end_sentinel + "\n"
    )

    text = rst_path.read_text()
    pattern = re.compile(
        re.escape(start_sentinel) + r".*?" + re.escape(end_sentinel),
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(new_section.strip(), text)
    else:
        text = text.rstrip("\n") + "\n" + new_section

    rst_path.write_text(text)
    print(f"  Updated {rst_path.name} — per-sample figure blocks")


# ── All-samples w(θ) overview grid ────────────────────────────────────────

def _plot_all_samples_wtheta_grid(entries, output_dir, nside, docs_dir):
    """3×3 grid of per-sample w(θ) figures showing all 6 decontamination methods.

    Reads per-sample wtheta_data.json files written during --figures-only runs
    and produces a single overview figure with observed + all corrected curves.
    """
    import json as _json_grid
    import matplotlib.pyplot as _plt_grid
    import numpy as _np_grid

    outdir = Path(output_dir)
    _METHOD_ORDER = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]

    # Collect data for each sample (sorted by stellar-mass threshold = natural order)
    _samples_data = []
    for _entry in entries:
        _sid = _entry.get("sample_id", "")
        _wdata_path = outdir / f"{_sid}_NSIDE{nside:04d}_wtheta_data.json"
        if not _wdata_path.exists():
            continue
        _wd = _json_grid.loads(_wdata_path.read_text())
        _samples_data.append((_sid, _wd))

    if not _samples_data:
        print("  No wtheta_data.json files found; skipping overview grid.")
        return

    _n = len(_samples_data)
    _ncols = 3
    _nrows = (_n + _ncols - 1) // _ncols

    fig, axes = _plt_grid.subplots(
        _nrows * 2, _ncols,
        figsize=(5 * _ncols, 3.2 * _nrows * 2),
        gridspec_kw={"height_ratios": [2, 1] * _nrows},
    )
    # axes layout: for row i, top=axes[2i], bottom=axes[2i+1]

    for _idx, (_sid, _wd) in enumerate(_samples_data):
        _row, _col = divmod(_idx, _ncols)
        _ax_top = axes[_row * 2, _col]
        _ax_bot = axes[_row * 2 + 1, _col]

        _theta = _np_grid.array(_wd["theta_arcmin"])
        _w_obs = _np_grid.array(_wd["w_obs"])
        _w_sig = (_w_obs > 0) & _np_grid.isfinite(_w_obs)
        _w_ref = _np_grid.where(_np_grid.abs(_w_obs) > 1e-8, _w_obs, _np_grid.nan)

        # Short label from sample_id
        try:
            _mstar = _sid.split("_ANY_")[1].split("_")[0]
            _zmax  = _sid.split("_z_")[1].split("_")[0]
            _label = f"log M*≥{_mstar},  z<{_zmax}"
        except Exception:
            _label = _sid

        # Top panel: w(θ) log-log
        _ax_top.set_xscale("log")
        _ax_top.set_yscale("log")
        _valid = _w_sig
        _ax_top.plot(_theta[_valid], _w_obs[_valid],
                     color=METHOD_COLORS.get("observed", "k"),
                     ls="-.", lw=2.5, label="Observed", zorder=10)
        for _m in _METHOD_ORDER:
            _wc = _wd["all_w_corr"].get(_m)
            if _wc is None:
                continue
            _wc = _np_grid.array(_wc)
            _ok = _valid & _np_grid.isfinite(_wc) & (_wc > 0)
            if not _ok.any():
                continue
            _ax_top.plot(_theta[_ok], _wc[_ok],
                         color=METHOD_COLORS.get(_m, "grey"),
                         ls=METHOD_LINESTYLES.get(_m, "-"), lw=1.3,
                         label=METHOD_LABELS.get(_m, _m))
        _ax_top.set_title(_label, fontsize=8)
        _ax_top.set_ylabel(r"$w(\theta)$", fontsize=7)
        _ax_top.tick_params(labelsize=6)
        _ax_top.set_xticklabels([])

        # Bottom panel: fractional correction δw/w
        _ax_bot.set_xscale("log")
        _ax_bot.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
        for _m in _METHOD_ORDER:
            _wc = _wd["all_w_corr"].get(_m)
            if _wc is None:
                continue
            _wc = _np_grid.array(_wc)
            _frac = (_wc - _w_obs) / _w_ref
            _frac[~_w_sig] = _np_grid.nan
            _ax_bot.plot(_theta, _frac,
                         color=METHOD_COLORS.get(_m, "grey"),
                         ls=METHOD_LINESTYLES.get(_m, "-"), lw=1.2,
                         label=METHOD_LABELS.get(_m, _m))
        _ax_bot.set_xlabel(r"$\theta$ [arcmin]", fontsize=7)
        _ax_bot.set_ylabel(r"$\delta w/w_{\rm obs}$", fontsize=7)
        _ax_bot.tick_params(labelsize=6)

    # Hide unused panels (if _n < _nrows * _ncols)
    for _idx in range(_n, _nrows * _ncols):
        _row, _col = divmod(_idx, _ncols)
        axes[_row * 2, _col].set_visible(False)
        axes[_row * 2 + 1, _col].set_visible(False)

    # Single shared legend from first panel
    _handles, _labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(_handles, _labels, loc="lower center",
               ncol=min(7, len(_handles)), fontsize=7,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "LS10 BGS VLIM — angular clustering before/after systematic correction (NSIDE=64, all 6 methods)",
        fontsize=9, y=1.001,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    _out = docs_dir / "wtheta_all_methods_nside64.png"
    fig.savefig(str(_out), dpi=150, bbox_inches="tight")
    _plt_grid.close(fig)
    print(f"\nAll-samples wtheta grid → {_out}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full sys_mapping pipeline on LS10 BGS galaxy catalogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--catalog-dir", required=True,
                        help="Directory containing *_DATA.fits and *_RAND.fits pairs.")
    parser.add_argument("--sample", default=None,
                        help="Process only this sample (omit _DATA.fits suffix).")
    parser.add_argument("--template-dir", default=None,
                        help="Directory of HEALPix systematic FITS files. "
                             "Uses 5 synthetic families if not set.")
    parser.add_argument("--nside", type=int, default=64, help="HEALPix NSIDE.")
    parser.add_argument("--n-walkers", type=int, default=210, help="MCMC walkers.")
    parser.add_argument("--n-steps", type=int, default=1500, help="MCMC steps.")
    parser.add_argument("--n-burn", type=int, default=300, help="MCMC burn-in.")
    parser.add_argument("--output-dir", default="data/sys_weights/",
                        help="Output directory for weights, params, and plots.")
    parser.add_argument("--only-methods", nargs="+", metavar="METHOD", default=None,
                        help="Run only these methods (e.g. --only-methods OLS ElasticNet). "
                             "Default: all methods. When MCMC-comb is absent only partial "
                             "results are saved and the RST is updated with a summary.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run samples even if partial/full result JSON already exists.")
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate weight-map and histogram figures from saved JSON "
                             "without re-running MCMC.  Requires existing params.json / "
                             "partial JSON files.")
    # Pre-selection
    parser.add_argument("--preselect", action="store_true",
                        help="Run ISD-based SNR pre-selection before decontamination.")
    parser.add_argument("--preselect-method", default="isd",
                        choices=["data", "template", "isd"],
                        help="SNR ranking method for pre-selection (default: isd).")
    parser.add_argument("--preselect-n-top", type=int, default=None,
                        help="Keep only the top-K templates; None keeps all passing the "
                             "p-value threshold.")
    parser.add_argument("--preselect-p-threshold", type=float, default=0.05,
                        help="ISD mock p-value threshold for template inclusion (default 0.05).")
    parser.add_argument("--preselect-n-mocks", type=int, default=100,
                        help="Number of GLASS mocks for ISD significance (default 100).")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir)

    # Load templates once
    if args.template_dir:
        templates, template_names = load_templates_from_dir(args.template_dir, args.nside)
        if templates is None:
            print(f"No FITS files in {args.template_dir}; using synthetic templates.")
            templates, template_names = synthetic_templates(args.nside)
        else:
            print(f"Loaded {templates.shape[0]} external templates: {template_names}")
    else:
        templates, template_names = synthetic_templates(args.nside)
        print(f"Using {templates.shape[0]} synthetic template families.")

    print(f"Only-methods   : {args.only_methods or 'all'}")

    # Discover samples
    if args.sample:
        data_files = list(catalog_dir.glob(f"{args.sample}_DATA.fits"))
    else:
        data_files = sorted(catalog_dir.glob("*_DATA.fits"))

    if not data_files:
        print(f"ERROR: no *_DATA.fits files found in {catalog_dir}")
        return

    entries = []
    for data_file in data_files:
        sample_id = data_file.name.replace("_DATA.fits", "")
        rand_file = data_file.parent / f"{sample_id}_RAND.fits"
        if not rand_file.exists():
            print(f"WARNING: random file not found for {sample_id}, skipping.")
            continue
        entry = run_sample(
            sample_id, data_file, rand_file, templates, template_names,
            args.nside, args.n_walkers, args.n_steps, args.n_burn, args.output_dir, args,
            only_methods=args.only_methods,
            force=args.force,
            figures_only=args.figures_only,
            preselect=args.preselect,
            preselect_method=args.preselect_method,
            preselect_n_top=args.preselect_n_top,
            preselect_p_threshold=args.preselect_p_threshold,
            preselect_n_mocks=args.preselect_n_mocks,
        )
        entries.append(entry)

    # Generate all-samples w(θ) overview grid (figures-only mode only)
    if args.figures_only and entries:
        _plot_all_samples_wtheta_grid(entries, args.output_dir, args.nside, _DOCS_STATIC_LS10)

    # Skip RST auto-update in figures-only mode to avoid re-contaminating
    # the manually maintained results_ls10.rst.
    if not args.figures_only and entries:
        if args.only_methods:
            _tag = (args.only_methods[0]
                    if len(args.only_methods) == 1
                    else "+".join(sorted(args.only_methods)))
            print(f"\nUpdating results_ls10.rst for [{_tag}] ...")
            update_ls10_rst(_tag, entries, args.output_dir, args.nside)
        else:
            # Full run — update MCMC-comb section
            update_ls10_rst("MCMC-comb", entries, args.output_dir, args.nside)

    _ran_full = "MCMC-comb" in (args.only_methods or ["MCMC-comb"])
    if _ran_full and entries:
        write_summary(entries, args.nside, args.output_dir)


if __name__ == "__main__":
    main()
