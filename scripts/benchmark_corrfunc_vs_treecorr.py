#!/usr/bin/env python3
"""Benchmark TreeCorr vs Corrfunc two-point estimators.

Tests three estimator types:
  1. NN Landy-Szalay  (measure_two_point_function)
  2. KK auto-correlation  (measure_kk_correlation)
  3. KK cross-correlation (measure_kk_correlation)

Across three dataset sizes (small / medium / large).
Reports wall-clock time and the RMS residual between the two results.

Usage
-----
    /path/to/python benchmark_corrfunc_vs_treecorr.py
"""

import time
import textwrap

import numpy as np

# Use the sys_map package from the repo root
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sys_mapping.utils import (
    measure_two_point_function,
    measure_two_point_function_corrfunc,
    measure_kk_correlation_treecorr,
    measure_kk_correlation_corrfunc,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _timer(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _rms(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(diff ** 2)))


def _header(title):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def _row(label, t_tc, t_cf, rms_theta, rms_xi):
    speedup = t_tc / t_cf if t_cf > 0 else float("inf")
    print(
        f"  {label:<22s}  treecorr {t_tc:6.2f}s  corrfunc {t_cf:6.2f}s"
        f"  x{speedup:4.1f}  rms_xi={rms_xi:.2e}"
    )


# ---------------------------------------------------------------------------
# dataset factory
# ---------------------------------------------------------------------------

def make_galaxy_catalog(n_gal, n_rand, rng, ra_range=(30.0, 60.0), dec_range=(-10.0, 10.0)):
    ra_gal  = rng.uniform(*ra_range,  n_gal ).astype(np.float64)
    dec_gal = rng.uniform(*dec_range, n_gal ).astype(np.float64)
    ra_rand  = rng.uniform(*ra_range,  n_rand).astype(np.float64)
    dec_rand = rng.uniform(*dec_range, n_rand).astype(np.float64)
    return ra_gal, dec_gal, ra_rand, dec_rand


def make_pixel_catalog(n_pix, rng, ra_range=(30.0, 60.0), dec_range=(-10.0, 10.0)):
    ra  = rng.uniform(*ra_range,  n_pix).astype(np.float64)
    dec = rng.uniform(*dec_range, n_pix).astype(np.float64)
    k   = rng.standard_normal(n_pix).astype(np.float64)
    w   = rng.uniform(0.5, 1.5, n_pix).astype(np.float64)
    return ra, dec, k, w


# ---------------------------------------------------------------------------
# benchmark scenarios
# ---------------------------------------------------------------------------

SIZES = [
    ("small",  5_000,  25_000,  10_000),   # (label, n_gal, n_rand, n_pix)
    ("medium", 20_000, 100_000, 40_000),
    ("large",  80_000, 400_000, 160_000),
]

NN_KW   = dict(min_sep=1.0,  max_sep=200.0, nbins=15, sep_units="arcmin")
KK_KW   = dict(min_sep=10.0, max_sep=250.0, nbins=20, sep_units="arcmin")


def run_nn_benchmark(rng, label, n_gal, n_rand):
    ra_g, dec_g, ra_r, dec_r = make_galaxy_catalog(n_gal, n_rand, rng)
    (theta_tc, w_tc), t_tc = _timer(measure_two_point_function,
                                     ra_g, dec_g, ra_r, dec_r, **NN_KW)
    (theta_cf, w_cf), t_cf = _timer(measure_two_point_function_corrfunc,
                                     ra_g, dec_g, ra_r, dec_r, **NN_KW)
    _row(f"NN {label} ({n_gal/1e3:.0f}k/{n_rand/1e3:.0f}k)",
         t_tc, t_cf,
         _rms(theta_tc, theta_cf),
         _rms(w_tc, w_cf))
    return w_tc, w_cf


def run_kk_auto_benchmark(rng, label, n_pix):
    ra, dec, k, w = make_pixel_catalog(n_pix, rng)
    (theta_tc, xi_tc), t_tc = _timer(measure_kk_correlation_treecorr,
                                      ra, dec, k, w, **KK_KW)
    (theta_cf, xi_cf), t_cf = _timer(measure_kk_correlation_corrfunc,
                                      ra, dec, k, w, **KK_KW)
    _row(f"KK-auto {label} ({n_pix/1e3:.0f}k pix)",
         t_tc, t_cf,
         _rms(theta_tc, theta_cf),
         _rms(xi_tc, xi_cf))
    return xi_tc, xi_cf


def run_kk_cross_benchmark(rng, label, n_pix):
    ra1, dec1, k1, w1 = make_pixel_catalog(n_pix, rng)
    ra2, dec2, k2, w2 = make_pixel_catalog(n_pix, rng)
    (theta_tc, xi_tc), t_tc = _timer(measure_kk_correlation_treecorr,
                                      ra1, dec1, k1, w1,
                                      ra2=ra2, dec2=dec2, k2=k2, w2=w2, **KK_KW)
    (theta_cf, xi_cf), t_cf = _timer(measure_kk_correlation_corrfunc,
                                      ra1, dec1, k1, w1,
                                      ra2=ra2, dec2=dec2, k2=k2, w2=w2, **KK_KW)
    _row(f"KK-cross {label} ({n_pix/1e3:.0f}k pix)",
         t_tc, t_cf,
         _rms(theta_tc, theta_cf),
         _rms(xi_tc, xi_cf))
    return xi_tc, xi_cf


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    print(textwrap.dedent("""
        TreeCorr vs Corrfunc benchmark
        ==============================
        Columns: wall-clock time for each backend, speedup factor,
                 RMS residual on xi(theta) between the two results.
        Note: bin centres differ slightly (TreeCorr uses meanlogr,
              Corrfunc uses arithmetic thetaavg) and TreeCorr applies
              bin_slop=0.01 approximate binning — small rms_xi is expected.
    """))

    _header("1. NN Landy-Szalay  w(theta) = (DD-2DR+RR)/RR")
    for label, n_gal, n_rand, _ in SIZES:
        run_nn_benchmark(rng, label, n_gal, n_rand)

    _header("2. KK auto-correlation  xi(theta) = <k_i k_j>_w")
    for label, _, _, n_pix in SIZES:
        run_kk_auto_benchmark(rng, label, n_pix)

    _header("3. KK cross-correlation  xi(theta) = <k1_i k2_j>_w")
    for label, _, _, n_pix in SIZES:
        run_kk_cross_benchmark(rng, label, n_pix)

    print()


if __name__ == "__main__":
    main()
