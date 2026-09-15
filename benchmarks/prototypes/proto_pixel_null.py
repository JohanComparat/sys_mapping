"""Prototype 5C.5: null overdensity fields drawn per pixel instead of from point catalogues.

Point path (scripts/run_ls10_analysis.null_overdensity_fields): GLASS field -> galaxy
positions -> randoms on the sphere -> pixelise both -> delta.
Pixel path: the same GLASS field -> N_g ~ Poisson(nbar (1 + delta)) and
N_r ~ Poisson(rand_factor nbar) on the footprint pixels -> delta.

The two are the same distribution; they consume random numbers differently, so the
realisations differ.  Compared here on what the calibrations use: the pixel variance of
the null fields and the scatter of the OLS amplitudes (calibrated_template_significance).
"""
import importlib.util
import sys
import time
import warnings

import healpy as hp
import numpy as np

import sys_mapping as sm
from sys_mapping.glass_mocks import generate_glass_delta_map

warnings.filterwarnings("ignore")


def pixel_null_fields(n_mocks, nside, good, n_total_footprint, z_max, seed, *,
                      rand_factor=2, cl_amplitude=None, cl_input=None):
    n_good = int(good.sum())
    nbar = n_total_footprint / n_good
    out = np.empty((n_mocks, n_good))
    for i in range(n_mocks):
        delta = generate_glass_delta_map(nside=nside, z_max=z_max, seed=seed + i,
                                         cl_amplitude=cl_amplitude, cl_input=cl_input)[good]
        rng = np.random.default_rng(seed + 10_000_019 + i)
        ng = rng.poisson(nbar * np.clip(1.0 + delta, 0.0, None)).astype(float)
        nr = rng.poisson(rand_factor * nbar, n_good).astype(float)
        norm = ng.sum() / nr.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            dg = ng / (nr * norm) - 1.0
        dg[~np.isfinite(dg)] = 0.0
        out[i] = dg
    return out


if __name__ == "__main__":
    nside, n_mocks = int(sys.argv[1]), int(sys.argv[2])
    spec = importlib.util.spec_from_file_location(
        "ls10", "/home/comparat/software/sys_mapping/scripts/run_ls10_analysis.py")
    ls10 = importlib.util.module_from_spec(spec)
    sys.modules["ls10"] = ls10
    spec.loader.exec_module(ls10)

    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(2)
    good = np.abs(hp.pix2ang(nside, np.arange(npix), lonlat=True)[1]) > 25
    T = sm.standardise_on_footprint(np.array(
        [hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(f))[good]
         for f in np.linspace(3, 20, 11)]))
    n_foot = 100 * int(good.sum())                      # 100 galaxies per pixel
    kw = dict(nside=nside, good_pix=good, n_total_footprint=n_foot, z_edges=np.array([0.05, 0.3]),
              nz=np.array([1.0]), seed=500, cl_amplitude=5e-3)

    t0 = time.perf_counter()
    point = ls10.null_overdensity_fields(n_mocks, **kw)
    t_point = time.perf_counter() - t0
    t0 = time.perf_counter()
    pix = pixel_null_fields(n_mocks, nside, good, n_foot, 0.3, 9000, cl_amplitude=5e-3)
    t_pix = time.perf_counter() - t0

    s_point = sm.calibrated_template_significance(point[0], T, point)["sigma"]
    s_pix = sm.calibrated_template_significance(pix[0], T, pix)["sigma"]
    ratio = s_pix / s_point
    se = np.sqrt(2.0 / (n_mocks - 1))
    print(f"NSIDE {nside}, {good.sum()} pixels, {n_mocks} mocks each, 100 galaxies/pixel")
    print(f"time per mock: point {t_point / n_mocks:.2f} s, pixel {t_pix / n_mocks:.2f} s "
          f"({t_point / t_pix:.1f}x)")
    print(f"pixel variance: point {point.var(axis=1).mean():.5f}, pixel {pix.var(axis=1).mean():.5f}")
    print(f"amplitude scatter ratio pixel/point per template: median {np.median(ratio):.3f}, "
          f"range {ratio.min():.3f}-{ratio.max():.3f} (sampling error per ratio ~{se:.3f})")
