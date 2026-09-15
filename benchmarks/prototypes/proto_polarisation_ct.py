"""Prototype: template correlation matrix from auto-correlations of summed fields.

xi_ij = 1/2 [xi(t_i + t_j) - xi_ii - xi_jj] is exact for a K-K correlation, which is
bilinear in the field values when the pair weights do not depend on them.  TreeCorr's
auto pass on one catalogue counts each pair once, a cross pass on the same catalogue
counts it twice, so n(n-1)/2 cross passes become as many cheaper auto passes.
"""
import sys
import time
import warnings

import numpy as np

import sys_mapping as sm
from sys_mapping.utils import measure_kk_correlation_treecorr

warnings.filterwarnings("ignore")


def matrix_by_polarisation(ra, dec, k, min_sep=0.5, max_sep=300.0, nbins=30,
                           sep_units="arcmin", bin_slop=0.01):
    n_sys = k.shape[0]
    kw = dict(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units,
              bin_slop=bin_slop)
    xi = np.zeros((n_sys, n_sys, nbins))
    theta = None
    for i in range(n_sys):
        theta, xi[i, i] = measure_kk_correlation_treecorr(ra, dec, k[i], **kw)
    for i in range(n_sys):
        for j in range(i + 1, n_sys):
            _, xs = measure_kk_correlation_treecorr(ra, dec, k[i] + k[j], **kw)
            xi[i, j] = xi[j, i] = 0.5 * (xs - xi[i, i] - xi[j, j])
    return theta, xi


if __name__ == "__main__":
    import healpy as hp
    n_gal, n_sys = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(0)
    nside = 64
    maps = np.array([hp.smoothing(rng.standard_normal(hp.nside2npix(nside)),
                                  fwhm=np.radians(3 + 2 * i)) for i in range(n_sys)])
    ra = rng.uniform(0, 360, n_gal)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-30)), np.sin(np.radians(60)), n_gal)))
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    k = maps[:, pix]
    k = (k - k.mean(1, keepdims=True)) / k.std(1, keepdims=True)

    t0 = time.perf_counter()
    th_ref, xi_ref = sm.template_correlation_matrix(ra, dec, k)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    th_pol, xi_pol = matrix_by_polarisation(ra, dec, k)
    t_pol = time.perf_counter() - t0
    off = ~np.eye(n_sys, dtype=bool)
    diff = np.abs(xi_pol - xi_ref)
    scale = np.max(np.abs(xi_ref[off]))
    print(f"{n_gal} galaxies, {n_sys} templates: library {t_ref:.1f} s, polarisation {t_pol:.1f} s "
          f"({t_ref / t_pol:.2f}x)")
    print(f"max |diff| off-diagonal {diff[off].max():.3e} (largest |xi_ij| {scale:.3e}), "
          f"diagonal {diff[~off].max():.3e}")
