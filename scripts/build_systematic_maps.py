"""Build HEALPix systematic template maps from GAIA DR2 or LS10 BGS randoms.

Outputs one FITS file per (source, quantity, NSIDE) combination in the output
directory, ready for consumption by the sys_mapping pipeline.

GAIA outputs  : GAIA_{quantity}_NSIDE_{nside:05d}.fits
LS10 outputs  : LS10_{quantity}_NSIDE_{nside:04d}.fits
"""
import argparse
import glob
import os
import re
import sys
import time

import healpy as hp
import numpy as np
from astropy.io import fits


# ── GAIA ──────────────────────────────────────────────────────────────────

GAIA_FLUX_QUANTITIES = ["phot_g_mean_flux", "phot_bp_mean_flux", "phot_rp_mean_flux"]
GAIA_MAG_BINS = {
    "nstar_bright": (-10, 10),
    "nstar_medium": (10, 14),
    "nstar_faint": (14, 17),
}
_RE_GAIA = re.compile(r"table_(-?\d+)_g_(-?\d+)\.fits$")


def _classify_gaia_file(basename):
    m = _RE_GAIA.search(basename)
    if m is None:
        return None, None, None
    g_min, g_max = int(m.group(1)), int(m.group(2))
    for bname, (lo, hi) in GAIA_MAG_BINS.items():
        if g_min >= lo and g_max <= hi:
            return g_min, g_max, bname
    mid = (g_min + g_max) / 2.0
    if mid < 10:
        return g_min, g_max, "nstar_bright"
    if mid < 14:
        return g_min, g_max, "nstar_medium"
    return g_min, g_max, "nstar_faint"


def build_gaia_maps(input_pattern, output_dir, nsides):
    paths = sorted(glob.glob(input_pattern))
    if not paths:
        print(f"ERROR: no files matching {input_pattern}", file=sys.stderr)
        sys.exit(1)

    files_info = []
    for path in paths:
        g_min, g_max, bin_name = _classify_gaia_file(os.path.basename(path))
        if bin_name is None:
            print(f"WARNING: cannot parse {os.path.basename(path)}, skipping")
            continue
        files_info.append(dict(path=path, g_min=g_min, g_max=g_max, mag_bin=bin_name))
    files_info.sort(key=lambda d: d["g_min"])
    print(f"Found {len(files_info)} GAIA input files")

    flux_sum = {
        nside: {q: np.zeros(hp.nside2npix(nside), np.float64) for q in GAIA_FLUX_QUANTITIES}
        for nside in nsides
    }
    count_sum = {
        nside: {b: np.zeros(hp.nside2npix(nside), np.int64) for b in GAIA_MAG_BINS}
        for nside in nsides
    }

    t0 = time.time()
    for fi in files_info:
        t1 = time.time()
        with fits.open(fi["path"], memmap=True) as hdul:
            d = hdul[1].data
            ra = np.asarray(d["ra"], np.float64)
            dec = np.asarray(d["dec"], np.float64)
            flux_arrays = {
                q: np.asarray(d[q], np.float64)
                for q in GAIA_FLUX_QUANTITIES
                if q in d.names
            }
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        for nside in nsides:
            pix = hp.ang2pix(nside, theta, phi)
            npix = hp.nside2npix(nside)
            for q in GAIA_FLUX_QUANTITIES:
                if q not in flux_arrays:
                    continue
                farr = flux_arrays[q]
                valid = ~np.isnan(farr)
                if valid.any():
                    flux_sum[nside][q] += np.bincount(pix[valid], weights=farr[valid], minlength=npix)
            count_sum[nside][fi["mag_bin"]] += np.bincount(pix, minlength=npix)
        print(f"  {os.path.basename(fi['path']):30s}  n={len(ra):9,d}  ({time.time()-t1:.1f} s)")

    print(f"All files processed in {time.time()-t0:.1f} s")

    output_paths = {}
    for nside in nsides:
        t1 = time.time()
        for q in GAIA_FLUX_QUANTITIES:
            smap = flux_sum[nside][q].copy()
            smap[smap == 0] = hp.UNSEEN
            fname = f"GAIA_{q}_NSIDE_{nside:05d}.fits"
            fpath = os.path.join(output_dir, fname)
            hp.write_map(
                fpath, smap,
                coord="C", column_names=[q],
                extra_header=[
                    ("SURVEY", "GAIA_DR2", "GAIA Data Release 2"),
                    ("QUANTITY", q, "flux column summed"),
                    ("AGGR", "SUM", "aggregation method"),
                    ("NSIDE", nside, "HEALPix NSIDE"),
                    ("ORDERING", "RING", "HEALPix pixel ordering"),
                    ("N_FILES", len(files_info), "number of input tables"),
                ],
                overwrite=True,
            )
            output_paths[(nside, q)] = fpath
        for bin_name, (lo, hi) in GAIA_MAG_BINS.items():
            cmap = count_sum[nside][bin_name].astype(np.float64)
            cmap[cmap == 0] = hp.UNSEEN
            fname = f"GAIA_{bin_name}_NSIDE_{nside:05d}.fits"
            fpath = os.path.join(output_dir, fname)
            hp.write_map(
                fpath, cmap,
                coord="C", column_names=[bin_name],
                extra_header=[
                    ("SURVEY", "GAIA_DR2", "GAIA Data Release 2"),
                    ("QUANTITY", bin_name, "star count label"),
                    ("AGGR", "COUNT", "aggregation method"),
                    ("GMAGLO", lo, "G mag lower bound (inclusive)"),
                    ("GMAGHI", hi, "G mag upper bound (exclusive)"),
                    ("NSIDE", nside, "HEALPix NSIDE"),
                    ("ORDERING", "RING", "HEALPix pixel ordering"),
                    ("N_FILES", len(files_info), "number of input tables"),
                ],
                overwrite=True,
            )
            output_paths[(nside, bin_name)] = fpath
        print(f"NSIDE={nside:4d} written in {time.time()-t1:.1f} s")

    _print_summary(output_paths, nsides, GAIA_FLUX_QUANTITIES + list(GAIA_MAG_BINS.keys()))


# ── LS10 ──────────────────────────────────────────────────────────────────

LS10_QUANTITIES = ["EBV", "GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_Z", "PSFSIZE_R", "NOBS_R"]


def build_ls10_maps(input_file, output_dir, nsides):
    if not os.path.exists(input_file):
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_file} …")
    t0 = time.time()
    with fits.open(input_file, memmap=True) as hdul:
        data = hdul[1].data
        ra = np.asarray(data["RA"], np.float64)
        dec = np.asarray(data["DEC"], np.float64)
        vals = {q: np.asarray(data[q], np.float64) for q in LS10_QUANTITIES if q in data.names}
    print(f"Loaded {len(ra):,d} randoms in {time.time()-t0:.1f} s")

    missing = [q for q in LS10_QUANTITIES if q not in vals]
    if missing:
        print(f"WARNING: columns not found in input and will be skipped: {missing}")

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    output_paths = {}
    for nside in nsides:
        t1 = time.time()
        npix = hp.nside2npix(nside)
        pix = hp.ang2pix(nside, theta, phi)
        pix_count = np.bincount(pix, minlength=npix).astype(np.float64)
        covered = pix_count > 0
        print(
            f"NSIDE={nside:4d}  npix={npix:7d}  covered={covered.sum():7d}"
            f"  ({covered.sum()/npix*100:.1f} %)  mean randoms/pix={pix_count[covered].mean():.1f}"
        )
        for qty in list(vals.keys()):
            pix_sum = np.bincount(pix, weights=vals[qty], minlength=npix)
            mean_map = np.full(npix, hp.UNSEEN, np.float64)
            mean_map[covered] = pix_sum[covered] / pix_count[covered]
            fname = f"LS10_{qty}_NSIDE_{nside:04d}.fits"
            fpath = os.path.join(output_dir, fname)
            hp.write_map(
                fpath, mean_map,
                coord="C", column_names=[qty],
                extra_header=[
                    ("SURVEY", "LS10", "Legacy Survey DR10"),
                    ("QUANTITY", qty, "systematic quantity"),
                    ("NSIDE", nside, "HEALPix NSIDE"),
                    ("ORDERING", "RING", "HEALPix pixel ordering"),
                    ("SRCFILE", os.path.basename(input_file), "input random catalog"),
                    ("N_RAND", len(ra), "number of randoms used"),
                ],
                overwrite=True,
            )
            output_paths[(nside, qty)] = fpath
        print(f"  → written in {time.time()-t1:.1f} s")

    _print_summary(output_paths, nsides, list(vals.keys()))


# ── Shared helpers ─────────────────────────────────────────────────────────

def _print_summary(output_paths, nsides, labels):
    print("\nGenerated files:")
    for nside in nsides:
        for label in labels:
            key = (nside, label)
            if key not in output_paths:
                continue
            fpath = output_paths[key]
            size_mb = os.path.getsize(fpath) / 1e6
            m = hp.read_map(fpath)
            good = m != hp.UNSEEN
            print(
                f"  {os.path.basename(fpath):<58}  {size_mb:6.2f} MB"
                f"  covered={good.sum():7d}  mean={m[good].mean():.4g}"
            )


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build HEALPix systematic template maps (GAIA DR2 or LS10 BGS randoms).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=["gaia", "ls10"], required=True,
        help="Survey source to process.",
    )
    parser.add_argument(
        "--nside", type=int, nargs="+", default=[32, 64, 128, 256],
        metavar="N",
        help="HEALPix NSIDE value(s) to produce.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/data/legacysurvey/dr10/systematics/"),
        help="Directory where output FITS files are written.",
    )
    # GAIA-specific
    parser.add_argument(
        "--gaia-input-pattern",
        default=os.path.expanduser("~/data/gaia_cat/table_*.fits"),
        help="Glob pattern for GAIA DR2 magnitude table FITS files (--source gaia).",
    )
    # LS10-specific
    parser.add_argument(
        "--ls10-input-file",
        default=os.path.expanduser("~/data/legacysurvey/dr10/randoms/resolve/randoms-1-0-BGS.fits"),
        help="Path to the LS10 BGS random catalog FITS file (--source ls10).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "gaia":
        build_gaia_maps(args.gaia_input_pattern, args.output_dir, args.nside)
    else:
        build_ls10_maps(args.ls10_input_file, args.output_dir, args.nside)


if __name__ == "__main__":
    main()
