"""
Patch *_params.json files with scalar diagnostics from per-method partial JSON files.

Handles two partial-file naming conventions:
  - Combined: _partial_ElasticNet_ISD-1_OLS.json  (OLS, ElasticNet, ISD-1 in one file)
  - Separate: _partial_ISD-3.json

Fields added/patched:
  sigma_hat_ols    ← OLS partial
  sigma_hat_enet   ← ElasticNet partial
  sigma_hat_isd1   ← ISD-1 partial
  sigma_hat_isd3   ← ISD-3 partial
  sigma_hat_add    ← MCMC-add partial (only if separate file exists)
  acceptance_fraction_add ← same
"""

import json
import glob
import os

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sys_weights")
WEIGHTS_DIR = os.path.normpath(WEIGHTS_DIR)

NSIDES = [32, 64, 128, 256]


def _patch_one(params_path):
    base = params_path.replace("_params.json", "")
    sample_id = os.path.basename(base)
    changed = False

    with open(params_path) as f:
        params = json.load(f)

    # ── Combined OLS+ElasticNet+ISD-1 partial ──────────────────────────────
    combined_path = f"{base}_partial_ElasticNet_ISD-1_OLS.json"
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            combo = json.load(f)
        for method, key in [("OLS", "sigma_hat_ols"),
                             ("ElasticNet", "sigma_hat_enet"),
                             ("ISD-1", "sigma_hat_isd1")]:
            mdata = combo.get(method, {})
            sigma = mdata.get("sigma_hat")
            if sigma is not None:
                params[key] = sigma
                changed = True
        print(f"  {sample_id[-35:]}: OLS={params.get('sigma_hat_ols',float('nan')):.4f}  "
              f"ElasticNet={params.get('sigma_hat_enet',float('nan')):.4f}  "
              f"ISD-1={params.get('sigma_hat_isd1',float('nan')):.4f}")
    else:
        # Legacy: separate files per method
        for method, key in [("OLS", "sigma_hat_ols"),
                             ("ElasticNet", "sigma_hat_enet"),
                             ("ISD-1", "sigma_hat_isd1")]:
            p = f"{base}_partial_{method}.json"
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                sigma = d.get(method, {}).get("sigma_hat")
                if sigma is not None:
                    params[key] = sigma
                    changed = True

    # ── ISD-3 partial (always separate) ───────────────────────────────────
    isd3_path = f"{base}_partial_ISD-3.json"
    if os.path.exists(isd3_path):
        with open(isd3_path) as f:
            d3 = json.load(f)
        sigma3 = d3.get("ISD-3", {}).get("sigma_hat")
        if sigma3 is not None:
            params["sigma_hat_isd3"] = sigma3
            print(f"  {sample_id[-35:]}: ISD-3={sigma3:.4f}")
            changed = True

    # ── MCMC-add partial (separate, only if sigma_hat_add broken) ─────────
    add_path = f"{base}_partial_MCMC-add.json"
    if os.path.exists(add_path) and (
        params.get("sigma_hat_add", -999.0) < -900
        or params.get("sigma_hat_add") is None
    ):
        with open(add_path) as f:
            da = json.load(f)
        mdata = da.get("MCMC-add", {})
        sigma_a = mdata.get("sigma_hat")
        acc_a = mdata.get("acceptance_fraction")
        if sigma_a is not None:
            params["sigma_hat_add"] = sigma_a
            params["acceptance_fraction_add"] = acc_a
            print(f"  {sample_id[-35:]}: sigma_hat_add patched → {sigma_a:.4f}")
            changed = True

    if changed:
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  → Wrote {os.path.basename(params_path)}")
    else:
        print(f"  {sample_id[-35:]}: nothing to patch")

    return changed


for nside in NSIDES:
    print(f"\n=== NSIDE {nside:04d} ===")
    pattern = os.path.join(WEIGHTS_DIR, f"*_NSIDE{nside:04d}_params.json")
    for params_path in sorted(glob.glob(pattern)):
        _patch_one(params_path)

print("\nDone.")
