"""
Shared matplotlib color and style constants for all sys_mapping analysis scripts.

Every script that produces figures should import from here so that the visual
representation of each decontamination method is consistent across all plots.

Usage
-----
>>> from sys_mapping.plotting import METHOD_COLORS, METHOD_LINESTYLES, METHOD_LABELS
>>> ax.plot(theta, w_corr, color=METHOD_COLORS['MCMC-comb'],
...         linestyle=METHOD_LINESTYLES['MCMC-comb'],
...         label=METHOD_LABELS['MCMC-comb'])

Color palette
-------------
The palette is inspired by Tableau-10 and tested for colorblind accessibility
(deuteranopia and protanopia safe):

  OLS        — steel blue    #4E79A7
  ElasticNet — forest green  #59A14F
  ISD-1      — amber         #F28E2B
  ISD-3      — coral red     #E15759
  MCMC-add   — muted purple  #B07AA1
  MCMC-comb  — teal          #76B7B2

Two additional keys cover the *data* and *uncorrected* reference curves:

  observed   — near-black    #333333   (raw measurement before correction)
  uncorrected— mid-grey      #888888   (baseline with no decontamination applied)
"""

# ---------------------------------------------------------------------------
# Per-method color (hex RGB, consistent with Tableau-10 palette)
# ---------------------------------------------------------------------------
METHOD_COLORS: dict[str, str] = {
    "OLS":         "#4E79A7",   # steel blue
    "ElasticNet":  "#59A14F",   # forest green
    "ISD-1":       "#F28E2B",   # amber
    "ISD-3":       "#E15759",   # coral red
    "MCMC-add":    "#B07AA1",   # muted purple
    "MCMC-comb":   "#76B7B2",   # teal
    # reference curves (not method-specific)
    "observed":    "#333333",   # near-black
    "uncorrected": "#888888",   # mid-grey
}

# ---------------------------------------------------------------------------
# Per-method line style for line plots (matplotlib linestyle spec)
# ---------------------------------------------------------------------------
METHOD_LINESTYLES: dict[str, object] = {
    "OLS":        ":",                  # dotted
    "ElasticNet": "--",                 # dashed
    "ISD-1":      "-.",                 # dash-dot
    "ISD-3":      (0, (3, 1, 1, 1)),   # densely dash-dot-dot
    "MCMC-add":   (0, (5, 2)),          # loosely dashed
    "MCMC-comb":  "-",                  # solid
    "observed":   "-.",
    "uncorrected": ":",
}

# ---------------------------------------------------------------------------
# Per-method marker style for scatter / errorbar plots
# ---------------------------------------------------------------------------
METHOD_MARKERS: dict[str, str] = {
    "OLS":        "o",   # circle
    "ElasticNet": "s",   # square
    "ISD-1":      "^",   # up triangle
    "ISD-3":      "D",   # diamond
    "MCMC-add":   "v",   # down triangle
    "MCMC-comb":  "*",   # star
    "observed":   "o",
    "uncorrected": "x",
}

# ---------------------------------------------------------------------------
# Human-readable labels for legends
# ---------------------------------------------------------------------------
METHOD_LABELS: dict[str, str] = {
    "OLS":         "OLS",
    "ElasticNet":  "ElasticNet",
    "ISD-1":       "ISD (poly-1)",
    "ISD-3":       "ISD (poly-3)",
    "MCMC-add":    "MCMC additive",
    "MCMC-comb":   "MCMC combined",
    "observed":    "Observed",
    "uncorrected": "Uncorrected",
}

# Canonical ordering for consistent legend / table ordering
METHOD_ORDER: list[str] = [
    "OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"
]
