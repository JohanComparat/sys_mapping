"""
Shared matplotlib color and style constants for all sys_mapping analysis scripts.

Every script that produces figures should import from here so that the visual
representation of each decontamination method is consistent across all plots.

Usage
-----
>>> from sys_mapping.plotting import METHOD_COLORS, METHOD_LINESTYLES, METHOD_LABELS
>>> ax.plot(theta, w_corr, color=METHOD_COLORS['MCMC-comb'],  # doctest: +SKIP
...         linestyle=METHOD_LINESTYLES['MCMC-comb'],
...         label=METHOD_LABELS['MCMC-comb'])

Color palette
-------------
The six method colours are taken from the Okabe--Ito set.
Under simulated protanopia, deuteranopia and tritanopia (Machado et al. 2009, full
severity), every pair stays at least 8.6 apart in OKLab (x100), and at least 15.6 apart
with normal vision; each method also has its own line style.
Related methods share a hue family: the least-squares pair in blue, the two ISD orders
in orange.

  OLS        — blue          #0072B2
  MCMC-add   — sky blue      #56B4E9
  ElasticNet — bluish green  #009E73
  ISD-1      — orange        #E69F00
  ISD-3      — vermillion    #D55E00
  MCMC-comb  — black         #000000

Two additional keys cover the *data* and *uncorrected* reference curves:

  observed   — mid-grey      #777777   (raw measurement before correction)
  uncorrected— light grey    #AAAAAA   (baseline with no decontamination applied)
"""

# ---------------------------------------------------------------------------
# Per-method color (hex RGB, Okabe--Ito)
# ---------------------------------------------------------------------------
METHOD_COLORS: dict[str, str] = {
    "OLS":         "#0072B2",   # blue
    "ElasticNet":  "#009E73",   # bluish green
    "ISD-1":       "#E69F00",   # orange
    "ISD-3":       "#D55E00",   # vermillion
    "MCMC-add":    "#56B4E9",   # sky blue
    "MCMC-comb":   "#000000",   # black
    # reference curves (not method-specific)
    "observed":    "#777777",   # mid-grey
    "uncorrected": "#AAAAAA",   # light grey
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
    "ISD-1":       "ISD-1",
    "ISD-3":       "ISD-3",
    "MCMC-add":    "MCMC-add",
    "MCMC-comb":   "MCMC-comb",
    "observed":    "Observed",
    "uncorrected": "Uncorrected",
}

# Canonical ordering for consistent legend / table ordering
METHOD_ORDER: list[str] = [
    "OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"
]
