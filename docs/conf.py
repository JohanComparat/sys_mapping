import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "sys_mapping"
author = "JohanComparat"
release = "0.9"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": -1,
    "titles_only": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Allow .. raw:: html blocks in RST pages (needed for direct _static image links)
raw_enabled = True

# Pass raw_enabled through to the docutils HTML writer
docutils_settings = {
    'raw_enabled': True,
    'file_insertion_enabled': True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# Suppress duplicate-object warnings from dataclass field auto-documentation
suppress_warnings = ["app.add_node", "ref.duplicate", "autodoc"]
