Installation
============

Requirements
------------

* Python 3.11 or later
* NumPy, SciPy, Astropy, healpy, emcee, joblib, TreeCorr, Matplotlib
* ``jax[cpu] >= 0.9`` and ``blackjax >= 1.2, < 1.6``

``pip`` installs these with the package.  Optional extras:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Installs
     - Needed for
   * - ``regression``
     - ``scikit-learn >= 1.3``
     - ElasticNet
   * - ``glass``
     - ``glass >= 2026.1``
     - GLASS mocks and every calibrated null
   * - ``corrfunc``
     - ``corrfunc``
     - Corrfunc pair counting
   * - ``dev``
     - ``pytest``, ``pytest-cov``, ``matplotlib``, ``scikit-learn >= 1.3``
     - The test suite

From PyPI
---------

.. code-block:: bash

   pip install sys-mapping
   pip install "sys-mapping[regression,glass]"

From source
-----------

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate sys_map
   pip install -e ".[dev,regression,glass]"

``environment.yml`` creates the ``sys_map`` environment with Python 3.11, the
scientific stack, JAX (CPU), TreeCorr, Corrfunc, GLASS, scikit-learn and Sphinx; the
editable install adds the remaining dependencies, BlackJAX among them.

Tests
-----

.. code-block:: bash

   pytest                 # tests/ and the doctests in sys_mapping/
   pytest -m "not slow"   # without the slow statistical tests

Documentation
-------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html
   # open docs/_build/html/index.html
