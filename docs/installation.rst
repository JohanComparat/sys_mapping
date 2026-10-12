Installation
============

``sys_mapping`` installs from PyPI or from source.

Requirements
------------

* Python 3.11 or later
* NumPy, SciPy, Astropy, healpy, emcee, joblib, TreeCorr, Matplotlib
* ``jax[cpu] >= 0.9`` and ``blackjax >= 1.2, < 1.6``

``pip`` installs these with the package.
Optional extras:

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

.. note::

   **Maintainer setup.** On the development laptop, use the shared ``dev``
   environment defined in ``~/software/dev_env`` (``conda activate dev``); this
   package is already installed there in editable mode. Do not create a separate
   environment for it: add missing dependencies to ``~/software/dev_env`` and
   rebuild.

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate sys_map
   pip install -e ".[dev,regression,glass]"

``environment.yml`` creates the ``sys_map`` environment with Python 3.11, the scientific stack, JAX (CPU), TreeCorr, Corrfunc, GLASS, scikit-learn and Sphinx; the editable install adds the remaining dependencies, BlackJAX among them.

Running the tests
-----------------

.. code-block:: bash

   pytest                 # tests/ and the doctests in sys_mapping/
   pytest -m "not slow"   # without the slow statistical tests

Building the documentation
--------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html
   # open docs/_build/html/index.html
