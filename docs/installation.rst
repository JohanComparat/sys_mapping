Installation
============

Requirements
------------

* Python 3.11+
* `mamba <https://mamba.readthedocs.io>`_ (recommended) or conda

From PyPI
---------

.. code-block:: bash

   pip install sys_mapping

For the optional ElasticNet regression method:

.. code-block:: bash

   pip install "sys_mapping[regression]"

Create the environment
----------------------

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate sys_map

This installs all dependencies including JAX (CPU build), healpy, emcee and
TreeCorr.

Install in editable mode
------------------------

.. code-block:: bash

   pip install -e .

Run the test suite
------------------

.. code-block:: bash

   pytest tests/ -v

Build the documentation
-----------------------

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme myst-parser
   cd docs
   make html
   # open docs/_build/html/index.html
