*************
Quick start
*************
This is an overview on how to get started with the mCorrect toolbox and to get familiar with the structure of the toolbox.

Installation
=============

To install the toolbox and the required packages, (it is recommended to create a virtual environment) simply run:

| ``git clone https://github.com/praneeth-b/mCorrect.git`` \
| ``cd mCorrect`` \
| ``python setup.py install``



Repository structure
=====================

- mCorrect.datagen_ : Consists of methods to generate synthetic multi-datasets based on a given correlation structure input.

- mCorrect.linear_mcorrect_ : Consists of linear techniques(algorithms) to perform correlation analysis on multi-datasets.

- mCorrect.nonlinear_mcorrect_ (Todo): Consists of non-linear techniques(algorithms) to perform correlation analysis on multi-datasets.

- mCorrect.examples: Contains example files describing the usage of the algorithms of the toolbox.
- mCorrect.visualization_ : Contains methods to graphically visualize the correlation sturcture in multiple datasets.

- mCorrect.metrics_ : Contains the methods to measure performance metrics of the algorithms.

- mCorrect.utils_ : Contains helper functions used throughout the toolbox.  


.. _mCorrect.datagen: file:///home/praneeth/projects/sst/git/mCorrect/docs/_build/html/mCorrect.datagen.html#module-mCorrect.datagen


.. _mCorrect.linear_mcorrect: file:///home/praneeth/projects/sst/git/mCorrect/docs/_build/html/mCorrect.linear_mcorrect.html#module-mCorrect.linear_mcorrect

.. _mCorrect.nonlinear_mcorrect: file:///home/praneeth/projects/sst/git/mCorrect/docs/_build/html/mCorrect.nonlinear_mcorrect.html#module-mCorrect.nonlinear_mcorrect

.. _mCorrect.visualization: file:///home/praneeth/projects/sst/git/mCorrect/docs/_build/html/mCorrect.visualization.html#module-mCorrect.visualization

.. _mCorrect.metrics: file:///home/praneeth/projects/sst/git/mCorrect/docs/_build/html/mCorrect.metrics.html#module-mCorrect.metrics

.. _mCorrect.utils: file:///home/praneeth/projects/sst/git/mCorrect/docs/_build/html/mCorrect.utils.html#module-mCorrect.utils


Examples
=========

In order to verify the toolbox you can start by running the example.py_ file which runs a sample correlation estimation on a synthetic dataset.
Additionally a tutorial style jupyter notebook describing various functionalities of the toolbox with executable cells can be found in the example_notebook_ .

.. _example.py: https://github.com/praneeth-b/mCorrect/blob/main/mCorrect/examples/linear_mcorrect/example.py

.. _example_notebook: https://github.com/praneeth-b/mCorrect/blob/main/mCorrect/examples/linear_mcorrect/example.ipynb





