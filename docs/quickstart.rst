*************
Quick start
*************


Installation
=============

To install the toolbox and the required packages, (it is recommended to create a virtual environment) simply run:

| ``git clone https://github.com/praneeth-b/mCorrect.git`` \
| ``cd mCorrect`` \
| ``python setup.py install``

In order to verify the toolbox you can start by running the example.py_ file which runs a sample correlation estimation on a synthetic dataset.
Additionally a tutorial style jupyter notebook describing various functionalities of the toolbox with executable cells can be found in the example_notebook_ .

.. _example.py: https://github.com/praneeth-b/mCorrect/blob/main/mCorrect/examples/linear_mcorrect/example.py

.. _example_notebook: https://github.com/praneeth-b/mCorrect/blob/main/mCorrect/examples/linear_mcorrect/example.ipynb


Repository structure
=====================

- ``mCorrect.datagen``: Consists of methods to generate synthetic multi-datasets based on a given correlation structure input.

- ``mCorrect.linear_mcorrect``: Consists of linear techniques(algorithms) to perform correlation analysis on multi-datasets.

- ``mCorrect.nonlinear_mcorrect`` (Todo): Consists of non-linear techniques(algorithms) to perform correlation analysis on multi-datasets.

- ``mCorrect.examples``: Contains example files describing the usage of the algorithms of the toolbox.
- ``mCorrect.visualization`` : Contains methods to graphically visualize the correlation sturcture in multiple datasets.

- ``mCorrect.metrics``: Contains the methods to measure performance metrics of the algorithms.

- ``mCorrect.utils``: Contains helper functions used throughout the toolbox.  










