
# mCorrect

A python implementation of the Techniques presented in '[1]' for model selection and correlation structure estimation in multiple datasets.
Given a multi-modal dataset, this technique estimates the following:
* The number of correlated components across the datasets.
* The structure of the correlated components

The technique solves the complete model selection problem shown above by employing bootstrap based hypothesis testing.

Cite the work as follows:

```
@article{hasija2019determining,
  title={Determining the dimension and structure of the subspace correlated across multiple data sets},
  author={Hasija, Tanuj and Lameiro, Christian and Marrinan, Timothy and Schreier, Peter J},
  journal={arXiv preprint arXiv:1901.11366},
  year={2019}
}

```

## Installation

To install the toolbox and the required packages, (it is recommended to create a virtual environment) simply run:
```
 git clone https://github.com/praneeth-b/corramal.git

cd mCorrect/

python setup.py install

```


## Repository Structure

- ``mCorrect.datagen``: Consists of methods to generate synthetic multi-datasets based on a given correlation structure input.

- ``mCorrect.linear_mcorrect``: Consists of linear techniques(algorithms) to perform correlation analysis on multi-datasets.

- ``mCorrect.nonlinear_mcorrect`` (Todo): Consists of non-linear techniques(algorithms) to perform correlation analysis on multi-datasets.

- ``mCorrect.examples``: Contains example files describing the usage of the algorithms of the toolbox. The example notebook contains a tutorial style jupyter notebook which demontrates the usage of various modules of the toolbox within executable cells which can be found [here](https://github.com/praneeth-b/mCorrect/blob/main/mCorrect/examples/linear_mcorrect/example.ipynb)
- ``mCorrect.visualization`` : Contains methods to graphically visualize the correlation sturcture in multiple datasets.

- ``mCorrect.metrics``: Contains the methods to measure performance metrics of the algorithms.

- ``mCorrect.utils``: Contains helper functions used throughout the toolbox.



## References
[1] T. Hasija, C. Lameiro, T. Marrinan,  and P. J. Schreier,"Determining the Dimension and Structure of the Subspace Correlated Across Multiple Data Sets,".

[2] T. Hasija, Y. Song, P. J. Schreier and D. Ramirez, "Bootstrap-based Detection of the Number of Signals Correlated across Multiple Data Sets," Proc. Asilomar Conf. Signals Syst. Computers, Pacific Grove, CA, USA, November 2016.


