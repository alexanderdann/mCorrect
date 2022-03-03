
# Corramal

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

cd Corramal/

python setup.py install


cd Corramal/

python setup.py install

```

## Repository Structure

* datagen: Contains methods to generate two or multiple datasets for a given correlation structure
* linear_corramal: consists algorithms for linear methods to perform correlation analysis
* metrics: implemtation of metrics to assess the performance of various algorithms
* visualization: methods to visualize the correllation structure between various datasets


## References
[1] T. Hasija, C. Lameiro, T. Marrinan,  and P. J. Schreier,"Determining the Dimension and Structure of the Subspace Correlated Across Multiple Data Sets,".

[2] T. Hasija, Y. Song, P. J. Schreier and D. Ramirez, "Bootstrap-based Detection of the Number of Signals Correlated across Multiple Data Sets," Proc. Asilomar Conf. Signals Syst. Computers, Pacific Grove, CA, USA, November 2016.


