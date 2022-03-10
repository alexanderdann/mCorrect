import numpy as np
from itertools import combinations
from mCorrect.datagen.MultisetDataGen import MultisetDataGen_CorrMeans
from mCorrect.linear_mcorrect.multiple_dataset.E_val_E_vec_tests import jointEVD
from mCorrect.visualization.graph_visu import visualization
from mCorrect.datagen.CorrelationStructureGen import CorrelationStructureGen
from mCorrect.metrics.metrics import Metrics
import matplotlib.pyplot as plt


def run_example():
    """
    This example performs correlation structure estimation on a multi-dataset consisting of 4 datasets with 5 signals
    each. Here, the correlation structure is such that: * 1 signal is correlated among all 4 datasets. * 1 signal is
    correlated among 3 of the 4 datasets. * 1 signal is correlated among just 2 of the 4 datasets. The jointEVD
    algorithm is used to estimate the number and structure of the correlated components of the multi-dataset.

    """

    n_sets = 1
    signum = 5
    corr_obj = CorrelationStructureGen(n_sets=n_sets, signum=signum,tot_corr=[4, 3, 2], percentage=False)

    corr_truth, p_matrix, sigma_signals, R_matrix = corr_obj.get_structure()
    datagen = MultisetDataGen_CorrMeans(n_sets=n_sets, signum=signum, p=p_matrix, R=R_matrix, M=800, color='white')  # chk color
    X, _ = datagen.generate()
    corr_test = jointEVD(X, B=1000)
    corr_estimate, d_cap, u_struc1 = corr_test.find_structure()
    plt.ion()
    viz = visualization(graph_matrix=corr_truth, num_dataset=n_sets, label_edge=False)
    viz.visualize("True Structure")
    plt.ioff()
    viz_op = visualization(graph_matrix=corr_estimate, num_dataset=n_sets, label_edge=False)
    viz_op.visualize("Estimated_structure")

    #sig = corr_test.estimate_signals(X, d_cap, corr_estimate)

    # print("experiment complete")
    print(f'experiment complete')


if __name__ == "__main__":
    run_example()
