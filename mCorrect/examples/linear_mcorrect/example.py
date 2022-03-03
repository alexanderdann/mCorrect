import numpy as np
from itertools import combinations
from mCorrect.datagen.MultisetDataGen import MultisetDataGen_CorrMeans
from mCorrect.linear_mcorrect.multiple_dataset.E_val_E_vec_tests import jointEVD
from mCorrect.visualization.graph_visu import visualization
from mCorrect.datagen.CorrelationStructureGen import CorrelationStructureGen
from mCorrect.metrics.metrics import Metrics
import matplotlib.pyplot as plt



corr_obj = CorrelationStructureGen(n_sets=2, signum=5, percentage=True, tot_corr=[100,75,50], corr_means=0.8 )  # swap signum and tot_dims
# nsets must be >=2
corr_truth, p_matrix, sigma_signals, R_matrix = corr_obj.get_structure()
datagen = MultisetDataGen_CorrMeans(p=p_matrix, R=R_matrix)   #combine struct gen and data gen
X, _ = datagen.generate()
corr_test = jointEVD(X, B=1000)
corr_estimate, d_cap, u_struc = corr_test.find_structure()
plt.ion()
viz = visualization(graph_matrix=corr_truth, num_dataset=4, label_edge=False)
viz.visualize("True Structure")
plt.ioff()
viz_op = visualization(graph_matrix=corr_estimate, label_edge=False)
viz_op.visualize("Estimated_structure")

# print("experiment complete")
print(f'the end')
