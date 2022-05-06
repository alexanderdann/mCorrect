import numpy as np
from itertools import combinations
import random
import math
import scipy as sp
import scipy.linalg as spl
from mCorrect.visualization.graph_visu import visualization
from mCorrect.utils.helper import ismember, comb, list_find


class CorrelationStructure:

    def __init__(self, n_sets, signum, p, R, corr_truth):
        self.n_sets = n_sets
        self.signum = signum
        self.p = p
        self.R = R
        self.corr_truth = corr_truth


class CorrelationStructureGen:
    """
    Description:
    This function generates a correlation structure and augmented covariance matrix for multiple data sets. Correlation
    coefficients for each signal are randomly selected from a normal distribution with a user prescribed mean and
    standard deviation. This function enforces the 'transitive correlation condition.'

    Use the generate() function to create the desired correlation structure.

    """

    def __init__(self, n_sets=4, signum=5, tot_corr=None, tot_dims=None, corr_means=None, corr_std=None, sigmad=10,
                 sigmaf=3, maxIters=100, percentage=False):
        """

        Args:
            n_sets (int): Number of datasets.
            tot_corr (int): Number of signals/features of the datasets correlated across all datasets.
            corr_means (list):   mean of the correlation coefficients associated with each of the correlated signals.
            corr_std (list):  standard deviation of the correlation coefficients associated with each of the correlated signals.
            signum (int): Total number of signals in each dataset.
            sigmad (float): Variance of the correlated components.
            sigmaf (float): Variance of the independent components.
            maxIters (int): Number of random draws of correlation coefficients allowed to find a positive definite
                            correlation matrix.
        """

        self.corrnum = len(tot_corr)
        self.n_sets = n_sets
        assert n_sets >= 2, "a minimum of 2 datasets need to be present in order to perform correlation analysis. "
        if not tot_dims:
            tot_dims = signum
        self.tot_dims = tot_dims
        if len(tot_corr) > signum:
            print(f"tot correllations requested is greater than the number of signals. clipping the excess values.")
            tot_corr = tot_corr[:signum]
        if not percentage:
            self.tot_corr = np.array(tot_corr)
        else:
            self.tot_corr = self.calc_input_corr_vals_from_percent(tot_corr)

        self.x_corrs = list(combinations(range(self.n_sets), 2))
        self.x_corrs = list(reversed(self.x_corrs))
        self.n_combi = len(self.x_corrs)
        self.corr_means = corr_means  # np.array([corr_means] * len(tot_corr))
        self.corr_std = corr_std  # np.array([corr_std] * len(tot_corr))
        if corr_means == None:
            self.corr_means = [0.8] * len(tot_corr)
        if corr_std == None:
            self.corr_std = [0.01] * len(tot_corr)

        assert len(tot_corr) == len(self.corr_means) == len(
            self.corr_std), "corr_means and corr_std must have same dimension as tot_corr "

        self.signum = signum
        self.sigmad = sigmad
        self.sigmaf = sigmaf
        self.maxIters = maxIters
        self.R = np.zeros((self.n_sets * self.signum, self.n_sets * self.signum))

        assert self.corrnum == np.shape(self.corr_means)[0] == np.shape(self.corr_std)[
            0], "error('\n Dimension mismatch in corrnum, corr_means and corr_std)"

    def calc_input_corr_vals_from_percent(self, corr_list):
        corr_vec = []
        for ele in corr_list:
            ca = int((ele / 100) * self.n_sets)
            corr_vec.append(ca)
        return np.array(corr_vec)

    def generateBlockCorrelationMatrix(self, sigma_signals, p):
        """
                Compute the pairwise correlation and assemble the correlation matrices into augmented block correlation matrix
                Returns:

                """
        Rxy = [0] * comb(self.n_sets, 2)
        for i in range(len(self.x_corrs)):
            Rxy[i] = np.sqrt(np.diag(sigma_signals[i, :]) * np.diag(sigma_signals[i, :])) * np.diag(
                p[i, :])

        # Assemble correlation matrices into augmented block correlation matrix
        for i in range(self.n_sets):
            t = np.zeros(len(self.x_corrs))
            idx = list_find(self.x_corrs, i)
            t[idx] = 1
            temp = sigma_signals[idx, :] == self.sigmad
            temp = temp.max(0)
            self.R[i * self.signum: (i + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = np.diag(
                temp * self.sigmad + np.logical_not(temp) * self.sigmaf)

            for j in range(i + 1, self.n_sets):
                a = np.zeros(len(self.x_corrs))
                b = np.zeros(len(self.x_corrs))
                idxa = list_find(self.x_corrs, i)
                idxb = list_find(self.x_corrs, j)
                a[idxa] = 1
                b[idxb] = 1
                # a = np.sum(self.x_corrs == i, 1)
                # b = np.sum(self.x_corrs == j, 2)
                c = np.nonzero(np.multiply(a, b))
                self.R[i * self.signum: (i + 1) * self.signum, j * self.signum: (j + 1) * self.signum] = Rxy[int(c[0])]
                self.R[j * self.signum: (j + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = Rxy[int(c[0])]
        Ev, Uv = np.linalg.eig(self.R)
        # assert min(Ev) > 0, "negative eigen value !!! "

        return self.R

    def get_structure(self):
        """

        Returns: An object CorrelationStructure containing the following as attributes.
                n_sets: the total number of datasets.
                signum: the number of signals in each dataset.
                p matrix : the matrix of correlations coefficients between the signals in the multi-dataset
                R matrix:  Augmented block correlation matrix of all the data sets.
                corr_truth: The ground truth correlation structure of the generated dataset.
        """
        max_attempts = 5
        attempts = 0
        ans = "n"
        u_struc = 0

        while ans != "y" and attempts < max_attempts:
            p, sigma_signals, R = self.generate()

            corr_truth = np.zeros((self.n_combi, self.tot_dims))  # self.tot_dims
            idx_c = np.nonzero(p)
            corr_truth[idx_c] = 1  # this is the ground truth correllation.
            corr_truth = np.transpose(corr_truth)
            # visualize input correllation structure
            if True:
                viz = visualization(graph_matrix=np.transpose(p), num_dataset=self.n_sets)
                viz.visualize("Generated corr structure")
                ans = input("Continue with generated correlation structure?: y/n")
            attempts += 1
        if attempts >= max_attempts:
            print("Maximum retries exceeded. Proceeding with previously generated structure")

        #return corr_truth, np.transpose(p), R
        return CorrelationStructure(self.n_sets, self.signum, np.transpose(p), R, corr_truth)


    def generate(self):
        """
        Generates the correlation structure of the multi dataset.
        Returns:

            p (ndarray): Matrix of size 'n_sets choose two' x signum. Rows have the same order as x_corrs.
            The ith element of the jth row is the correlation coefficient between the ith signals in the data sets
            indicated by the jth row of self.x_corrs.

            sigma_signals (ndarray): Matrix of size (n_sets x signum) x (n_sets x signum). Augmented block
            correlation matrix of all the data sets. Each block is of size signum x signum and the i-jth block is the
            correlation matrix between data set i and data set j.

            R (ndarray): Matrix of size (n_sets x signum) x (n_sets x signum). Augmented block correlation matrix of
            all the data sets. Each block is of size signum x signum and the i-jth block is the correlation matrix
            between data set i and data set j.

        """

        minEig = -1
        attempts = 0

        while minEig <= 0:
            p = np.zeros((self.n_combi, self.signum))
            sigma_signals = np.zeros((self.n_combi, self.signum))

            corr_samples = []
            for j in range(self.corrnum):
                t = random.sample(list(range(self.n_sets)), int(self.tot_corr[j]))
                corr_samples.append(t)
                t1 = ismember(self.x_corrs, t)
                t2 = t1.sum(axis=1) == 2

                temp = self.corr_means[j] + self.corr_std[j] * np.random.randn(t2.sum(), 1)
                corr_arr = [0] * len(t2)
                idx = 0
                for k in range(len(t2)):
                    if t2[k] == 1:
                        p[k, j] = max(min(temp[idx], 1), 0)
                        sigma_signals[k, j] = self.sigmad
                        idx += 1
                    else:
                        p[k, j] = 0
                        sigma_signals[k, j] = self.sigmaf

            if self.corrnum < self.signum:
                sigma_signals[:, self.corrnum: self.signum] = self.sigmaf * np.ones(
                    (self.n_combi, self.signum - self.corrnum))
            # minEig = 1

            R = self.generateBlockCorrelationMatrix(sigma_signals, p)

            attempts += 1
            e, ev = np.linalg.eig(self.R)
            minEig = np.min(e)
            if attempts > self.maxIters and minEig < 0:
                raise Exception("A positive definite correlation matrix could not be found with prescribed correlation "
                                "structure. Try providing a different correlation structure or reducing the standard "
                                "deviation")

        return p, sigma_signals, R
