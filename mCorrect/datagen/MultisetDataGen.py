import numpy as np
from itertools import combinations
import random
import math
import scipy as sp
import scipy.linalg as spl
from mCorrect.utils.helper import ismember, comb, list_find


class MultisetDataGen_CorrMeans(object):
    """
    Description:
    This class implements the generation of multiple data sets with a prescribed correlation structure. It enforces
    the transitive correlation condition.

    """

    def __init__(self, corr_structure, tot_dims=None, mixing='orth', sigmad=10, sigmaf=3, snr=10, color='white',
                 M=300, MAcoeff=1, ARcoeff=1, Distr='gaussian'):
        """

        Args:
            corr_structure (CorrelationStructure): Object containing the correlation structure information of
            the generated correllation structure.

            x_corrs (array): List of tuples containing pair-wise combinations of the datasets.
            mixing (str): 'orth' or 'randn'. Describes the type of mixing matrix.
            snr (float): The required signal to noise ratio.
            color (str): 'white' for additive white noise and 'color' for colored noise.
            tot_dims (int): The required dimension of each dataset in the multi-dataset.
            The ith element of the jth row is the correlation coefficient between the ith signals in the data sets
            indicated by the jth row of self.x_corrs.
            sigmad (int): Standard deviation of the correlation coefficient of the correlated components.
            sigmaf (int): Standard deviation of the correlation coefficient of the independent components.
            snr (float): The required signal to noise ratio in the signals of the generated dataset.
            M (int): Number of samples per dataset
            MAcoeff (array): array of size 'degree of MA dependency' x 1. Moving average coefficients for colored noise.
            ARcoeff (array): array of size 'degree of AR dependency' x 1. Auto-regressive coefficients for colored noise.
            Distr (str): 'gaussian' or 'laplacian'. Specifies the distribution of the signal components.
        """

        self.n_sets = corr_structure.n_sets
        self.signum = corr_structure.signum

        if not tot_dims:
            self.tot_dims = self.signum
        else:
            self.tot_dims = tot_dims
        self.subspace_dims = np.array([self.tot_dims] * self.n_sets)
        self.x_corrs = list(combinations(range(self.n_sets), 2))
        self.x_corrs = list(reversed(self.x_corrs))
        self.mixing = mixing
        self.sigmaN = sigmad / (10 ** (0.1 * snr))
        self.color = color

        self.p = corr_structure.p
        self.sigma_signals = self.get_sigma_signals(self.p, sigmad, sigmaf)
        self.M = M
        self.MAcoeff = MAcoeff
        self.ARcoeff = ARcoeff
        self.Distr = Distr
        self.sigmad = sigmad
        self.sigmaf = sigmaf
        self.R = corr_structure.R
        self.A = [0] * self.n_sets
        self.S = [0] * self.n_sets
        self.N = [0] * self.n_sets
        self.X = [0] * self.n_sets

    def get_sigma_signals(self, p, sigmad, sigmaf):
        """
        Computes the matrix of standard deviations for the multi-dataset

        Returns: sigma_signals (ndarray): Matrix of size (n_sets x signum) x (n_sets x signum). Augmented block.
            correlation matrix of all the data sets. Each block is of size signum x signum and the i-jth block is the
            correlation matrix between data set i and data set j.
        """

        sigma_signals = np.ones((p.shape[0], p.shape[1]))*sigmaf
        sigma_signals[np.nonzero(p)] = sigmad
        return sigma_signals



    def generateMixingMatrix(self):
        """
        computes the mixing matrices

        """
        # print(self.mixing)
        if self.mixing == 'orth':
            for i in range(self.n_sets):
                if self.tot_dims >= self.signum:
                    orth_Q = spl.orth(np.random.randn(self.tot_dims,
                                                  self.signum))
                else:
                    orth_Q = spl.orth(np.transpose(np.random.randn(self.tot_dims,
                                                  self.signum)))
                self.A[i] = orth_Q

        elif self.mixing == 'randn':
            for i in range(self.n_sets):
                self.A[i] = np.random.randn(self.tot_dims, self.signum)  # replace with totdims

        else:
            raise Exception("Unknown mixing matrix property")

    def generateBlockCorrelationMatrix(self):
        """
        Compute the pairwise correlation and assemble the correlation matrices into augmented block correlation matrix

        """
        Rxy = [0] * comb(self.n_sets, 2)
        for i in range(len(self.x_corrs)):
            Rxy[i] = np.sqrt(np.diag(self.sigma_signals[i, :]) * np.diag(self.sigma_signals[i, :])) * np.diag(
                self.p[i, :])  # Assemble correlation matrices into augmented block correlation matrix
        for i in range(self.n_sets):
            t = np.zeros(len(self.x_corrs))
            idx = list_find(self.x_corrs, i)
            t[idx] = 1
            temp = self.sigma_signals[idx, :] == self.sigmad
            temp = temp.max(0)
            self.R[i * self.signum: (i + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = np.diag(
                temp * self.sigmad + np.logical_not(temp) * self.sigmaf)  # recheck the indices

            for j in range(i + 1, self.n_sets):  # check this again
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
        assert min(Ev) > 0, "negative eigen value !!! "
        print("R ready")

    def generateData(self):
        """
         Generates the data which is formed by mixing the signal with the desired type of noise
        """
        if self.Distr == 'gaussian':
            # evr, evec = np.linalg.eig(self.R)
            # evr = np.sort(evr)
            fullS = np.matmul(sp.linalg.sqrtm(self.R), np.random.randn(self.n_sets * self.signum, self.M))

        elif self.Distr == 'laplacian':
            # signum_aug = self.n_sets * self.signum
            # fullS = np.zeros(signum_aug, self.M)
            # for m in range(self.M):
            #     pass  # figure out how to generate laplacian samples in py
            # raise NotImplementedError

            fullS = np.matmul(sp.linalg.sqrtm(self.R), np.random.laplace(size=(self.n_sets * self.signum, self.M)))

        else:
            raise Exception("Unknown source distribution: {}".format(self.Distr))

        for i in range(self.n_sets):
            self.S[i] = fullS[i * self.signum: (i + 1) * self.signum, :]
            self.N[i] = np.sqrt(self.sigmaN) * np.random.randn(self.subspace_dims[i], self.M)

        # add a return if needed

    def filterNoise(self):
        """
        Filter the noise to be colored if specified

        """

        if self.color == 'white':
            return

        if self.color == 'colored':
            for i in range(self.n_sets):
                self.N[i] = sp.signal.lfilter(self.MAcoeff, self.ARcoeff, self.N[i])  # check for correctness

        else:
            raise Exception("Unknkown noise color option")

    def generateNoiseObservation(self):
        """
        Compute the final observation (signal + noise)
        Args:
        Returns:

        """

        for i in range(self.n_sets):
            self.X[i] = self.A[i] @ self.S[i] + self.N[i]

        #return self.X

    def generate(self):
        """
        This function retuns the generated data sets with a prescribed correlation structure.
        Returns: X (list of ndarrays): List of size n_sets x 1. The ith cell contains a matrix of size
        'ith element of subspace_dims'  by M.  It is the matrix of observations of the ith data set plus noise.

        R (ndarray): Matrix of size (n_sets x signum) x (n_sets x signum). Augmented block correlation matrix of all the
         data sets. Each block is of size signum x signum and the i-jth block is the correlation matrix between data set
         i and data set j.

        """

        self.generateMixingMatrix()
        # if self.R.all() == 0:
        #     self.R = np.zeros((self.n_sets * self.signum, self.n_sets * self.signum))
        #     self.generateBlockCorrelationMatrix()
        self.generateData()
        self.filterNoise()
        self.generateNoiseObservation()

        return self.X, self.R
