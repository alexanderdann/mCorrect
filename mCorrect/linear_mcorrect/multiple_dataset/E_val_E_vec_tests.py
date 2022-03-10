import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, block_diag
import scipy as sp
from mCorrect.utils.helper import arr_sort, list_find
from mCorrect.linear_mcorrect.multiple_dataset.hypothesis_test import Hypothesis_Test
from itertools import combinations


class jointEVD(object):
    """
    Description: This class contains the implementation of algorithms 1 and 2 of the paper:  T. Hasija, C. Lameiro, T. Marrinan
    and P. J. Schreier,"Determining the Dimension and Structure of the Subspace Correlated Across Multiple Data Sets."

    Given multiple datasets, each with multiple features, the technique estimates the number of correlated features
    accross the datasets and also their correlation structure. The technique uses bootstrapping based detection for
    estimating the number and structure of the correlated components.


    """

    def __init__(self, X_cell, P_fa_eval=0.05, P_fa_evec=0.05, B=1000, evec_threshold=0):
        """

        Args:
            X_cell (list of ndarrays): list containing the data samples of all the datasets. The i'th element of
                                    the list conststs of the data samples of the i'th dataset in the form of an ndarray.
            P_fa_eval (float): False alarm probability used for the eigen value test.
            P_fa_evec (float): False alarm probabiity used for the eigen vector test.
            B (int): Number of bootstrap sampling to be performed.
        """
        self.x_cell = X_cell
        self.P_fa_eval = P_fa_eval
        self.P_fa_evec = P_fa_evec
        self.B = B
        self.evev_threshold = evec_threshold
        self.M = X_cell[0].shape[1]

    def augmentData(self, data_cell):
        """

        Args:
            data_cell: (list of ndarray) : The multiple dataset

        Returns: Augmented array of all the datasets in the input multi-dataset.


        """
        P = len(data_cell)  # number of datasets
        M = data_cell[0].shape[1]  # number of samples
        x_aug = []
        m = []  # dimension of each dataset
        Rxx_mH = [0] * P
        for i in range(P):
            x_aug.extend(data_cell[i])
            data_cell_h = np.transpose(data_cell[i].conjugate())
            Rxx_mH[i] = sqrtm(inv(np.matmul(data_cell[i], data_cell_h) / M))  # verify
            m.append(data_cell[i].shape[0])
        return np.array(x_aug), Rxx_mH, m

    def generateInv_RD_cap(self, data_cell):
        """

        Args:
            data_cell: The multiple dataset

        Returns:

        """

        P = len(data_cell)  # number of datasets
        M = data_cell[0].shape[1]  # number of samples

        _, Rxx_mH, m = self.augmentData(data_cell)
        aug_dim = np.zeros(P)
        Rd_mh = Rxx_mH[0]
        aug_dim[0] = m[0]
        for i in range(1, P):
            Rd_mh = block_diag(Rd_mh, Rxx_mH[i])
            aug_dim[i] = aug_dim[i - 1] + m[i]

        return Rd_mh, aug_dim

    def generateR_xx_aug(self, data_cell):
        """
        Returns:  augmented data covariance matrix

        """
        M = data_cell[0].shape[1]
        X_aug, _, _ = self.augmentData(data_cell)
        X_aug_h = np.transpose(X_aug.conjugate())
        Rxx_aug = np.matmul(X_aug, X_aug_h) / self.M
        return Rxx_aug

    def generate_C(self, data_cell):
        """
        Returns: Augmented coherence matrix

        """
        R_cap = self.generateR_xx_aug(data_cell)
        R_d, aug_dim = self.generateInv_RD_cap(data_cell)
        C = np.matmul(R_d, np.matmul(R_cap, R_d))
        return C, aug_dim

    def calc_Eval_Evec(self, mat):
        """

        Args:
            mat (np.array): input square matrix

        Returns: eigen value and eigen vector sorted by absolute values of eigen values

        """
        E, U = np.linalg.eig(mat)
        E = np.array(list(map(abs, E)))  # absolute value, if complex eigen values
        E, idx = arr_sort(E, order='descending')
        U = U[:, idx]
        return E, U

    def bootstrap(self, data_cell, num_samples=0):
        """
        bootstraps samples out of data_cell and returns bootstrapped samples
        Args:
            num_samples (int):
            data_cell (cell): list of ndarrays for

        Returns: data cell i.e. list of datasets for different modalities bootstrapped from original dataset

        """
        P = len(data_cell)
        bs_cell = [0] * P
        M = data_cell[0].shape[1]  # number of samples
        idx_list = list(range(M))
        bs_idx = np.random.choice(idx_list, replace=True, size=M)

        for i in range(P):
            bs_cell[i] = data_cell[i][:, bs_idx]
            # make the bootstrapped samples zero mean by subtracting the mean from the samples
            temp1 = np.mean(bs_cell[i], 1)
            bs_mean = np.transpose(np.tile(temp1, (M, 1)))
            bs_cell[i] = bs_cell[i] - bs_mean

        return bs_cell

    def find_structure(self):
        """
        Function to estimate the number of correlated components and their structure.
        Returns: Number of correlated components 'd_cap' and the estimated correlation structure as a matrix. The correlation
        structre matrix 'corr_struc' is composed of the signals along the row axis and all possible pairs of datasets
        along the column axis.

        """
        Cxx_aug, aug_dim = self.generate_C(self.x_cell)
        E, U = self.calc_Eval_Evec(Cxx_aug)
        P = len(self.x_cell)  # number of datasets

        E_star_matrix = []
        U_star_matrix = []
        for b in range(self.B):
            x_cell_star = self.bootstrap(self.x_cell)
            Cxx_aug_star, _ = self.generate_C(x_cell_star)
            E_star, U_star = self.calc_Eval_Evec(Cxx_aug_star)
            E_star_matrix.append(E_star)
            U_star_matrix.append(U_star)

        E_star_matrix = np.array(E_star_matrix).T
        # U_star_matrix = U_star_matrix

        m_min = self.x_cell[0].shape[0]  # assuming all datasets have same num of features.
        d_cap = Hypothesis_Test().Eigen_value_test(P, m_min, self.P_fa_eval, E, E_star_matrix, self.B)
        U_struc = Hypothesis_Test().Eigen_vector_test(P, aug_dim, self.P_fa_evec, d_cap, U, U_star_matrix,
                                                      self.B, self.evev_threshold)

        # compute the correllation map
        x_corrs = list(combinations(range(P), 2))
        x_corrs = list(reversed(x_corrs))
        n_comb = len(x_corrs)

        corr_struc = np.zeros((n_comb, m_min))
        corr_struc[:, :d_cap] = np.ones((n_comb, d_cap))

        for s in range(d_cap):
            for p in range(P):
                if U_struc[s, p] == 0:
                    i1 = list_find(x_corrs, p)
                    for idx in range(len(i1)):
                        iz = i1[idx]
                        corr_struc[iz, s] = 0

        corr_struc = np.transpose(corr_struc)
        return corr_struc, d_cap

    def get_reduced_structure_matrix(self, n_sets, d_cap, corr_struc):
        """

        Args:
            d_cap: number of correlated components
            corr_struc: correlation structure matrix

        Returns: Reduced correlation structure matrix

        """
        corr_idx = list(combinations(range(n_sets), 2))
        corr_idx = list(reversed(corr_idx))
        u_struc = np.zeros((d_cap, n_sets))

        for i in range(d_cap):
            n_id = np.where(corr_struc[i, :] == 1)
            n_id = n_id[0]
            print(n_id)
            for j in range(len(n_id)):
                u_idx = corr_idx[n_id[j]]
                u_struc[i, u_idx[0]] = 1
                u_struc[i, u_idx[1]] = 1

        return u_struc

    def estimate_signals(self, data_cell, d_cap, corr_est):
        """

        Args:
            data_cell: Input multi-dataset as a list of ndarray.
            d_cap:  The number of correlated components in the multi-dataset
            u_struc: The correlation structure within the multi-dataset.

        Returns:

        """
        P = len(data_cell)
        M = data_cell[0].shape[1]
        _, Rxx_mH, m = self.augmentData(data_cell)
        u_struc = self.get_reduced_structure_matrix(n_sets=len(data_cell), d_cap=d_cap, corr_struc=corr_est)
        Cxx_aug, aug_dim = self.generate_C(data_cell)
        E, U = self.calc_Eval_Evec(Cxx_aug)
        Sig_estimate = []

        for i in range(d_cap):
            Sig = np.zeros((P, M))

            for p in range(P):
                if p == 0:
                    dim1 = 0
                else:
                    dim1 = int(aug_dim[p - 1])

                dim2 = int(aug_dim[p])
                u_i = U[dim1:dim2, i]

                if int(u_struc[i, p]) > 0.1:
                    Sig[p, :] = u_i @ Rxx_mH[p] @ np.array(data_cell[p])

            Sig_estimate.append(Sig)

        return Sig_estimate
