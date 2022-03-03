import numpy as np
from scipy import linalg as lg
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
from .helpers_general import pls
import tikzplotlib


def generate_sources(d=3, sigma_x=1, sigma_y=1, f_x=0, f_y=0, rho=np.array([0.9, 0.8, 0.7]), M=200):
    """
    Generate source vectors s_x, s_y. s_x / s_y each consist of d correlated sources and f_x / f_y independent sources.

    :param d: number of correlated sources between s_x and s_y
    :param f_x: number of independent sources in s_x
    :param f_y: number of independent sources in s_y
    :param sigma_x: standard deviation of sources in s_x
                    - one value if it is equal for all sources
                    - or an array of length (d+f_x) with the variance of every source (first variance of correlated)
    :param sigma_y: standard deviation of sources in s_y
                    - one value if it is equal for all sources
                    - or an array of length (d+f_y) with the variance of every source (first variance of correlated)
    :param rho: correlation coefficients between correlated sources in x and y
                    - one value if it is equal for all correlated sources
                    - or an array of length d with the correlation coefficients between all correlated sources
    :param M: number of samples (subjects)
    :return s_x, s_y: Source vectors of dimension (d+f_x) x M, (d+f_y) x M
    """

    # test shapes

    if type(sigma_x) is not np.ndarray:
        sigma_x = np.ones(d + f_x) * sigma_x
    else:
        assert (d + f_x == sigma_x.shape[0]), 'sigma_x has incorrect shape'

    if type(sigma_y) is not np.ndarray:
        sigma_y = np.ones(d + f_y) * sigma_y
    else:
        assert (d + f_y == sigma_y.shape[0]), 'sigma_y has incorrect shape'

    if type(rho) is not np.ndarray:
        rho = np.ones(d) * rho
    else:
        assert (d == rho.shape[0]), 'rho has incorrect shape'

    # auto covariance matrices
    R_sxsx = np.eye(d + f_x) * sigma_x ** 2
    R_sysy = np.eye(d + f_y) * sigma_y ** 2

    # cross covariance matrix
    R_sxsy = np.zeros((d + f_x, d + f_y))

    R_sxsy[:d, :d] = np.diag(rho * sigma_x[:d] * sigma_y[:d])

    # sources are jointly gaussian

    # jointly covariance matrix
    R = np.concatenate((np.hstack((R_sxsx, R_sxsy)), np.hstack((np.conj(R_sxsy.T), R_sysy))))
    # jointly mean
    mu = np.zeros(f_x + d + f_y + d)

    # draw source vector of joint distribution
    s = np.random.multivariate_normal(mu, R, M).T

    # divide in sx and sy
    s_x = s[:d + f_x, :]
    s_y = s[d + f_x:, :]

    return s_x, s_y


def generate_observed_data(s_x, s_y, m=10, n=10, sigma_nx=0.2, sigma_ny=0.2):
    """
    Generate observations x and y (mixtures of sources).

    :param s_x: source vector of dimension (d+f_x) x M, M: number of samples (subjects)
    :param s_y: source vector of dimension (d+f_y) x M
    :param m: dimension of x (x has dimension m x M)
    :param n: dimension of y (y has dimension n x M)
    :param sigma_nx: standard deviation of noise in x
    :param sigma_ny: standard deviation of noise in y
    :return x, y: observations x,y
    """

    # number of samples
    M = s_x.shape[1]

    # mixing matrices
    A_x = np.random.randn(m, s_x.shape[0])
    A_y = np.random.randn(n, s_y.shape[0])

    # noise
    n_x = np.random.randn(m, M) * sigma_nx
    n_y = np.random.randn(n, M) * sigma_ny

    # observed signals
    x = A_x @ s_x + n_x
    y = A_y @ s_y + n_y

    return x, y


def calculate_rank_reduced_internal_representations(x, y, r_x=None, r_y=None, labels=None,
                                                    reduction_method='pca', correlation_method='cca'):
    """
    Perform a dimension reduction and then a CCA.
    Calculate the rank reduced internal representations a = S x_rx and b = T y_ry, such that a @ b.H = diag(k).
    Also calculate canonical correlations k.

    For reference, see
    @article{song2016canonical,
    title={Canonical correlation analysis of high-dimensional data with very small sample support},
    author={Song, Yang and Schreier, Peter J and Ram{\'\i}rez, David and Hasija, Tanuj},
    url={https://arxiv.org/pdf/1604.02047.pdf}
    }

    :param x: one observation, dim(x) = m x M, M: number of samples (subjects)
    :param y: the other observation, dim(y) = n x M
    :param r_x: reduced rank of x
    :param r_y: reduced rank of y
    :param labels: (only necessary for PLS or DCCA) labels for each sample of x, dim(labels) = 1 x M
    :param reduction_method: 'pca' or 'pls'. 'pls is slower and can only be used if labels is not None
    :return a, b, k, T_x, T_y: internal representations of dim p x M with p = min(r_x,r_y),
            p sample canonical correlations, transformation matrices
    """

    # svd
    x_zero_mean = x - np.mean(x, axis=1)[:, None]
    y_zero_mean = y - np.mean(y, axis=1)[:, None]
    U_x, sigma_x, Vh_x = np.linalg.svd(x_zero_mean, full_matrices=False)
    U_y, sigma_y, Vh_y = np.linalg.svd(y_zero_mean, full_matrices=False)

    # number of samples
    M = x.shape[1]

    # define r_x, r_y. If not defined, they are min(m,M//3), min(n, M//3)
    if r_x is None:
        r_x = np.minimum(x.shape[0], M // 3)
    if r_y is None:
        r_y = np.minimum(y.shape[0], M // 3)
    assert r_x <= x.shape[0], 'r_x is ' + str(r_x) + ', maximum of r_x is ' + str(x.shape[0])
    assert r_y <= y.shape[0], 'r_y is ' + str(r_y) + ', maximum of r_y is ' + str(y.shape[0])

    # dimension reduction with pca
    if reduction_method == 'pca':

        # rank reduced PCA descriptions of x, y
        x_rx = np.conj(U_x[:, :r_x].T) @ x_zero_mean
        y_ry = np.conj(U_y[:, :r_y].T) @ y_zero_mean

    # dimension reduction with pls
    else:
        assert labels is not None, "'labels' must not be None"

        x_rx = pls(x, labels)[:r_x, :]
        y_ry = pls(y, labels)[:r_y, :]

    # CCA

    # covariance and cross covariance matrices of rank reduced x,y
    C_xx_tilde = 1 / M * x_rx @ np.conj(x_rx.T)
    C_yy_tilde = 1 / M * y_ry @ np.conj(y_ry.T)

    if correlation_method == 'cca':

        # coherence matrix
        C_tilde = Vh_x[:r_x, :] @ np.conj(Vh_y.T)[:, :r_y]

        # svd
        F, k, Gh = np.linalg.svd(C_tilde, full_matrices=False)

    elif correlation_method == 'dcca':

        assert np.allclose(x, x_zero_mean), "x must be zero mean"
        assert np.allclose(y, y_zero_mean), "x must be zero mean"
        assert labels is not None, "'labels' must not be None"

        # A matrix
        unique, counts = np.unique(labels, return_counts=True)
        A_list = []
        for (class_idx, count_idx) in zip(unique.astype(int), range(unique.shape[0])):
            A_list.append(np.ones((counts[count_idx], counts[count_idx])))
        A = lg.block_diag(*A_list)

        # H matrix
        H = 1 / np.linalg.norm(A) ** 2 * np.linalg.inv(lg.sqrtm(C_xx_tilde)) @ x_rx @ A @ y_ry.T @ np.linalg.inv(
            lg.sqrtm(C_yy_tilde))

        # svd
        F, k, Gh = np.linalg.svd(H, full_matrices=False)

        # # find last gap in singular values before values are equal
        # w_diff = np.diff(k)
        # w_threshold_p = w_diff.mean() + 3 * w_diff.std()
        # w_threshold_n = w_diff.mean() - 3 * w_diff.std()
        #
        # n = k.shape[0]
        # for w_idx in range(w_diff.shape[0], 0, -1):
        #     if w_diff[w_idx - 1] > w_threshold_p or w_diff[w_idx - 1] < w_threshold_n:
        #         n = w_idx
        #         break

        # n_classes-1 singular values are unequal to 0
        n = unique.shape[0] - 1

        F = F[:, 0:n]
        k = k[0:n]
        Gh = Gh[0:n, :]
    else:
        raise Exception("Choose 'cca' or 'dcca' as correlation method.")

    # linear transformations
    S = np.conj(F.T) @ np.linalg.inv(lg.sqrtm(C_xx_tilde))
    T = Gh @ np.linalg.inv(lg.sqrtm(C_yy_tilde))

    # internal representations
    a = S @ x_rx
    b = T @ y_ry

    # transformation matrices
    T_x = S @ np.conj(U_x[:, :r_x].T)
    T_y = T @ np.conj(U_y[:, :r_y].T)

    return a, b, k, T_x, T_y


def calculate_internal_representations(x, y, labels=None, correlation_method='cca'):
    """
    Perform a CCA.
    Calculate the internal representations a = S x and b = T y, such that a @ b.H = diag(k).
    Also calculate canonical correlations k.

    For reference, see
    @article{song2016canonical,
    title={Canonical correlation analysis of high-dimensional data with very small sample support},
    author={Song, Yang and Schreier, Peter J and Ram{\'\i}rez, David and Hasija, Tanuj},
    url={https://arxiv.org/pdf/1604.02047.pdf}
    }

    :param x: one observation, dim(x) = m x M, M: number of samples (subjects)
    :param y: the other observation, dim(y) = n x M
    :return a, b, k, S, T: internal representations of dim p x M with p = min(m,n), p sample canonical correlations,
                           transformation matrices
    """

    x_zero_mean = x - np.mean(x, axis=1)[:, None]
    y_zero_mean = y - np.mean(y, axis=1)[:, None]

    # number of samples
    M = x.shape[1]

    # covariance and cross covariance matrices
    C_xx = 1 / M * x_zero_mean @ np.conj(x_zero_mean.T)
    C_xy = 1 / M * x_zero_mean @ np.conj(y_zero_mean.T)
    C_yy = 1 / M * y_zero_mean @ np.conj(y_zero_mean.T)

    if correlation_method == 'cca':

        # coherence matrix
        C = np.linalg.inv(lg.sqrtm(C_xx)) @ C_xy @ np.linalg.inv(lg.sqrtm(C_yy))
        # svd
        F, k, Gh = np.linalg.svd(C, full_matrices=False)

    elif correlation_method == 'dcca':

        assert np.allclose(x, x_zero_mean), "x must be zero mean"
        assert np.allclose(y, y_zero_mean), "x must be zero mean"
        assert labels is not None, "'labels' must not be None"

        # A matrix
        unique, counts = np.unique(labels, return_counts=True)
        A_list = []
        for (class_idx, count_idx) in zip(unique.astype(int), range(unique.shape[0])):
            A_list.append(np.ones((counts[count_idx], counts[count_idx])))
        A = lg.block_diag(*A_list)

        # H matrix
        H = 1 / np.linalg.norm(A) ** 2 * np.linalg.inv(
            lg.sqrtm(C_xx)) @ x_zero_mean @ A @ y_zero_mean.T @ np.linalg.inv(lg.sqrtm(C_yy))

        # svd
        F, k, Gh = np.linalg.svd(H, full_matrices=False)

        # # find last gap in singular values before values are equal
        # w_diff = np.diff(k)
        # w_threshold_p = w_diff.mean() + 3 * w_diff.std()
        # w_threshold_n = w_diff.mean() - 3 * w_diff.std()
        #
        # n = k.shape[0]
        # for w_idx in range(w_diff.shape[0], 0, -1):
        #     if w_diff[w_idx - 1] > w_threshold_p or w_diff[w_idx - 1] < w_threshold_n:
        #         n = w_idx
        #         break

        # n_classes-1 singular values are unequal to 0
        n = unique.shape[0] - 1

        F = F[:, 0:n]
        k = k[0:n]
        Gh = Gh[0:n, :]

    else:
        raise Exception("Choose 'cca' or 'dcca' as correlation method.")

    # linear transformations
    S = np.conj(F.T) @ np.linalg.inv(lg.sqrtm(C_xx))
    T = Gh @ np.linalg.inv(lg.sqrtm(C_yy))

    # internal representations
    a = S @ x_zero_mean
    b = T @ y_zero_mean

    return a, b, k, S, T


def plot_k(k):
    """
    Plot the canonical correlations k.

    :param k: array containing the canonical correlations
    """

    plt.figure()
    plt.plot(k, '*')
    plt.ylim(-0.1, 1.1)
    plt.xticks(np.arange(len(k)), np.arange(1, len(k) + 1))
    plt.show()
    plt.close()


def calculate_bartlett_lawley_statistic(M, r_x, r_y, s, k):
    """
    Calculate the Bartlett Lawley statistic for a specific model order s and specific ranks r_x, r_y.

    :param M: number of samples (subjects)
    :param r_x: rank reduced dimension of x
    :param r_y: rank reduced dimension of y
    :param s: estimated model order
    :param k: canonical correlations for
    :return C: Bartlett Lawley statistic for specific s, r_x, r_y
    """

    C = -1 * (M - s - (r_x + r_y + 1) / 2 + np.sum(k[:s] ** (-2))) * np.log(np.prod(1 - k[s:] ** 2) + 1e-20)
    return C


def calculate_test_threshold(r_x, r_y, s, P_fa=0.01):
    """
    Calculate the test threshold for a given model order, ranks and false alarm probability.

    :param r_x: rank reduced dimension of x
    :param r_y: rank reduced dimension of y
    :param s: estimated model order
    :param P_fa: probability of false alarm
    :return T: test threshold
    """

    # degrees of freedom
    df = 1 * (r_x - s) * (r_y - s)

    # test threshold
    T = chi2.ppf(1 - P_fa, df=df)

    return T


def plot_test_statistic_histogram(d, f_x, f_y, s, r_x, r_y, rho=np.array([0.9, 0.8, 0.7]), M=200, m=10, n=10,
                                  sigma_x=1, sigma_y=1, sigma_nx=0.2, sigma_ny=0.2, P_fa=0.01, iterations=1000,
                                  save=False):
    """
    Plot the probability density function of chi^2, the histogram of the Bartlett Lawley statistic and the test
    threshold for given parameters.

    :param d: number of correlated sources between s_x and s_y
    :param f_x: number of independent sources in s_x
    :param f_y: number of independent sources in s_y
    :param s: estimated number of correlated sources
    :param r_x: reduced rank of x
    :param r_y: reduced rank of y
    :param rho: correlation coefficients between correlated sources in x and y
                    - one value if it is equal for all correlated sources
                    - or an array of length d with the correlation coefficients between all correlated sources
    :param M: number of samples (subjects)
    :param m: dimension of x (x has dimension m x M)
    :param n: dimension of y (y has dimension n x M)
    :param sigma_x: standard deviation of sources in s_x
                    - one value if it is equal for all sources
                    - or an array of length (d+f_x) with the variance of every source (first variance of correlated)
    :param sigma_y: standard deviation of sources in s_y
                    - one value if it is equal for all sources
                    - or an array of length (d+f_y) with the variance of every source (first variance of correlated)
    :param sigma_nx: standard deviation of noise in x
    :param sigma_ny: standard deviation of noise in y
    :param P_fa: probability of false alarm
    :param iterations: number of points in the histogram
    :param save: if the image should be saved
    """

    # test threshold
    T = calculate_test_threshold(r_x, r_y, s, P_fa)

    # degrees of freedom
    df = 1 * (r_x - s) * (r_y - s)

    # store values of Bartlett Lawley statistic
    C_list = []

    for i in range(iterations):
        # generate sources
        s_x, s_y = generate_sources(d=d, sigma_x=sigma_x, sigma_y=sigma_y, f_x=f_x, f_y=f_y, rho=rho, M=M)

        # observations
        x, y = generate_observed_data(s_x, s_y, m, n, sigma_nx, sigma_ny)

        # calculate rank reduced internal representations
        a, b, k, _, _ = calculate_rank_reduced_internal_representations(x, y, r_x=r_x, r_y=r_y)

        # calculate Bartlett Lawley statistic
        C = calculate_bartlett_lawley_statistic(M, r_x, r_y, s, k)
        C_list.append(C)

    # calculate normalized histogram of C
    hist, bin_edges = np.histogram(np.array(C_list), bins=100, density=True)

    # center bins
    centered_bins = np.convolve(bin_edges, np.ones((2,)) / 2, mode='valid')

    # plot
    plt.figure()
    x = np.arange(centered_bins.min() * 0.8, centered_bins.max() * 1.3, 0.1)
    plt.semilogx(centered_bins, hist, label='$C(%i,%i,%i)$' % (r_x, r_y, s))
    plt.semilogx(x, chi2.pdf(x, df), '--', label='$\chi^2_{(%i-%i)(%i-%i)}$' % (r_x, s, r_y, s))
    plt.xlim(x.min(), x.max())
    plt.axvline(T, ymin=0, ymax=0.2, color='red', linewidth=3, label='$T(%i,%i,%i)$' % (r_x, r_y, s))
    plt.legend()
    plt.grid()
    if save:
        tikzplotlib.save('C_chi_f_%i_d_%i_r_%i_s_%i.tex' % (f_x, d, r_x, s))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # number of time points
    M = 50

    # number of correlated sources
    d = 2

    # number of independent sources
    f_x = 3
    f_y = 4

    # standard deviations
    sigma_x = np.array([np.sqrt(5), np.sqrt(5), np.sqrt(7), np.sqrt(1.5), np.sqrt(1.5)])
    sigma_y = np.array([np.sqrt(5), np.sqrt(5), np.sqrt(7), np.sqrt(6), np.sqrt(1.5), np.sqrt(1.5)])

    # dimension of data m=n (number of observations in x)
    m = 40
    n = 40

    # correlation coefficients
    rho = np.array([0.8, 0.7])

    # probability of false alarm
    P_fa = 0.005

    s_x, s_y = generate_sources(d=d, sigma_x=sigma_x, sigma_y=sigma_y, f_x=f_x, f_y=f_y, rho=rho, M=M)

    # observations
    x, y = generate_observed_data(s_x=s_x, s_y=s_y, m=m, n=n)
    calculate_rank_reduced_internal_representations(x, y, 8, 8, correlation_method='cca')

    # plot_test_statistic_histogram(d,f_x,f_y, d_hat, r_hat, r_hat, rho,M, m, n ,sigma_x, sigma_y, P_fa=P_fa, iterations=10000)
