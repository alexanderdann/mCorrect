import numpy as np

from .helpers_CCA import calculate_rank_reduced_internal_representations, calculate_bartlett_lawley_statistic, \
    calculate_test_threshold, generate_sources, generate_observed_data
from .helpers_general import find_number_of_significantly_different_components, Nadakuditi
from scipy.stats import pearsonr as pearson


def detector_1(x, y, P_fa=0.01, labels=None, reduction_method='pca'):
    """
    Estimate the number of correlated components and reduced ranks.

    :param x: one observation, dim(x) = m x M, M: number of samples (subjects)
    :param y: the other observation, dim(y) = n x M
    :param P_fa: probability of false alarm
    :param labels: labels of the data, dim(labels) = M. If labels is None, assume 50% in each class
    :param reduction_method: 'pca' or 'pls'
    :return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat: estimated number of correlated components,
            estimated reduced ranks, estimated canonical correlation coefficients, estimated transformation matrices
    """

    # calculate parameters
    m = x.shape[0]
    n = y.shape[0]
    M = x.shape[1]
    r_max = np.min([M // 3, m, n])

    s_found = False
    s_min = 0
    d_hat = 0
    r_x_hat = 1
    r_y_hat = 1
    k_hat = 0
    T_x_hat = 0
    T_y_hat = 0
    for r_x_idx in range(1, r_max + 1):
        # for r_y_idx in range(1, r_max + 1):
        r_y_idx = r_x_idx

        # calculate rank reduced internal representations
        _, _, k, T_x, T_y = calculate_rank_reduced_internal_representations(x, y, r_x=r_x_idx, r_y=r_y_idx,
                                                                            labels=labels,
                                                                            reduction_method=reduction_method)

        # for current r: look for minimum s
        r = np.minimum(r_x_idx, r_y_idx)
        for s_idx in range(r):
            if calculate_bartlett_lawley_statistic(M, r_x_idx, r_y_idx, s_idx, k) < calculate_test_threshold(
                    r_x_idx,
                    r_y_idx,
                    s_idx,
                    P_fa):
                s_found = True
                s_min = s_idx
                # condition was not fulfilled for a smaller s than the current one, all other s will be bigger
                break
        if s_found is False:
            s_min = r
        else:
            s_found = False

        # for current r: if current minimum of s is higher than d_hat, overwrite value of d_hat and corresponding r_hat
        if s_min > d_hat:
            d_hat = s_min
            r_x_hat = r_x_idx
            r_y_hat = r_y_idx
            k_hat = k[:d_hat]
            T_x_hat = T_x
            T_y_hat = T_y

    return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat


def detector_2(x, y, labels=None, reduction_method='pca'):
    """
    Estimate the number of correlated components and reduced ranks.

    :param x: one observation, dim(x) = m x M, M: number of samples (subjects)
    :param y: the other observation, dim(y) = n x M
    :param labels: labels of the data, dim(labels) = M. If labels is None, assume 50% in each class
    :param reduction_method: 'pca' or 'pls'
    :return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat: estimated number of correlated components,
            estimated reduced ranks, estimated canonical correlation coefficients, estimated transformation matrices
    """

    # calculate parameters
    m = x.shape[0]
    n = y.shape[0]
    M = x.shape[1]
    r_max = np.min([M // 3, m, n])

    s_found = False
    s_min = 0
    d_hat = 0
    r_x_hat = 1
    r_y_hat = 1
    k_hat = 0
    T_x_hat = 0
    T_y_hat = 0
    for r_x_idx in range(1, r_max + 1):
        # for r_y_idx in range(1, r_max + 1):
        r_y_idx = r_x_idx

        # calculate rank reduced internal representations
        _, _, k, T_x, T_y = calculate_rank_reduced_internal_representations(x, y, r_x=r_x_idx, r_y=r_y_idx,
                                                                            labels=labels,
                                                                            reduction_method=reduction_method)

        # for current r: look for minimum s
        r = np.minimum(r_x_idx, r_y_idx)
        for s_idx in range(r):
            if M / 2 * np.log(np.prod(1 - k[s_idx:] ** 2) + 1e-20) > - np.log(M) / 2 * (r_x_idx - s_idx) * (
                    r_y_idx - s_idx):
                s_found = True
                s_min = s_idx
                # condition was not fulfilled for a smaller s than the current one, all other s will be bigger
                break
        if s_found is False:
            s_min = r
        else:
            s_found = False

        # for current r: if current minimum of s is higher than d_hat, overwrite value of d_hat and corresponding r_hat
        if s_min > d_hat:
            d_hat = s_min
            r_x_hat = r_x_idx
            r_y_hat = r_y_idx
            k_hat = k[:d_hat]
        T_x_hat = T_x
        T_y_hat = T_y

    return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat


def detector_3(x, y, s_a=None, s_b=None, labels=None, reduction_method='pca'):
    """
    Estimate the number of correlated components and reduced ranks.

    :param x: one observation, dim(x) = m x M, M: number of samples (subjects)
    :param y: the other observation, dim(y) = n x M
    :param labels: labels of the data, dim(labels) = M. If labels is None, assume 50% in each class
    :param reduction_method: 'pca' or 'pls'
    :return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat: estimated number of correlated components,
            estimated reduced ranks, estimated canonical correlation coefficients, estimated transformation matrices
    """

    # calculate parameters
    m = x.shape[0]
    n = y.shape[0]
    M = x.shape[1]
    r_max = np.min([M // 3, m, n])

    d_hat = 0
    r_x_hat = 1
    r_y_hat = 1
    k_hat = 0
    T_x_hat = 0
    T_y_hat = 0
    for r_x_idx in range(1, r_max + 1):

        # for r_y_idx in range(1, r_max + 1):
        #     print(r_x_idx, r_y_idx)
        r_y_idx = r_x_idx
        # calculate rank reduced internal representations
        _, _, k, T_x, T_y = calculate_rank_reduced_internal_representations(x, y, r_x=r_x_idx, r_y=r_y_idx,
                                                                            labels=labels,
                                                                            reduction_method=reduction_method)

        # for current r: look for minimum s
        I_mdl_min = np.inf
        s_min = np.inf
        r = np.minimum(r_x_idx, r_y_idx)
        for s_idx in range(r + 1):
            I_mdl = M / 2 * np.log(np.prod(1 - k[:s_idx] ** 2) + 1e-20) + np.log(M) / 2 * (
                    (r_x_idx * r_y_idx) - (r_x_idx - s_idx) * (r_y_idx - s_idx))
            if I_mdl < I_mdl_min:
                I_mdl_min = I_mdl
                s_min = s_idx

        # for current r: if current minimum of s is higher than d_hat, overwrite value of d_hat and corresponding r_hat
        if s_min > d_hat:
            d_hat = s_min
            r_x_hat = r_x_idx
            r_y_hat = r_y_idx
            k_hat = k[:d_hat]
            T_x_hat = T_x
            T_y_hat = T_y

    if s_a is not None:

        # find idx of significantly different rows
        a, b, k, T_x, T_y = calculate_rank_reduced_internal_representations(x, y, r_x=r_x_hat, r_y=r_y_hat)
        d_a, t_a, idx_a = find_number_of_significantly_different_components(a, labels=labels)
        d_b, t_b, idx_b = find_number_of_significantly_different_components(b, labels=labels)
        d_t = np.maximum(d_a, d_b)

        # calculate correlation between estimated component and true component
        correlation_a, _ = pearson(s_a, a[idx_a])
        correlation_b, _ = pearson(s_b, b[idx_b])
        correlation = np.mean([correlation_a, correlation_b])

        return d_t, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat, correlation

    else:
        return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat


def detector_t_test(x, y, s_a=None, s_b=None, labels=None, correlation_method='cca', reduction_method='pca'):
    """
    Estimate the number of correlated components and reduced ranks.

    :param x: one observation, dim(x) = m x M
    :param y: the other observation, dim(y) = n x M
    :param s_a: the correlated source component, dim(s_a) = 1 x M
    :param s_b: the correlated source component, dim(s_b) = 1 x M
    :param labels: labels of the data, dim(labels) = M. If labels is None, assume 50% in each class
    :param correlation_method: 'cca' or 'dcca'
    :param reduction_method: 'pca' or 'pls'
    :return d_hat, r_hat: estimated number of correlated components and estimated reduced rank
    """

    # calculate parameters
    m = x.shape[0]
    n = y.shape[0]
    M = x.shape[1]
    r_max = np.min([M // 3, m, n])
    d_hat = 0
    r_x_hat = 1
    r_y_hat = 1
    k_hat = 0
    T_x_hat = 0
    T_y_hat = 0
    t_max = - np.inf
    for r_x_idx in range(1, r_max + 1):
        # for r_y_idx in range(1, r_max + 1):
        r_y_idx = r_x_idx
        # calculate rank reduced internal representations
        a, b, k, T_x, T_y = calculate_rank_reduced_internal_representations(x, y, r_x=r_x_idx, r_y=r_y_idx,
                                                                            labels=labels,
                                                                            reduction_method=reduction_method,
                                                                            correlation_method=correlation_method)

        # find idx of significantly different rows
        d_a, t_a, idx_a = find_number_of_significantly_different_components(a, labels=labels)
        d_b, t_b, idx_b = find_number_of_significantly_different_components(b, labels=labels)

        # idx_a = find_idx_of_significantly_different_rows(a, labels=labels)
        # idx_b = find_idx_of_significantly_different_rows(b, labels=labels)

        # d = np.maximum(len(idx_a), len(idx_b))
        d = np.maximum(d_a, d_b)
        t = np.maximum(t_a, t_b)
        # print(r_x_idx, d, t)

        # for current r: if current number of significantly diffenrent components is higher than d_hat,
        # overwrite value of d_hat and corresponding others

        if t > t_max:
            t_max = t
            d_hat = d
            r_x_hat = r_x_idx
            r_y_hat = r_y_idx
            k_hat = k[:d_hat]
            T_x_hat = T_x
            T_y_hat = T_y
            a_hat = a
            idx_a_hat = idx_a
            b_hat = b
            idx_b_hat = idx_b
        elif t < t_max:
            break

    if s_a is not None:
        # calculate correlation between estimated component and true component
        correlation_a, _ = pearson(s_a, a_hat[idx_a_hat])
        correlation_b, _ = pearson(s_b, b_hat[idx_b_hat])
        correlation = np.mean([correlation_a, correlation_b])

        return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat, correlation
    else:

        return d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat


def detector_pca(x):
    """
    Detect the number of signals in noisy data with eigenvalue decomposition.

    :param x: data matrix of dimension n x M, (M samples)
    :return k_Nadakuditi: estimated number of signals
    """

    n = x.shape[0]
    M = x.shape[1]
    x_zero_mean = x - np.mean(x, axis=1)[:, None]
    U, sigma, Vh = np.linalg.svd(x_zero_mean, full_matrices=False)
    beta = 1  # real case
    r, _ = Nadakuditi(sigma ** 2, n, M, beta)

    if r == 0:
        T = 0
    else:
        T = U[:, :r].T

    return r, T


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

    # calculate probability
    iterations = 1
    detected = 0

    for idx in range(iterations):
        # generate sources
        s_x, s_y = generate_sources(d=d, sigma_x=sigma_x, sigma_y=sigma_y, f_x=f_x, f_y=f_y, rho=rho, M=M)

        # observations
        x, y = generate_observed_data(s_x=s_x, s_y=s_y, m=m, n=n)

        labels = np.zeros(M)
        labels[labels.shape[0] // 2:] = 1
        x = x - np.mean(x, axis=1)[:, None]
        y = y - np.mean(y, axis=1)[:, None]

        # estimate d_hat
        d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat = detector_1(x, y, P_fa)
        # d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat = detector_2(x,y)
        # d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat = detector_3(x, y)
        # d_hat, r_x_hat, r_y_hat, k_hat, T_x_hat, T_y_hat = detector_t_test(x, y)
        print(d_hat, r_x_hat, r_y_hat)
        if d_hat == d:
            detected += 1

    # probability of detection
    p_detect = detected / iterations

    print(p_detect)

    r_hat, T_hat = detector_pca(x)

    print(r_hat)
