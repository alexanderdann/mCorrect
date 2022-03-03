import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import stats
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def find_idx_of_significantly_different_rows(x, labels=None, d=4):
    """
    Divide a row of x in two halves: use label information or two equal halfs.
    If the first half is significantly different from the second half
    (that means t test is passed), save the idx of the corresponding row.
    If the t test is not passed for any row, save the idx of the d rows with the highest t value.

    :param x: array for that the t test is done for every row, dim(x) = d (dimensions) x M (subjects)
    :return keep_idx: list containing all indices of significantly different rows
    """

    # save all indizes for that the t values pass the t test
    keep_idx = []
    t = np.zeros(x.shape[0])

    # find idx that separates both classes
    if labels is None:
        mid_point = x.shape[1] // 2
    else:
        # index where values start changing (0th index: change because of rolling)
        mid_point = np.where(np.roll(labels, 1) != labels)[0][1]

    for idx in range(x.shape[0]):
        a = x[idx, 0:mid_point]
        b = x[idx, mid_point:]
        # use t test for related distributions
        t[idx], p_o = stats.ttest_ind(a, b)
        if p_o < 0.05:
            keep_idx.append(idx)

    # if there is no such t value, save the index of the heighest t value
    if not keep_idx:
        if d == 0:
            keep_idx = [np.argsort(-np.abs(t))[d]]
        else:
            keep_idx = list(np.argsort(-np.abs(t))[:d])

    return keep_idx


def find_number_of_significantly_different_components(x, labels=None):
    """
    Divide a row of x in two halves: use label information or two equal halfs.
    If the first half is significantly different from the second half
    (that means t test is passed), save the idx of the corresponding row.
    If the t test is not passed for any row, save the idx of the row with the highest t value.

    :param x: array for that the t test is done for every row, dim(x) = d (dimensions) x M (subjects)
    :param labels: if it is not the same ratio for each class, use labels to see which values belong to which class
    :return keep_idx: list containing all indices of significantly different rows
    """

    # save all indizes for that the t values pass the t test
    d = 0
    keep_idx = []
    t = np.zeros(x.shape[0])
    p_o = np.zeros(x.shape[0])

    # find idx that separates both classes
    if labels is None:
        mid_point = x.shape[1] // 2
    else:
        # index where values start changing (0th index: change because of rolling)
        mid_point = np.where(np.roll(labels, 1) != labels)[0][1]

    for idx in range(x.shape[0]):
        a = x[idx, 0:mid_point]
        b = x[idx, mid_point:]
        # use t test for related distributions
        t[idx], p_o[idx] = stats.ttest_ind(a, b)
        if p_o[idx] < 0.05:
            d += 1
            keep_idx.append(idx)

    if not keep_idx:
        t_max = 0
        t_idx = np.argmax(np.abs(t))
    else:
        t_max = np.amax(np.abs(t[keep_idx]))
        t_idx = np.where(t_max == np.abs(t))[0][0]

    return d, t_max, t_idx


def running_mean(x):
    x_with_values = x[~np.isnan(x)]
    x_mean = np.zeros(x_with_values.shape[0])
    for n in range(x_with_values.shape[0]):
        x_mean[n] = np.mean(x_with_values[:n + 1])
    return x_mean


def Nadakuditi(L, n, m, beta=1):
    """
    Returns estimated PCA rank.

    :param L: vector containing eigenvalues of xx.T or squared singular values of x
    :param n: x.shape[0]
    :param m: x.shape[1]
    :param beta: 1 for real case, 2 for complex case
    :return Kmin, obj: estimated PCA rank, vector containing values of objective for each k
    """

    p = np.minimum(n, m)
    obj = np.zeros(p)

    for k in range(p):
        tk = ((np.sum(L[k:n] ** 2) / np.sum(L[k:n]) ** 2) * (n - k) - (1 + n / m)) * n - (2 / beta - 1) * (n / m)
        obj[k] = ((beta / 4) * ((m / n) ** 2) * tk ** 2) + 2 * (k + 1)

    Kmin = np.argmin(obj)

    return Kmin, obj


def pca(x, r=None):
    """
    Calculate the transformation of x with PCA.

    :param x: data matrix of dimensions m x M
    :param r: reduced rank, if None: return transformed x without dimension reduction
    :return x_transformed: transformation of x
    """

    U, sigma, Vh = np.linalg.svd(x, full_matrices=True)
    if r is None:
        x_transformed = U.T @ x
        return x_transformed, U.T
    else:
        x_transformed = U[:, 0:r].T @ x
        return x_transformed, U[:, 0:r].T


def pls(x, labels):
    """
    Implement PLS1/2. Assumptions: score vectors are good predictors of y. Mutual orthogonality of t is guarantied.
    Finds transformated matrices by including label information -> supervised technique.

    :param X: datamatrix to reduce dimension, dim(X) = m x M
    :param y: labels of the data_matrix, dim(Y) = 1 x M
    :return t_matrix: transformed datamatrix
    """

    # store copy to not change inputs
    X = x - np.mean(x, axis=1)[:, None]
    # fit shape of labels
    if len(labels.shape) == 1:
        y = (labels - np.mean(labels))[None, :]
    else:
        y = labels - np.mean(labels, axis=1)

    # calculate transformed data (t scores)
    rank = np.linalg.matrix_rank(X)
    t_matrix = np.zeros((rank, X.shape[1]))
    for i in range(rank):
        # eigenvector corresponding to highest eigenvalue of R_xy is first column of U
        U, sigma, Vh = np.linalg.svd(X @ y.T, full_matrices=False)
        w_x = U[:, [0]]

        # calculate and save t score
        t = w_x.T @ X
        t_matrix[i, :] = t

        # calculate deflated X
        X = X - X @ t.T @ t / (t @ t.T)

        # calculate deflated y (only necessary if labels has more than 1 dimension)
        if y.shape[0] > 1:
            y = y - y @ t.T @ t / (t @ t.T)

    return t_matrix


def cut_data(data, sampling_frequency, window_length, window_shift=None):
    """
    Cut the data into segments of length window_length, overlap is possible by assigning a value to window_shift,
    and write them in a training matrix.

    :param data: dataset to use, dimensions: timepoints x samples (subjects)
    :param sampling_frequency: sampling frequency of the dataset
    :param window_length: length of each window to calculate the correlations (in minutes)
    :param window_shift: minutes that window is shifted, if None: shifted such that no data overlaps
    :return data_matrix: data matrix with cutted and concatenated data, from last to first window, window_length x #windows*#patients
    """

    # go backward
    last_value = data.shape[0]  # - 60 * sampling_frequency
    step = int(window_length * 60 * sampling_frequency)

    # calculate shift
    if window_shift is None:
        shift = step
    else:
        shift = window_shift * 60 * sampling_frequency

    data_list = []
    labels=[]
    for idx in range((last_value - step) // shift + 1):
        # get indices of time_windows
        stop_idx = last_value - idx * shift
        start_idx = stop_idx - step
        data_list.append(data[start_idx:stop_idx, :])
        labels.append(start_idx/(60*sampling_frequency))

    data_array = np.asarray(data_list)
    data_matrix = np.reshape(np.swapaxes(data_array, 0, 1),
                             (data_array.shape[1], data_array.shape[0] * data_array.shape[2]))

    return data_matrix, labels


def create_training_and_test_data(preictal, interictal, sampling_frequency, idx=0, window_length=15,
                                  test_window_shift=None, sampling=None):
    """
    Create training and test data matrices. Test matrix containinig of sample 'idx', training of all others.
    Concatenate data of windows with length window_length.

    :param preictal: dict with datasets to use for preictal data
    :param interictal: dict with datasets to use for preictal data
    :param sampling_frequency: dict with sampling requency for each dataset
    :param idx: index of sample that should be used for testing
    :param window_length: length of each window to calculate the correlations (in minutes)
    :param test_window_shift: minutes that window is shifted, if None: shifted such that no data overlaps
    :param sampling: None, 'up' (copies of minority class) or 'down' (not all samples of majority class)
    :return training_data (dict, each entry with shape window_length x samples), test_data, training_labels, test_labels: as the name says
    """

    # generate training and test data
    number_patients = preictal[list(preictal.keys())[0]].shape[1]

    # get idx of training data for all combinations
    train_idx = np.delete(np.arange(number_patients), idx, axis=0)
    test_idx = [idx]
    training_preictal = {}
    training_interictal = {}
    training_mean = {}
    test_preictal = {}
    test_interictal = {}

    for modality in sorted(preictal.keys()):
        # preictal
        training_preictal[modality] = preictal[modality][:, train_idx]
        training_interictal[modality] = interictal[modality][:, train_idx]
        test_preictal[modality] = preictal[modality][:, test_idx]
        test_interictal[modality] = interictal[modality][:, test_idx]

    # concatenate data of preictal and interictal phase
    training_data = {}
    test_data = {}
    for modality in sorted(preictal.keys()):
        # make data zero mean across training subjects
        pre_training,_ = cut_data(training_preictal[modality], sampling_frequency[modality], window_length=window_length)
        inter_training,_ = cut_data(training_interictal[modality], sampling_frequency[modality],
                                  window_length=window_length)

        if sampling == 'up':
            pre_training_sampled = resample(pre_training.T,
                                            replace=True,  # sample with replacement
                                            n_samples=inter_training.shape[1],  # to match majority class
                                            random_state=123).T  # reproducible results
            inter_training_sampled = inter_training

        elif sampling == 'down':
            pre_training_sampled = pre_training
            # Downsample majority class
            inter_training_sampled = resample(inter_training.T,
                                              replace=False,  # sample without replacement
                                              n_samples=pre_training.shape[1],  # to match minority class
                                              random_state=123).T  # reproducible results

        elif sampling is None:
            pre_training_sampled = pre_training
            inter_training_sampled = inter_training

        else:
            raise Exception('sampling must be up, down or None')

        training_data[modality] = np.hstack([pre_training_sampled, inter_training_sampled])
        training_mean[modality] = np.mean(training_data[modality], axis=1)[:, None]
        training_data[modality] = training_data[modality] - training_mean[modality]

        pre_test,_ = cut_data(test_preictal[modality], sampling_frequency[modality], window_length=window_length,
                            window_shift=test_window_shift)
        inter_test,_ = cut_data(test_interictal[modality], sampling_frequency[modality], window_length=window_length,
                              window_shift=test_window_shift)

        #  subtract mean from test subjects
        test_data[modality] = np.hstack([pre_test, inter_test])
        test_data[modality] = test_data[modality] - training_mean[modality]

    # set preictal (1) and interictal (0) labels
    training_labels = np.hstack([np.ones(pre_training_sampled.shape[1]), np.zeros(inter_training_sampled.shape[1])])
    test_labels = np.hstack([np.ones(pre_test.shape[1]), np.zeros(inter_test.shape[1])])

    return training_data, test_data, training_labels, test_labels


def train_classifier(D, labels, classifier='svm', optimize=False):
    """
    Train a classifier.

    :param D:
    :param labels:
    :param classifier: 'svm', 'knn' or 'rf' ('rf' is very slow)
    :param optimize: if the classifier should be optimized or default values should be used
    :return:
    """

    if classifier == 'svm':
        if optimize:
            # Set the parameters by cross-validation
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100, 1000], 'class_weight': ['balanced']},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
                                'class_weight': ['balanced']}]
            # score function
            score = make_scorer(fbeta_score, beta=1)
            # grid search to find best parameters
            clf = GridSearchCV(SVC(), tuned_parameters, cv=3, iid=False)#, scoring=score)
        else:
            clf = SVC(kernel='rbf', gamma='auto', class_weight='balanced')

    elif classifier == 'knn':
        if optimize:
            # Set the parameters by cross-validation
            tuned_parameters = {'n_neighbors': [3, 5, 11, 19, np.sqrt(D.shape[0]).astype(int)],
                                'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
            # score function
            score = make_scorer(fbeta_score, beta=1)
            # grid search to find best parameters
            clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=3, iid=False)#, scoring=score)
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors=np.sqrt(D.shape[0]).astype(int))

    elif classifier == 'rf':
        if optimize:
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]  # Create the random grid
            random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                           'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            clf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=random_grid, n_iter=100,
                                     cv=3, random_state=42, n_jobs=-1, iid=False)
        else:
            clf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=10, min_samples_split=5,
                                         min_samples_leaf=2, bootstrap=False)
    else:
        raise Exception("Classifier not known. Choose 'svm', 'knn' or 'rf' as classifier")

    clf.fit(D, labels)

    return clf


if __name__ == '__main__':

    x = np.random.randn(100) * 3 + 2
    x[5:20] = np.nan
    plt.plot(x[~np.isnan(x)])
    plt.plot(running_mean(x))
    # plt.show()

    # load data
    preictal = spio.loadmat('../Epilepsy_data/100118_FinalData.mat')
    interictal = spio.loadmat('../Epilepsy_data/All_MATS_interictal_zscoredEDA_45min.mat')

    # keep only recordings
    preictal = {i[4:-3]: preictal[i] for i in preictal if "BS" in i}
    interictal = {i[4:-9]: interictal[i] for i in interictal if "__" not in i}

    preictal = {key: preictal[key] for key in sorted(preictal.keys())}
    interictal = {key: interictal[key] for key in sorted(interictal.keys())}

    # sampling frequency
    sampling_frequency = {'RR': 1, 'HR': 1, 'Temp': 4, 'EDA': 4}

    # delete bad data in patients
    sub_del = np.array([1, 6, 11, 12])
    for modality in preictal.keys():
        preictal[modality] = np.delete(preictal[modality], sub_del, 1)

    create_training_and_test_data(preictal, interictal, sampling_frequency)
