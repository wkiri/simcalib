# Similarity-based Calibration methods
# Also includes Platt scaling and temperature scaling
# for comparison.
#
# Kiri Wagstaff

import sys
import numpy as np
import scipy
from progressbar import ProgressBar, Bar, ETA
from progressbar import Counter as pCounter
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, \
    euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler


# Find optimal A, B with NLL loss function
# for Platt scaling applied to model scores
def nll_platt_fn(param, *args):
    A = param[0]
    B = param[1]
    scores, labels = args
    # use A, B to convert scores to probs
    probs = 1 / (1 + np.exp(A * scores + B))
    # NLL (per item); make it negative so optimizer
    # goes in the right direction
    nll = -np.sum(labels * np.log(probs) +
                  (1 - labels) * np.log(1 - probs)) / len(probs)

    return nll


# Scale test probabilities (of class 1) using
# labeled data and their probs from the calibration set.
def platt_scaling(probs, labels, test_probs):

    if len(probs.shape) > 1 and probs.shape[1] != 2:
        print('Error: Platt only works for binary problems.')
        sys.exit(1)

    if probs.shape[1] == 2:
        # Convert to single class prob (of class 1)
        probs = probs[:, 1]

    # Convert labels from 0/1 using Platt's recipe
    pos_labels = labels == 1
    neg_labels = labels == 0
    n_pos = np.sum(pos_labels)
    n_neg = np.sum(neg_labels)
    platt_labels = np.zeros((len(labels)))
    platt_labels[pos_labels] = (n_pos + 1) / (n_pos + 2)
    platt_labels[neg_labels] = 1.0 / (n_neg + 2)

    res = scipy.optimize.minimize(nll_platt_fn,
                                  #[0.0, np.log((n_neg + 1)/(n_pos + 1))],
                                  [1.0, np.log((n_neg + 1)/(n_pos + 1))],
                                  args=(probs, platt_labels),
                                  method='BFGS', tol=1e-12)
    A, B = res.x[0], res.x[1]

    test_probs_class1 = 1 / (1 + np.exp(A * test_probs[:, 1] + B))
    new_test_probs = np.stack(((1 - test_probs_class1),
                               test_probs_class1),
                              axis=1)

    return new_test_probs


# To optimize NLL for temperature scaling (Zhang et al., 2020)
# https://github.com/zhang64-llnl/Mix-n-Match-Calibration
def nll_fn(t, *args):
    # find optimal temperature with NLL loss function
    logit, label = args
    # adjust logits by T
    logit = logit / t
    # convert logits to probabilities
    n = np.sum(np.exp(logit), 1)
    probs = np.exp(logit) / n[:, None]
    # avoid values too close to 0 or 1
    eps = 1e-20
    probs = np.clip(probs, eps, 1 - eps)
    # NLL
    nll = -np.sum(label * np.log(probs)) / probs.shape[0]

    return nll


# To optimize MSE for temperature scaling (Zhang et al., 2020)
# https://github.com/zhang64-llnl/Mix-n-Match-Calibration
def mse_fn(t, *args):
    ## find optimal temperature with MSE loss function
    logit, label = args
    # adjust logits by T
    logit = logit / t
    # convert logits to probabilities
    n = np.sum(np.exp(logit), 1)
    probs = np.exp(logit) / n[:, None]
    # MSE
    mse = np.mean((probs - label) ** 2)

    return mse


# Use temperature scaling to modify logits, given labels.
# This is a good entry point if using neural networks.
# If test_logits is given, return calibrated probabilities.
# Based on:
# https://github.com/zhang64-llnl/Mix-n-Match-Calibration
def temp_scaling(logits, labels, n_classes,
                 test_logits=np.array(()), optim='mse'):

    y = np.eye(n_classes)[labels] # one-hot encoding
    if optim == 'mse':
        opt_fn = mse_fn
    elif optim == 'nll':
        opt_fn = nll_fn
    else:
        print('Error: unknown optimization method %s' % optim)
        sys.exit(1)
    t = scipy.optimize.minimize(opt_fn, 1.0, args=(logits, y),
                                method='L-BFGS-B', bounds=((0.05, 5.0),),
                                tol=1e-12)
    t = t.x

    # If provided, generate calibrated probs for the test set
    if len(test_logits) > 0:
        test_logits = test_logits / t
        new_test_probs = np.exp(test_logits) / \
            np.sum(np.exp(test_logits), 1)[:, None]
        return t, new_test_probs

    return t


# Use temperature scaling to modify probabilities, given labels.
# This is a good entry point if you have probabilities
# but not logits.
# If test_probs is given, return its calibrated version too.
def temp_scaling_probs(probs, labels, n_classes, test_probs=np.array(()),
                       optim='mse'):

    eps = 1e-7
    ts_probs = np.clip(probs, eps, 1 - eps)
    ts_logits = np.log(ts_probs) - np.log(1 - ts_probs)

    # If provided, generate calibrated probs for the test set
    if len(test_probs) > 0:
        test_probs = np.clip(test_probs, eps, 1 - eps)
        test_logits = np.log(test_probs) - np.log(1 - test_probs)
    else:
        test_logits = np.array(())

    return temp_scaling(ts_logits, labels, n_classes,
                        test_logits, optim=optim)


# Compute similarity matrix (or prep for it) between X_test and X_cal
# using y_cal if needed (for sim_method='RFprox').
#
# Options:
# - sim_method options:
#   'RFprox': RF proximity
#   'Isoprox': Isolation forest proximity
#   'cosine': cosine sim. (kernel)
#   'rbf': RBF kernel
#   'sim_euclid': Euclidean distance (for sim)
#   (for testing purposes:)
#   'all_one': all pairs of items have similarity 1
#   '<method>-1NN': set nearest neighor to have sim. 1, else 0
# - sim: pre-computed similarity matrix (optional, used only by RFprox/Isoprox)
#
# Returns n x m array where n = |X_test| and m = |X_cal|,
# or leaf assignments if sim_method == 'RFprox' or 'Isoprox'.
def calc_sim(X_test, X_cal, y_cal=None, sim_method='sim_euclid', sim=None):
    """
    >>> np.random.seed(0)
    >>> X_cal = np.random.rand(5, 3)
    >>> X_test = np.random.rand(2, 3)
    >>> calc_sim(X_test, X_cal)
    Euclid. sim: min -1.277106, max -0.341592
    array([[-0.86543108, -0.63809564, -0.94847268, -0.88049546, -1.27710576],
           [-0.46672837, -0.60361979, -0.34159159, -0.6034734 , -0.93324293]])

    >>> calc_sim(X_test, X_cal, sim_method='sim_euclid-1NN')
    Euclid. sim: min -1.277106, max -0.341592
    array([[0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.]])

    >>> calc_sim(X_test, X_cal, sim_method='cosine')
    Cosine sim: min 0.569869, max 0.996167
    array([[0.81061331, 0.87508384, 0.87038602, 0.78490629, 0.56986892],
           [0.99616697, 0.99472739, 0.98846092, 0.98122596, 0.89717548]])

    >>> calc_sim(X_test, X_cal, sim_method='cosine-1NN')
    Cosine sim: min 0.569869, max 0.996167
    array([[0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.]])

    >>> calc_sim(X_test, X_cal, sim_method='rbf')
    RBF sim: min 0.000000, max 0.016007
    array([[7.83897810e-09, 2.95034542e-07, 1.51550553e-07, 5.60989834e-06,
            8.96936190e-12],
           [1.60068127e-02, 2.61365087e-03, 6.65707955e-04, 2.25654935e-05,
            2.14475312e-03]])

    >>> calc_sim(X_test, X_cal, sim_method='rbf-1NN')
    RBF sim: min 0.000000, max 0.016007
    array([[0., 0., 0., 1., 0.],
           [1., 0., 0., 0., 0.]])

    >>> calc_sim(X_test, X_cal, y_cal=[0, 0, 1, 1, 1], sim_method='RFprox')
    RFprox sim: min 0.090000, max 0.680000
    array([[0.44, 0.58, 0.58, 0.47, 0.09],
           [0.55, 0.39, 0.67, 0.4 , 0.68]])

    >>> calc_sim(X_test, X_cal, y_cal=[0, 0, 1, 1, 1], sim_method='RFprox-1NN')
    RFprox sim: min 0.090000, max 0.680000
    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.]])

    >>> calc_sim(X_test, X_cal, sim_method='Isoprox')
    Isoprox sim: min 0.000000, max 0.470000
    array([[0.02, 0.34, 0.27, 0.39, 0.  ],
           [0.34, 0.11, 0.47, 0.01, 0.24]])

    >>> calc_sim(X_test, X_cal, sim_method='Isoprox-1NN')
    Isoprox sim: min 0.000000, max 0.470000
    array([[0., 0., 0., 1., 0.],
           [0., 0., 1., 0., 0.]])

    >>> calc_sim(X_test, X_cal, sim_method='all_one')
    All pairwise similarities = 1.0
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])
    """

    if sim_method.startswith('RFprox'):
        if sim is None:
            # Train an RF for similarity estimation
            prox_clf = RandomForestClassifier(n_estimators=100,
                                              #ccp_alpha=0.001,
                                              random_state=0)
            prox_clf.fit(X_cal, y_cal)
            # Apply it to the calibration set
            cal_leaf_id = prox_clf.apply(X_cal).T  # leaf assignments
            test_leaf_id = prox_clf.apply(X_test).T  # leaf assignments
            # Convert to similarity
            n_test, n_cal = len(X_test), len(X_cal)
            sim = np.zeros((n_test, n_cal))
            pbar_args = ['RFprox: ', pCounter(), '/%d' % n_test, Bar('='), ETA()]
            progress = ProgressBar(widgets=pbar_args)
            for i in progress(range(n_test)):
                for j in range(n_cal):
                    sim[i, j] = np.mean(np.equal(cal_leaf_id[:, j],
                                                 test_leaf_id[:, i]))
        print('RFprox sim: min %f, max %f' % (np.min(sim), np.max(sim)))

    elif sim_method.startswith('Isoprox'):
        if sim is None:
            # Train an isolation forest for similarity estimation
            iso_clf = IsolationForest(n_estimators=100,
                                       random_state=0)
            iso_clf.fit(X_cal)
            # Apply it to the calibration and test sets
            cal_leaf_id, test_leaf_id = [], []
            for t in iso_clf.estimators_:
                cal_leaf_id += [t.apply(X_cal).T]  # leaf assignments
                test_leaf_id += [t.apply(X_test).T]  # leaf assignments
            cal_leaf_id = np.array(cal_leaf_id)
            test_leaf_id = np.array(test_leaf_id)
            # Convert to similarity
            n_test, n_cal = len(X_test), len(X_cal)
            sim = np.zeros((n_test, n_cal))
            pbar_args = ['Isoprox: ', pCounter(), '/%d' % n_test, Bar('='), ETA()]
            progress = ProgressBar(widgets=pbar_args)
            for i in progress(range(n_test)):
                for j in range(n_cal):
                    if np.sum(cal_leaf_id[:, j] != -1) > 0:
                        sim[i, j] = (np.sum(np.equal(cal_leaf_id[:, j],
                                                     test_leaf_id[:, i])) /
                                     np.sum(cal_leaf_id[:, j] != -1))
        print('Isoprox sim: min %f, max %f' % (np.min(sim), np.max(sim)))

    elif sim_method.startswith('cosine'):
        # Normalize to range 0,1 instead of -1,1
        sim = (cosine_similarity(X_test, X_cal) + 1) / 2.0
        print('Cosine sim: min %f, max %f' % (np.min(sim), np.max(sim)))

    elif sim_method.startswith('sim_euclid'):
        # Inverted Euclidean distance (to yield similarity)
        dist = euclidean_distances(X_test, X_cal)
        sim = -dist
        print('Euclid. sim: min %f, max %f' % (np.min(sim), np.max(sim)))

    elif sim_method.startswith('rbf'):
        # Normalize data first, using cal data as reference
        scaler = StandardScaler()
        scaler.fit(X_cal)
        X_cal = scaler.transform(X_cal)
        X_test = scaler.transform(X_test)
        sim = rbf_kernel(X_test, X_cal) #, gamma=0.05)
        print('RBF sim: min %f, max %f' % (np.min(sim), np.max(sim)))

    elif sim_method == 'all_one':
        sim = np.ones((len(X_test), len(X_cal)))
        print('All pairwise similarities = 1.0')

    else:
        raise ValueError('Unknown similarity measure %s' % sim_method)

    if sim_method.endswith('-1NN'):
        # Set nearest neighbor to sim 1, everything else to 0
        nn1 = np.argmax(sim, axis=1)
        for i in range(len(X_test)):
            sim[i, :] = 0
            sim[i, nn1[i]] = 1

    return sim


# Calibrate using neighbors in X_cal for each X_test.
#
# Options:
# - sim_method options:
#   'RFprox': RF proximity
#   'Isoprox': Isolation forest proximity
#   'cosine': cosine sim. (kernel)
#   'rbf': RBF kernel
#   'sim_euclid': Euclidean distance (for sim)
#   (for testing purposes:)
#   'all_one': all pairs of items have similarity 1
#   '<method>-1NN': set nearest neighor to have sim. 1, else 0
# - nn: number of nearest neighbors to use (-1 = all)
# - weighted: combine neighbor votes with similarity weights;
#   else just compute the average
# - hh: if not None, use to set a similarity threshold for neighbors
# - seed: random seed
# - verbose: 0 for nothing, 1 for progress bars, 2 for per-item reporting
#
# Return calibrated test probs.
def calib_sim(X_cal, y_cal, X_test, test_probs,
              sim_method='cosine', nn=-1, weighted=True, hh=None, verbose=1):

    (n_test, n_classes) = test_probs.shape
    calib_test_probs = np.zeros_like(test_probs)
    n_cal = len(X_cal)

    if len(X_test.shape) == 1:
        X_test = X_test.reshape(1, -1)
    if nn == -1:
        nn = len(X_cal)

    # Compute the similarity matrix
    sim = calc_sim(X_test, X_cal, y_cal, sim_method, sim=None)

    progress = ProgressBar(widgets=['Calib-sim: ', pCounter(), '/%d' % n_test,
                                    Bar('='), ETA()])
    sim_mass = np.zeros((n_test))
    for i in progress(range(n_test)):
        if sim_method in ['RFprox', 'Isoprox'] and isinstance(sim, tuple):
            (cal_leaf_id, test_leaf_id) = sim
            sim_i = np.zeros((n_cal))
            for j in range(n_cal):
                if np.sum(cal_leaf_id[:, j] != -1) > 0:
                    sim_i[j] = (np.sum(np.equal(cal_leaf_id[:, j],
                                                test_leaf_id[:, i])) /
                                np.sum(cal_leaf_id[:, j] != -1))
        else:
            sim_i = sim[i]

        # Get the nearest neighbors (top NN most similiar)
        nns = np.argsort(sim_i)[-nn:]

        # If HH provided, limit neighbors to those with
        # similarity greater than HH
        if hh is not None:
            # Keep only those with sim >= hh / 2.0
            nnind = np.where(sim_i[nns] >= (hh[i] / 2.0))[0]
            if len(nnind) == 0:
                # Keep at least one (closest) item
                nns = np.array([np.argmax(sim_i)])
            else:
                nns = nns[nnind]

        sim_mass[i] = np.sum(sim_i[nns])

        # If it has no similarity mass (!), leave it as-is
        if sim_mass[i] == 0:
            calib_test_probs[i] = test_probs[i]
            if verbose > 1:
                print('%d) Neighbor similarity mass = 0; set q-hat to p-hat' % i)
            continue

        # Calibrate using similarity-weighted nearest neighbors
        if verbose > 1:
            print('%d) P-hat: ' % i + ','.join(['%.2f' % p
                                                for p in test_probs[i]]))

        for c in range(n_classes):
            if weighted:
                calib_test_probs[i, c] = np.sum((y_cal[nns] == c) * sim_i[nns])
            else:
                calib_test_probs[i, c] = np.mean(y_cal[nns] == c)

        if weighted:
            if verbose > 1:
                print(' Q-hat: ' + ','.join(['%.2f' % p for p in calib_test_probs[i]]))
                print(' norm by %.2f' % np.sum(calib_test_probs[i]))
            # Normalize to make it a probability
            calib_test_probs[i] = calib_test_probs[i] / np.sum(calib_test_probs[i])

        if verbose > 1:
            print(' Q-hat: ' + ','.join(['%.2f' % p
                                         for p in calib_test_probs[i]]))
            input()

    return calib_test_probs, sim_mass


# Calculate the Brier score (MSE) of the predicted probabilities
# vs. true labels.  We need to know all classes for this to be possible.
# Note: multiclass results (if probs array is 2D) will, for binary problems,
# be 2X what you get if using a single prob-of-class-1 1D array.
def brier(probs, labels, classes):
    """Calculate Brier score for a set of predictions and labels.
    # Binary (one prob per item)
    >>> brier(np.array([0.0, 0.0, 1.0, 1.0]), ['0', '0', '1', '1'], ['0', '1'])
    0.0

    >>> brier(np.array([0.5, 1.0]), ['0', '1'], ['0', '1'])
    0.125

    # Multiclass
    >>> brier(np.array([[1.0, 0.0], [0.0, 1.0]]), ['0', '1'], ['0', '1'])
    0.0

    >>> brier(np.array([[0.5, 0.5], [0.0, 1.0]]), ['0', '1'], ['0', '1'])
    0.25

    # 3-class
    >>> brier(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ['a', 'b'], \
                       ['a', 'b', 'c'])
    0.0

    >>> brier(np.array([[0.5, 0.25, 0.25], [0.0, 1.0, 0.0]]), ['a', 'b'], \
                       ['a', 'b', 'c'])
    0.1875

    >>> brier(np.array([[0, 1, 0]]), ['b'], ['a', 'b', 'c'])
    0.0

    >>> brier(np.array([[0, 1, 0], [1, 0, 0]]), ['b', 'b'], ['a', 'b', 'c'])
    1.0
    """
    ce = 0
    if len(probs.shape) == 1: # binary problem with 1D probs
        # Calculate MSE
        for (p, l) in zip(probs, labels):
            ce += (1 - p) ** 2 if l == classes[1] else p ** 2
    else: # multiclass (could be two-class)
        mismatch = probs ** 2  # penalty for wrong class
        if isinstance(classes, list):
            class_ind = [classes.index(c) for c in labels]
        else:
            if isinstance(labels[0], int):
                class_ind = labels
            else:
                class_ind = [np.where(classes == c)[0][0] for c in labels]
        # Overwrite the ones for the correct class
        row_inds = np.arange(len(labels))
        corr_probs = probs[row_inds, class_ind]
        mismatch[row_inds, class_ind] = np.power(1 - corr_probs, 2)
        ce = np.sum(mismatch)
        '''
        # Classes must line up with prob columns for this to make sense
        for (p, l) in zip(probs.tolist(), labels):
            for c in range(len(classes)):
                ce += (1 - p[c]) ** 2 if l == classes[c] else p[c] ** 2
        '''

    # Normalize by number of items
    ce = float(ce) / len(labels)

    return ce


# Compute hidden heterogeneity (HH) for each test prediction (test_probs)
# using a probability neighborhood over the calibration set, of radius r.
# HH is the difference in log-loss between the predicted probs
# and what could be achieved by a local RF (with optimized pruning).
# Returns an array of size n x 1, given test_probs size n x c, and
# a nested list of neighbor indices per test item.
# - if r is Inf, don't use it to filter
# - verbose: 0 for nothing, 1 for progress bars, 2 for per-item reporting
def hidden_hetero(X_cal, y_cal, cal_probs, test_probs, r=0.1,
                  seed=0, verbose=1):
    """
    >>> np.random.seed(0)
    >>> X_cal = np.random.rand(10, 3)
    >>> test_probs = np.random.rand(5, 2)
    >>> test_probs[:, 0] = 1 - test_probs[:, 1]
    >>> cal_probs = np.random.rand(10, 2)
    >>> cal_probs[:, 0] = 1 - cal_probs[:, 1]
    >>> y_cal = np.random.randint(2, size=10)

    >>> hidden_hetero(X_cal, y_cal, cal_probs, test_probs, r=-3)
    Traceback (most recent call last):
     ...
    ValueError: Error: r (-3.000000) must be >= 0

    >>> #test_probs, cal_probs
    >>> hh, nbrs, _ = hidden_hetero(X_cal, y_cal, cal_probs, test_probs, r=0.5, verbose=2)
    0) HH p-hat 0.23,0.77, ccp_alpha 0.0, OOB score 0.88, Brier 0.903 -> 0.036
    1) HH p-hat 0.43,0.57, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    2) HH p-hat 0.38,0.62, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    3) HH p-hat 0.38,0.62, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    4) HH p-hat 0.32,0.68, ccp_alpha 0.0, OOB score 0.89, Brier 0.982 -> 0.035

    >>> hh
    array([0.86757124, 1.04085693, 1.04085693, 1.04085693, 0.94684633])
    >>> nbrs
    [array([0, 2, 3, 4, 5, 7, 8, 9]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([0, 2, 3, 4, 5, 6, 7, 8, 9])]

    >>> hh = hidden_hetero(X_cal, y_cal, cal_probs, test_probs[1].reshape(1, -1), r=0.5, verbose=2)
    0) HH p-hat 0.43,0.57, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020

    >>> hh = hidden_hetero(X_cal, y_cal, cal_probs, test_probs, r=np.inf, verbose=2)
    0) HH p-hat 0.23,0.77, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    1) HH p-hat 0.43,0.57, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    2) HH p-hat 0.38,0.62, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    3) HH p-hat 0.38,0.62, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020
    4) HH p-hat 0.32,0.68, ccp_alpha 0.0, OOB score 0.90, Brier 1.061 -> 0.020

    >>> cal_probs = np.random.rand(10, 3)
    >>> hh = hidden_hetero(X_cal, y_cal, cal_probs, test_probs, r=0.5)
    Traceback (most recent call last):
     ...
    ValueError: Test (2) and calibration (3) probs must have same number of classes (columns).
    """

    # Check arguments
    if r < 0:
        raise ValueError('Error: r (%f) must be >= 0' % r)
    if test_probs.shape[1] != cal_probs.shape[1]:
        raise ValueError('Test (%d) and calibration (%d) probs must have'
                         ' same number of classes (columns).' %
                         (test_probs.shape[1], cal_probs.shape[1]))
    if X_cal.shape[0] != cal_probs.shape[0]:
        raise ValueError('Calibration X (%d) and probs (%d) do not have'
                         ' the same number of items.' %
                         (X_cal.shape[0], cal_probs.shape[0]))
    if X_cal.shape[0] != y_cal.shape[0]:
        raise ValueError('Calibration X (%d) and labels y (%d) do not have'
                         ' the same number of items.' %
                         (X_cal.shape[0], y_cal.shape[0]))

    n_test = len(test_probs)
    classes = np.unique(y_cal)
    # Pruning CCP_alphas to search over
    #ccp_alphas = [0]
    ccp_alphas = np.linspace(0, 0.03, 7)

    # Get neighbors within radius r in probability simplex
    # Following Krstovski et al. (2013), we take the element-wise
    # square root of each probability distribution,
    # then use regular Euclidean distance to find neighbors
    sq_cal_probs = np.sqrt(cal_probs)
    sq_test_probs = np.sqrt(test_probs)
    if not np.isinf(r):
        nn = NearestNeighbors(radius=r * np.sqrt(2))
        nn.fit(sq_cal_probs)

    hh = {} # cache HH results to avoid recalculating them
    hh_per_item = []
    nbrs = [] # nested list of neighbors, per test item
    rfs = [] # list of random forest classifiers, per test item
    if verbose > 0:
        progress = ProgressBar(widgets=['Items (HH): ',
                                        pCounter(), '/%d' % n_test,
                                        Bar('='), ETA()])
    else:
        progress = list
    for p in progress(range(n_test)):
        sqrt_pr = sq_test_probs[p]
        pr = test_probs[p]
        # Only compute HH for unique probabilities once
        # Use probabilities to 3 decimal places for caching
        pr_simple = (pr * 1000).astype(int) / 1000.0
        if tuple(pr_simple) in hh:
            hh_data = hh[tuple(pr_simple)]
            hh_per_item += [hh_data[0]]
            nbrs += [hh_data[1]]
            rfs += [hh_data[2]]
            continue

        if np.isinf(r):
            nbrs += [range(len(y_cal))]
        else:
            # Hellinger distance is 1/sqrt(2) * Euclidean distance
            # of sqrt of prob vectors, so convert radius
            # Index 0 because we only have one item and this gets its neighbors.
            nbrs += [nn.radius_neighbors(sqrt_pr.reshape(1, -1),
                                         radius=r * np.sqrt(2),
                                         return_distance=False)[0]]

        # if no neighbors, or only one class present, set HH = 0
        if len(nbrs[p]) == 0 or len(np.unique(y_cal[nbrs[p]])) == 1:
            hh_per_item += [0]
            hh[tuple(pr_simple)] = (hh_per_item[p], nbrs[p], None)
            rfs += [None]
            continue

        # Compute Brier score of original predictions
        bs_orig = brier(cal_probs[nbrs[p]], y_cal[nbrs[p]], classes)
        bs_new = np.inf

        # Search over CCP_alpha pruning values using OOB score to select best
        best_score = 0
        rfs += [None]
        for ccp_alpha in ccp_alphas:
            # Train a bagged ensemble (use all features per split)
            hh_clf = RandomForestClassifier(n_estimators=100,
                                            max_features=None,
                                            oob_score=True,
                                            #min_samples_leaf=10,
                                            ccp_alpha=ccp_alpha,
                                            random_state=seed)
            hh_clf.fit(X_cal[nbrs[p]], y_cal[nbrs[p]])
            # Ensure some model is saved
            if rfs[-1] is None:
                rfs[p] = hh_clf
            s = hh_clf.oob_score_
            if s > best_score:
                best_score = s
                rfs[p] = hh_clf
                # Compute HH as difference in Brier score
                y_pred_probs = np.array(hh_clf.predict_proba(X_cal[nbrs[p]]))
                # If not all classes were present, flesh it out
                for k, c in enumerate(classes):
                    if c not in hh_clf.classes_:
                        if k == 0:
                            y_pred_probs = np.hstack((np.zeros((len(nbrs[p]),1)),
                                                      y_pred_probs[:,k:]))
                        else:
                            y_pred_probs = np.hstack((y_pred_probs[:,:k],
                                                      np.zeros((len(nbrs[p]),1)),
                                                      y_pred_probs[:,k:]))
                bs_new = brier(y_pred_probs, y_cal[nbrs[p]], classes)
                if verbose > 1:
                    print('%d) HH p-hat %s, ccp_alpha %s,'
                          ' OOB score %.2f, Brier %.3f -> %.3f' %
                          (p, ','.join(['%.2f' % p for p in pr]),
                           ccp_alpha, s, bs_orig, bs_new))

        if best_score == 0 and verbose > 0:
            print('%d) Warning: multiple classes,' % p,
                  'but no ccp_alpha OOB score > 0.')

        # Compute difference (clip to 0)
        hh_per_item += [max(bs_orig - bs_new, 0)]
        hh[tuple(pr_simple)] = (hh_per_item[p], nbrs[p], rfs[p])

    # Ensure we got an HH value for every test item
    assert len(hh_per_item) == n_test
    return np.array(hh_per_item), nbrs, rfs
