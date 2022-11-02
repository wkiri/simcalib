# Utility functions to get classifiers and data
# and compute Brier score and ECE values.
#
# Kiri Wagstaff

import sys
import os
import csv
import gzip
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_moons, make_blobs


# Create a classifier of the specified type, usually with default parameters.
def get_classifier(clf_type):

    filestem = clf_type # by default

    if clf_type == 'DT':
        # Decision tree; default min_samples_leaf=1
        #clf = DecisionTreeClassifier(random_state=0)
        clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
        #clf = DecisionTreeClassifier(random_state=0, max_depth=2)
        filestem = 'DT-minsample10'
    elif clf_type == 'MC': # Majority Class
        # Have to specify alpha=1 to instate pruning; otherwise
        # max_depth=1 still has 1 split (3 nodes)
        clf = DecisionTreeClassifier(random_state=0, max_depth=1,
                                     ccp_alpha=1)
    elif clf_type == 'SVM':
        # SVM with the ability to output posteriors; default C=1.0
        clf = SVC(kernel='linear', probability=True, random_state=0)
    elif clf_type == 'RBFSVM':
        # SVM with Gaussian kernel; default C=1.0, gamma='scale'
        clf = SVC(kernel='rbf', probability=True, random_state=0)
    elif clf_type == 'RF':
        # RF; default n_estimators=100, min_samples_split=2
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        #clf = RandomForestClassifier(n_estimators=10, random_state=0)
    elif clf_type == 'GBT':
        # Gradient-boosted trees; default learning_rate=0.1,
        # n_estimators=100, subsample=1.0 (no sampling),
        # criterion='friedman_mse', min_samples_split=2, max_depth=3 (!),
        # loss='deviance' (LR) but can try 'exponential' for AdaBoost
        clf = GradientBoostingClassifier(n_estimators=200, random_state=0)
    elif clf_type == 'NB':
        # Naive Bayes
        clf = GaussianNB()
    else:
        print('Unknown classifier type %s' % clf_type)
        sys.exit(1)

    return (clf, filestem)


def parse_tom_credit(infile):
    X = []
    y = []
    y_prob = []

    hdr = []
    if not os.path.exists(infile):
        print('Data file %s not found.' % infile)
        sys.exit(1)

    with open(infile, 'r') as inf:
        rdr = csv.reader(inf)
        # first two columns both capture an id, so skip them
        hdr = next(rdr)[2:-2]
        print('Features:', hdr)
        for vals in rdr:
            X += [[float(v) for v in vals[2:-2]]]
            y += [int(vals[-2])]
            # Tom provided the prob of the predicted class;
            # flesh this out to a distribution
            prob = float(vals[-1])
            y_prob += [[1 - prob, prob]]

    return (np.array(X), np.array(y), np.array(y_prob), hdr)


# Load the specified data set and return n_samples
# For pre-trained models, pass in the model_name as well.
def get_dataset(dataset, n_samples, model_name='', seed=0, verbose=True):

    np.random.seed(seed)

    if dataset == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=seed)
        ids = range(len(y))
    elif dataset == 'mixed':
        X, y = make_blobs(n_samples=n_samples, centers=[[5, 5], [5, 5]],
                          random_state=seed)
        ids = range(len(y))
    elif dataset == 'sep':
        X, y = make_blobs(n_samples=n_samples, centers=[[1, 0], [8, 7]],
                          random_state=seed)
        ids = range(len(y))
    elif dataset == '2gaussians':
        X, y = make_blobs(n_samples=n_samples, centers=[[1, 2], [5, 2]],
                          random_state=seed)
        ids = range(len(y))
    elif dataset == '2gaussians-sep':
        X, y = make_blobs(n_samples=n_samples, centers=[[1, 2], [8, 2]],
                          random_state=seed)
        ids = range(len(y))
    elif dataset == '3gaussians':
        X, y = make_blobs(n_samples=n_samples, centers=[[1, 2], [3, 2], [2, 1]],
                          cluster_std=0.5,
                          random_state=seed)
        ids = range(len(y))
    elif dataset == '3gaussians-sep':
        X, y = make_blobs(n_samples=n_samples, centers=[[1, 3], [4, 3], [2.5, 1]],
                          cluster_std=0.5,
                          random_state=seed)
        ids = range(len(y))
    elif dataset == '3gaussians-multimodal':
        X, y = make_blobs(n_samples=n_samples, centers=[[1, 3], [4, 3], [2.5, 1], [5.5, 1]],
                          cluster_std=0.5,
                          random_state=seed)
        # Make cluster 3 have the same label as cluster 0 so we have 3 classes
        y[y == 3] = 0
        ids = range(len(y))
    elif dataset.startswith('mnist'):
        mnist_file = 'data/uci/%s-%d-s%d.npz' % (dataset, n_samples, seed)
        if not os.path.exists(mnist_file):
            data_dir = os.path.dirname(mnist_file)
            if not os.path.isdir(data_dir):
                print('Creating data directory %s' % data_dir)
                os.makedirs(data_dir)
            print('Fetching MNIST data.')
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # Handle mnist-?v? type problems (2 digits specified)
            if dataset.startswith('mnist-') and dataset[-2] == 'v':
                c1 = dataset[-3]
                c2 = dataset[-1]
                use = (y == c1) | (y == c2)
            elif dataset == 'mnist3':
                use = (y == '4') | (y == '8') | (y == '9')
            elif dataset == 'mnist10':
                use = [True] * len(y)
            else:
                print('Unknown data set %s' % dataset)
                sys.exit(1)
            X = np.array(X[use])
            y = np.array(y[use])
            if n_samples > len(y):
                raise ValueError('Cannot get %d from %d examples.' %
                                 (n_samples, len(y)))
            ids = np.where(use)[0]
            if n_samples > -1:
                keep = np.random.choice(range(len(y)), n_samples,
                                        replace=False)
                X = X[keep,:]
                y = y[keep]
                ids = ids[keep]
            print('Using %d of %d samples' % (n_samples, sum(use)))
            np.savez(mnist_file, X=X, y=y, ids=ids)
        else:
            npz = np.load(mnist_file, allow_pickle=True)
            X = npz['X']
            y = npz['y']
            ids = npz['ids']
        classes = np.unique(y).tolist()
        # Remap classes to integers so Kumar et al. are happy
        y = np.array([classes.index(yval) for yval in y])
        if verbose:
            for yval in np.unique(y):
                print('  Class %s: %d' % (yval, sum(y == yval)))
    elif dataset == 'fashion-mnist':
        # From https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
        lblpath = 'data/fashion-mnist/train-labels-idx1-ubyte.gz'
        if not os.path.exists(lblpath):
            print('Cannot find %s; please add it.' % lblpath)
            sys.exit(1)
        with gzip.open(lblpath, 'rb') as lbpath:
            y1 = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        lblpath = 'data/fashion-mnist/t10k-labels-idx1-ubyte.gz'
        if not os.path.exists(lblpath):
            print('Cannot find %s; please add it.' % lblpath)
            sys.exit(1)
        with gzip.open(lblpath, 'rb') as lbpath:
            y2 = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        y = np.concatenate((y1, y2))
        datapath = 'data/fashion-mnist/train-images-idx3-ubyte.gz'
        if not os.path.exists(datapath):
            print('Cannot find %s; please add it.' % lblpath)
            sys.exit(1)
        with gzip.open(datapath, 'rb') as dtpath:
            X1 = np.frombuffer(dtpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(y1), 784)
        datapath = 'data/fashion-mnist/t10k-images-idx3-ubyte.gz'
        if not os.path.exists(datapath):
            print('Cannot find %s; please add it.' % lblpath)
            sys.exit(1)
        with gzip.open(datapath, 'rb') as dtpath:
            X2 = np.frombuffer(dtpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(y2), 784)
        X = np.concatenate((X1, X2))
        ids = np.arange(len(y))
        if n_samples > -1:
            orig_samples = len(y)
            keep = np.random.choice(range(len(y)), n_samples, replace=False)
            X = X[keep,:]
            y = y[keep]
            ids = ids[keep]
            print('Using %d of %d samples' % (n_samples, orig_samples))
    elif dataset == 'glass':
        fn = 'data/glass.data'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        Xy  = np.genfromtxt(fn, delimiter=',')
        X   = Xy[:, 1:-1]
        # Remap class labels for Kumar et al.'s constraints,
        # since np.genfromtxt() makes them floats.
        class_map = {1.0: 0, 2.0: 1, 3.0: 2, # no class 4
                     5.0: 3, 6.0: 4, 7.0: 5}
        y   = np.array([class_map[y] for y in Xy[:, -1]])
        ids = Xy[:, 0]
    elif dataset == 'ctg': # Cardiotocography
        fn = 'data/uci/CTG-data.csv'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        Xy  = np.genfromtxt(fn, delimiter=',', skip_header=1)
        if n_samples > Xy.shape[0]:
            raise ValueError('Cannot get %d from %d examples.' %
                             (n_samples, Xy.shape[0]))
        if n_samples > -1:
            keep = np.random.choice(range(len(Xy)), n_samples, replace=False)
            Xy = Xy[keep]
            print('Using %d random samples.' % n_samples)
        X   = Xy[:, :-2]
        # Remap class labels to ints for Kumar et al.'s constraints,
        # since np.genfromtxt() makes them floats..
        class_map = {1.0: 0, 2.0: 1, 3.0: 2}
        y = np.array([class_map[y] for y in Xy[:, -1]])
        ids = np.arange(len(y))
    elif dataset == 'haberman':
        fn = 'data/uci/haberman.data'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        Xy  = np.genfromtxt(fn, delimiter=',')
        X   = Xy[:, :-1]
        # Remap class labels to ints for Kumar et al.'s constraints,
        # since np.genfromtxt() makes them floats.
        class_map = {1.0: 0, 2.0: 1}
        y   = np.array([class_map[y] for y in Xy[:, -1]])
        ids = np.arange(len(y))
    elif dataset == 'heartH':
        fn = 'data/uci/heart.dat'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        Xy  = np.genfromtxt(fn, delimiter=' ')
        X   = Xy[:, :-1]
        # Remap class labels to ints for Kumar et al.'s constraints,
        # since np.genfromtxt() makes them floats..
        class_map = {1.0: 0, 2.0: 1}
        y   = np.array([class_map[y] for y in Xy[:, -1]])
        ids = np.arange(len(y))
    elif dataset == 'credit':
        fn = 'data/tom-credit-bias/train.csv'  # 7500 items
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        (X, y, _, _)  = parse_tom_credit(fn)
        ids = np.arange(len(y))
    elif dataset == 'letter':
        fn = 'data/uci/letter-recognition.data'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it from %s .' %
                  (fn, 'https://archive.ics.uci.edu/ml/datasets/letter+recognition'))
            sys.exit(1)
        # Specify how to convert the label (column 0)
        conv = {0: lambda s: float(ord(s) - 65)}
        Xy  = np.genfromtxt(fn, delimiter=',',
                            converters=conv)
        X   = Xy[:, 1:]
        # Convert class labels to ints
        # (converter above has to be float so genfromtxt() behaves)
        y = np.array([int(c) for c in Xy[:, 0]])
        ids = np.arange(len(y))
        if n_samples > len(y):
            raise ValueError('Cannot get %d from %d examples.' %
                             (n_samples, len(y)))
        if n_samples > -1:
            keep = np.random.choice(range(len(y)), n_samples, replace=False)
            X = X[keep]
            y = y[keep]
            ids = ids[keep]
    elif dataset == 'absenteeism':
        fn = 'data/uci/Absenteeism_at_work.csv'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        Xy = np.genfromtxt(fn, delimiter=';',
                           skip_header=1)
        X = Xy[:, 1:-1]
        # Convert class labels (absenteeism hours) to ints
        y = np.array([int(c) for c in Xy[:, -1]])
        # Make classes sequential
        classes = list(np.unique(y))
        y = np.array([classes.index(c) for c in y])
        ids = Xy[:, 0]
    elif dataset == 'covid':
        fn = 'data/covidpred/data/corona_tested_individuals_ver_006.english.csv'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        X = []
        y = []
        ids = []
        label_map = {'positive': 1,
                     'negative': 0}
        i = -1
        with open(fn) as inf:
            rdr = csv.reader(inf)
            header = next(rdr)
            if verbose:
                print('Features:', header)
            for vals in rdr:
                i += 1
                # Only keep the date range they used
                if (vals[0] < '2020-03-22' or
                    vals[0] > '2020-04-07'):
                    continue
                # Unclear what 'other' means; skip it
                if vals[6] == 'other':
                    continue
                # Missing values?
                if 'None' in vals[1:9]:
                    continue
                X += [[int(v) for v in vals[1:6]] +
                      [0 if vals[7] == 'No' else 1] + # age >= 60
                      [0 if vals[8] == 'male' else 1]] # gender
                y += [label_map[vals[6]]] # label
                ids += [(i, vals[0])]
        X = np.array(X)
        y = np.array(y)
        ids = np.array(ids)
        if n_samples > -1:
            keep = np.random.choice(range(len(y)), n_samples, replace=False)
            X = X[keep]
            y = y[keep]
            ids = ids[keep]
    elif dataset == 'starcraft':
        fn = 'data/starcraft/classification.csv'
        if not os.path.exists(fn):
            print('Cannot find %s; please add it.' % fn)
            sys.exit(1)
        # Specify how to convert the label (column 3)
        # Must be float so it can be a 2D array of same type
        conv = {3: lambda s: 1.0 if s.decode('utf-8') == 'TRUE' else 0.0}
        Xy  = np.genfromtxt(fn, delimiter=',', skip_header=1,
                            converters=conv)
        if n_samples > Xy.shape[0]:
            raise ValueError('Cannot get %d from %d examples.' %
                             (n_samples, Xy.shape[0]))
        if -1 < n_samples < Xy.shape[0]:
            keep = np.random.choice(range(len(Xy)), n_samples)
            Xy = Xy[keep]
            print('Using %d random samples.' % n_samples)
        X   = Xy[:, :-1]
        # Remap class labels to ints for Kumar et al.'s constraints
        # np.genfromtxt() makes them floats..
        class_map = {0.0: 0, 1.0: 1}
        y   = np.array([class_map[y] for y in Xy[:, -1]])
        ids = np.arange(len(y))
    elif (dataset in ['imagenet', 'cifar10', 'cifar100', 'msl'] or
          dataset.startswith('starcraft-formations')):
        # model_name should be like pretrained-<featmodel>-<predmodel>
        # where featmodel is used for the X vectors,
        # and predmodel is the source of the probabilities to calibrate
        model_name = model_name.split('-')[1] # remove 'pretrained-'
        print(model_name)
        if dataset == 'imagenet':
            d = np.load('data/imagenet/val_%s_latent.npz' % model_name)
            y = np.genfromtxt('data/imagenet/val_%s_labels.csv' % model_name)
        elif dataset.startswith('cifar'):
            datfile = 'data/cifar/%s_test_%s_latent.npz' % (dataset, model_name)
            d = np.load(datfile)
            print('loading features from %s' % datfile)
            #d = np.load('data/cifar/%s_test_%s_fc.npz' % (dataset, model_name))
            y = np.genfromtxt('data/cifar/%s_test_%s_labels.csv' % (dataset, model_name))
        elif dataset == 'msl':
            # Two files: val (calibration) and test
            # Since we can only return one set, we'll concatenate them
            basedir = 'data/msl-labeled-data-set-v2.1'
            d1 = np.load(os.path.join(basedir, '%s_val_%s_latent.npz' % (dataset, model_name)))
            y1 = np.genfromtxt(os.path.join(basedir, '%s_val_%s_labels.csv' %
                                            (dataset, model_name)))
            d2 = np.load(os.path.join(basedir, '%s_test_%s_latent.npz' % (dataset, model_name)))
            y2 = np.genfromtxt(os.path.join(basedir, '%s_test_%s_labels.csv' %
                                            (dataset, model_name)))
            d = {'latent': np.vstack((d1['latent'], d2['latent']))}
            y = np.concatenate((y1, y2))
        else:  # starcraft-formations
            # Two files: val (calibration) and test
            # Since we can only return one set, we'll concatenate them
            formation = dataset.split('-')[2] # assume it comes after starcraft-formations-
            fn = ('data/starcraft-formations-yunye/' +
                  'val_features_121221/resnet18_%s_val_data.npz' % formation)
            if not os.path.exists(fn):
                print('Cannot find %s; please add it.' % fn)
                sys.exit(1)
            d1 = np.load(fn)
            y1 = d1['labels']
            fn = ('data/starcraft-formations-yunye/' +
                  'test_features_121221/resnet18_%s_test_data.npz' % formation)
            if not os.path.exists(fn):
                print('Cannot find %s; please add it.' % fn)
                sys.exit(1)
            d2 = np.load(fn)
            y2 = d2['labels']
            d = {'latent': np.vstack((d1['features'], d2['features']))}
            y = np.concatenate((y1, y2))

        X = d['latent']
        #X = d['fc']
        y = y.astype(np.int)
        ids = np.arange(len(y))
        if n_samples not in [-1, len(y)]:
            raise ValueError('Cannot subsample %s (n=%d) for %s'
                             ' since probs are loaded elsewhere' % (dataset, len(y), model_name))
    else:
        print('Unknown data set %s' % dataset)
        sys.exit(1)

    # If desired, use the first n_samples
    if -1 < n_samples < len(y):
        X = X[:n_samples]
        y = y[:n_samples]
        ids = ids[:n_samples]
        print('Using first %d samples.' % n_samples)

    return (X, y, ids)


# Read in stored predicted probs for pre-trained models;
# return X feature vectors and probs for cal, test.
def read_pretrained_probs(dataset, clf_type):

    if clf_type.count('-') == 2:
        # model_name should be like pretrained-<featmodel>-<predmodel>
        # where featmodel is used for the X vectors,
        # and predmodel is the source of the probabilities to calibrate
        model_name = clf_type.split('-')[2]
    else:
        # otherwise, use the same model for features and probabilities
        model_name = clf_type.split('-')[1]
    # Initialize to empty
    cal_probs = []
    cal_logits = None
    # Load predicted probabilities from file
    if dataset == 'imagenet':
        d = np.load('data/imagenet/val_%s_probs.npz' % model_name)
        l = np.load('data/imagenet/val_%s_logits.npz' % model_name)
        classes = list(range(1000))
    elif dataset in ['cifar10', 'cifar100']:
        probsfile = 'data/cifar/%s_test_%s_probs.npz' % (dataset, model_name)
        d = np.load(probsfile)
        print('loading probs from %s' % probsfile)
        l = np.load('data/cifar/%s_test_%s_logits.npz' % (dataset, model_name))
        classes = list(range(10 if dataset == 'cifar10' else 100))
    elif dataset == 'msl':
        d = np.load('data/msl-labeled-data-set-v2.1/%s_val_%s_probs.npz' %
                    (dataset, model_name))
        cal_probs = d['probs']
        l = np.load('data/msl-labeled-data-set-v2.1/%s_val_%s_logits.npz' %
                    (dataset, model_name))
        cal_logits = l['logits']

        d = np.load('data/msl-labeled-data-set-v2.1/%s_test_%s_probs.npz' %
                    (dataset, model_name))
        l = np.load('data/msl-labeled-data-set-v2.1/%s_test_%s_logits.npz' %
                    (dataset, model_name))
        classes = list(range(cal_probs.shape[1]))
    elif dataset.startswith('starcraft-formations'):
        formation = dataset.split('-')[2] # assume it comes after starcraft-formations-
        fn = ('data/starcraft-formations-yunye/' +
              'val_features_121221/resnet18_%s_val_data.npz' % formation)
        dat = np.load(fn)
        cal_logits = dat['logits']
        # convert to probs
        sum_logit = np.sum(np.exp(cal_logits), 1)
        cal_probs = np.exp(cal_logits) / sum_logit[:, None]

        fn = ('data/starcraft-formations-yunye/' +
              'test_features_121221/resnet18_%s_test_data.npz' % formation)
        dat = np.load(fn)
        d, l = {}, {}
        l['logits'] = dat['logits']
        # convert to probs
        sum_logit = np.sum(np.exp(l['logits']), 1)
        d['probs'] = np.exp(l['logits']) / sum_logit[:, None]
        classes = list(range(d['probs'].shape[1]))
    else:
        print('Unknown pretrained data set: %s' % dataset)
        sys.exit(1)

    ret = {}
    ret['test_probs'] = d['probs']
    ret['test_logits'] = l['logits']
    ret['classes'] = classes
    ret['cal_probs'] = cal_probs
    ret['cal_logits'] = cal_logits

    print('Read probs:', ret['test_probs'].shape)

    return ret


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
    """
    ce = 0
    if len(probs.shape) == 1: # binary problem with 1D probs
        # Calculate MSE
        for (p, l) in zip(probs, labels):
            ce += (1 - p) ** 2 if l == classes[1] else p ** 2
    else: # multiclass (could be two-class)
        # Classes must line up with prob columns for this to make sense
        for (p, l) in zip(probs.tolist(), labels):
            for c in range(len(classes)):
                ce += (1 - p[c]) ** 2 if l == classes[c] else p[c] ** 2

    # Normalize by number of items
    ce = float(ce) / len(labels)

    return ce


if __name__ == "__main__":
    import doctest
    doctest.testmod()
