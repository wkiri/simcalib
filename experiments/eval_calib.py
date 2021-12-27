#!/usr/bin/env python3
# Evaluate proximity-based Bayesian calibration.
#
# Kiri Wagstaff
# April 14, 2021

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as pl
from progressbar import Counter as pCounter
from scipy import ndimage
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import calibration as cal # Kumar et al. (2019)
from simcalib import calib_sim, platt_scaling, \
    temp_scaling, temp_scaling_probs, hidden_hetero
from utils import get_dataset, get_classifier, brier, read_pretrained_probs
from calib import calib_info

# Include both TS and Platt in this list.
# Binary problems will omit TS, and multi-class problems will omit Platt.
calib_methods = ['uncal', 'platt', 'ts', 'hist', 'sba', 'swc', 'swc-hh']


# Evaluate all calibration methods in the calib_methods list.
# - clf is only used by 'retrain'
# - prob_radius is only used by SWC-HH
# - cal/test_logits are only used by temperature scaling
def eval_calib(cal_probs, test_probs, classes,
               X_cal, y_cal, X_test, y_test,
               prob_radius, sim_method,
               resfile, clf=None, cal_logits=None, test_logits=None,
               seed=0, verbose=1):

    res = {}
    # Load previous results, if available 
    if os.path.exists(resfile):
        with open(resfile, 'rb') as inf:
            print(' Loading results from %s' % resfile)
            res = pickle.load(inf)

    for m in calib_methods:
        # Only apply a method if results do not exist
        if m not in res:
            print('Evaluating %s' % m)
            if m not in res:
                res[m] = {}

            if m == 'uncal':
                res[m]['test_probs'] = test_probs

            elif m == 'retrain':
                # Retrain same model type from scratch on the calibration set
                clf_retrain = clone(clf)
                clf_retrain.fit(X_cal, y_cal)
                res[m]['test_probs'] = clf_retrain.predict_proba(X_test)

            elif m == 'platt':
                res[m]['test_probs'] = platt_scaling(cal_probs, y_cal,
                                                     test_probs)

            elif m == 'ts':
                # Calibrate with temperature scaling
                # Use logits if available; else use probabilities
                if cal_logits is not None:
                    _, res[m]['test_probs'] = temp_scaling(
                        cal_logits, y_cal, len(classes),
                        test_logits, optim='nll')
                else:
                    _, res[m]['test_probs'] = temp_scaling_probs(
                        cal_probs, y_cal, len(classes),
                        test_probs, optim='nll')

            elif m == 'hist':
                # Calibrate with histogram binning (From Kumar et al.)
                # Use 100 bins if possible, or if not,
                # the min number of unique prob values per class (column)
                n_bins = min(100, min([len(np.unique(cal_probs[:, i]))
                                       for i in range(len(classes))]))
                if verbose > 0:
                    print('hist: %d bins' % n_bins)
                c = cal.HistogramMarginalCalibrator(num_calibration=len(y_cal),
                                                    num_bins=n_bins)
                c.train_calibration(cal_probs, y_cal)
                res[m]['test_probs'] = c.calibrate(test_probs)

            elif m == 'scalebin':
                # Calibrate with histogram binning (From Kumar et al.)
                # Use 100 bins if possible, or if not,
                # the min number of unique prob values per class (column)
                n_bins = min(100, min([len(np.unique(cal_probs[:, i]))
                                       for i in range(len(classes))]))
                c = cal.PlattBinnerMarginalCalibrator(
                    num_calibration=len(y_cal), num_bins=n_bins)
                c.train_calibration(cal_probs, y_cal)
                res[m]['test_probs'] = c.calibrate(test_probs)

            elif m == 'sba':
                # Similarity-binning averaging (Bella et al., 2012)
                # with (inverted) Euclidean distance for similarity
                # in augmented feature space with 10 nearest neighbors
                # (averaged, no similarity-based weighting).
                res[m]['test_probs'], res[m]['sim_mass'] = \
                    calib_sim(np.hstack((X_cal, cal_probs)), y_cal,
                              np.hstack((X_test, test_probs)), test_probs,
                              nn=10, sim_method='sim_euclid', weighted=False,
                              verbose=verbose)

            elif m in ['swc', 'platt-swc', 'ts-swc', 'swc-hh']:
                # Similarity-weighted calibration
                # in augmented feature space with all neighbors.

                # Compute globally calibrated solution
                if len(classes) == 2:
                    cal_probs_g = platt_scaling(cal_probs, y_cal, cal_probs)
                    test_probs_g = platt_scaling(cal_probs, y_cal, test_probs)
                else:
                    # Temperature scaling
                    # Use logits if available; else use probabilities
                    if cal_logits is not None:
                        _, cal_probs_g = temp_scaling(
                            cal_logits, y_cal, len(classes),
                            cal_logits, optim='nll')
                        _, test_probs_g = temp_scaling(
                            cal_logits, y_cal, len(classes),
                            test_logits, optim='nll')
                    else:
                        _, cal_probs_g = temp_scaling_probs(
                            cal_probs, y_cal, len(classes),
                            cal_probs, optim='nll')
                        _, test_probs_g = temp_scaling_probs(
                            cal_probs, y_cal, len(classes),
                            test_probs, optim='nll')
                if m in ['swc', 'swc-hh']:
                    # Use uncalibrated probabilities as input
                    test_probs_in = test_probs
                    cal_probs_in = cal_probs
                else:
                    # First apply Platt/TS, then refine with SWC/SWC-HH
                    test_probs_in = test_probs_g
                    cal_probs_in = cal_probs_g
                scaler = StandardScaler()
                scaler.fit(cal_probs_in)
                cal_probs_scaled = scaler.transform(cal_probs_in)
                test_probs_scaled = scaler.transform(test_probs_in)

                if m == 'swc-hh':
                    # Calcul[ate hidden heterogeneity.
                    if 'swc' not in res:
                        res['swc'] = {}
                    if 'test_HH' not in res['swc']:
                        res['swc']['test_HH'], _, _ = \
                            hidden_hetero(X_cal, y_cal,
                                          cal_probs_in, test_probs_in,
                                          r=prob_radius)
                        # Otherwise assume HH was already calculated
                    hh = res['swc']['test_HH']
                else:
                    hh = None

                res[m]['test_probs'], res[m]['sim_mass'] = \
                    calib_sim(np.hstack((X_cal, cal_probs_scaled)), y_cal,
                              np.hstack((X_test, test_probs_scaled)),
                              test_probs_in, nn=-1, hh=hh,
                              sim_method=sim_method, verbose=verbose)

            # Evaluate predictions
            calib_test_probs = res[m]['test_probs']
            test_preds = np.argmax(calib_test_probs, axis=1)
            res[m]['brier'] = brier(calib_test_probs, y_test, classes)
            res[m]['acc'] = accuracy_score(y_test, test_preds)

            # Save new results
            with open(resfile, 'wb') as outf:
                print(' Saving results to %s' % resfile)
                pickle.dump(res, outf)

        print(' %15s: %.4f Brier, %.4f acc' %
              (m, res[m]['brier'], res[m]['acc']))

    return res


# Perform domain shift on the first n_rotate items
def shift_test_set(X, n_rotate):
    if n_rotate > len(X):
        raise ValueError('Cannot rotate %d > %d items' %
                         (n_rotate, len(X)))
    bg_value = -0.5 # Background value
    for i in range(n_rotate):
        img = np.reshape(X[i], (28, 28))
        # Rotates 90 degrees counter-clockwise
        rot_img = ndimage.rotate(img, 90, reshape=False,
                                 cval=bg_value)
        rot_img = (rot_img.astype(int)).astype(float)
        #print(np.min(rot_img), np.max(rot_img))
        X[i]  = np.reshape(rot_img, 28 * 28)

    return X


# Evaluate proximity-based calibration with Bayesian update
# for different base classifiers
def main(res_dir, dataset, n_cal, clf_type,
         prob_radius, sim_method,
         n_rotate=0, plot_results=True, seed=0):

    np.random.seed(seed)

    # Check arguments
    if not os.path.isdir(res_dir):
        print('Creating output directory %s' % res_dir)
        os.mkdir(res_dir)

    if prob_radius < 0:
        raise ValueError('Error: prob_radius (%f) must be >= 0' % prob_radius)

    if n_cal < 1:
        raise ValueError('Error: n_cal (%d) must be > 0' % n_cal)

    #(X, y, ids) = get_dataset(dataset, n_cal, model_name=clf_type)
    if (dataset in ['imagenet', 'msl'] or
        dataset.startswith('starcraft-formations') or
        dataset == 'covid'):
        (X, y, ids) = get_dataset(dataset, -1, model_name=clf_type, seed=seed)
    else:
        (X, y, ids) = get_dataset(dataset, 10000, model_name=clf_type,
                                  seed=seed)
    classes = np.unique(y)

    if len(classes) == 2:
        for m in ['ts', 'ts-swc']:
            if m in calib_methods:
                calib_methods.remove(m)
    else: # multiclass
        for m in ['platt', 'platt-swc']:
            if m in calib_methods:
                calib_methods.remove(m)

    # Split into training and test sets
    if dataset == 'covid':
        # "The training-validation set consisted of records from
        # 51,831 tested individuals (of whom 4769 were confirmed to
        # have COVID-19), from the period March 22th, 2020 through
        # March 31st, 2020. The test set contained data from the
        # subsequent week, April 1st through April 7th (47,401 tested
        # individuals, of whom 3624 were confirmed to have
        # COVID-19). The training-validation set was further divided
        # to training and validation sets at a ratio of 4:1 (Table
        # 1)."
        # Split by date; training before 4/1/2020
        tr = ids[:, 1] < '2020-04-01'
        X_test = X[np.logical_not(tr)]
        y_test = y[np.logical_not(tr)]
        X_train, X_cal, y_train, y_cal = \
            train_test_split(X[tr], y[tr], test_size=.25, random_state=seed)
    elif 'pretrained' in clf_type:
        # Read in predicted probs
        ret = read_pretrained_probs(dataset, clf_type)
        classes = ret['classes']

        if dataset == 'msl':
            # Pre-defined cal/test sets
            X_train, y_train = np.array(()), np.array(())
            X_cal, y_cal = X[:300], y[:300]
            X_test, y_test = X[300:], y[300:]
            
            cal_probs = ret['cal_probs']
            cal_logits = ret['cal_logits']
            test_probs = ret['test_probs']
            test_logits = ret['test_logits']
        elif dataset.startswith('starcraft-formations'):
            # Pre-defined cal/test sets
            X_train, y_train = np.array(()), np.array(())
            X_cal, y_cal = X[:1800], y[:1800]
            X_test, y_test = X[1800:], y[1800:]
            
            cal_probs = ret['cal_probs']
            cal_logits = ret['cal_logits']
            test_probs = ret['test_probs']
            test_logits = ret['test_logits']
            # Subset cal if desired
            if n_cal < len(y_cal):
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_cal,
                                             random_state=seed)
                # There is only one but this generator wants a loop
                for cal_idx, _ in sss.split(X_cal, y_cal):
                    print('selecting %d cal items' % n_cal)
                    X_cal = X[cal_idx]
                    y_cal = y[cal_idx]
                    cal_probs = cal_probs[cal_idx]
                    cal_logits = cal_logits[cal_idx]
        else:
            # Re-split the data into cal and test (no train)
            X_train, y_train = [], []
            # We always want 5k test, and n_cal indicates cal size
            # Select val/test to preserve class distribution
            te_size = 10000 if dataset == 'imagenet' else 5000
            sss = StratifiedShuffleSplit(n_splits=1, test_size=te_size,
                                         random_state=seed)
            #sss = StratifiedShuffleSplit(n_splits=1, test_size=100,
            # There is only one but this generator wants a loop
            for cal_idx, test_idx in sss.split(X, y):
                X_cal = X[cal_idx]
                y_cal = y[cal_idx]
                cal_probs = ret['test_probs'][ids][cal_idx]
                cal_logits = ret['test_logits'][ids][cal_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]
                test_probs = ret['test_probs'][ids][test_idx]
                test_logits = ret['test_logits'][ids][test_idx]
            # Sub-select only n_cal from cal (may not be class-balanced)
            X_cal = X_cal[:n_cal]
            y_cal = y_cal[:n_cal]
            cal_probs = cal_probs[:n_cal]
            cal_logits = cal_logits[:n_cal]
            # First 100 test items
            #X_test = X_test[:100]
            #y_test = y_test[:100]
            #test_probs = test_probs[:100]
            #test_logits = test_logits[:100]

        # No training data
        X_train, y_train = np.array(()), np.array(())
    else:
        trte_size = {'letter': 2000,
                     'mnist10': 1000}
        te_size = trte_size[dataset] if dataset in trte_size else 500
        # Create a fixed set of te_size test items
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=te_size, random_state=seed)

        # Create a fixed set of tr_size training items
        tr_size = trte_size[dataset] if dataset in trte_size else 500
        # Everything else (up to n_cal) is calibration
        X_cal, X_train, y_cal, y_train = \
            train_test_split(X_train, y_train, test_size=tr_size,
                             random_state=seed)
        X_cal = X_cal[:n_cal]
        y_cal = y_cal[:n_cal]
        
    file_basename = 'n%d_clf%s_r%s_seed%d' % \
        (n_cal, clf_type, prob_radius, seed)
    resfilebase = os.path.join(res_dir, 'res-%s-%s.pkl' %
                               (dataset, file_basename))

    print('train/cal/test:', len(y_train), len(y_cal), len(y_test))
    # Check representation
    n_classes = [len(set(yvals)) for yvals in [y_cal, y_test]]
    if np.any([nc < len(classes) for nc in n_classes]):
        print('Lack of class representation: '
              '%d in full data set vs. %d cal, %d test' %
              (len(classes), n_classes[0], n_classes[1]))
        print(' Try specifying a larger sample.')
        sys.exit(1)
    print('%d features' % X_test.shape[1])

    # -1. If desired, rotate some test items
    if (n_rotate > 0 and
        (dataset.startswith('mnist') or
         dataset == 'fashion-mnist')):
        X_test = shift_test_set(X_test, n_rotate)
    
    # 0. Scale the data
    scaler = StandardScaler()
    if clf_type.startswith('pretrained'):
        scaler.fit(X_cal)
    else:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)

    # 1. Train the base classifier
    if clf_type.startswith('pretrained'):
        clf = None
    else:
        cal_logits = None
        test_logits = None
        (clf, _) = get_classifier(clf_type)

        clf.fit(X_train, y_train)
        cal_probs = clf.predict_proba(X_cal)
        test_probs = clf.predict_proba(X_test)
    print(cal_probs.shape)

    # 2. Calibrate its predictions
    res = eval_calib(cal_probs, test_probs, classes,
                     X_cal, y_cal, X_test, y_test,
                     prob_radius, sim_method,
                     resfilebase, clf=clf,
                     cal_logits=cal_logits, test_logits=test_logits,
                     seed=seed)

    print('\nSummary:')
    for m in res.keys():
        if 'brier' in res[m]:
            print(' %15s: %.4f Brier, %.4f acc' %
                  (m, res[m]['brier'], res[m]['acc']))
    if plot_results:
        # 3. Plot results
        pl.figure()
        labels = [calib_info[c][0] for c in calib_methods]
        colors = [calib_info[c][1] for c in calib_methods]
        for (metric, yl, ymin, ymax) in \
            [('acc', 'Accuracy', 0.80, 0.85),
             ('brier', 'Brier score (MSE)', 0.0, 0.2),
             #('brier', 'Brier score (MSE)', 0.0, 0.1),
             ]:
            pl.clf()
            xvals = range(1, len(calib_methods) + 1)
            pl.bar(xvals, [res[c][metric] for c in calib_methods],
                   color=colors)
            pl.xticks(xvals, labels, fontsize=12, rotation=30)
            pl.yticks(fontsize=14)
            #pl.ylim((ymin, ymax))
            pl.title('%s: %s, %s' % (dataset, clf_type, yl), fontsize=18)
            pl.savefig(os.path.join(res_dir, '%s-cmp-%s-%s.pdf' %
                                    (dataset, metric, file_basename)),
                       bbox_inches='tight')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('res_dir',
                        help='Where to save plots (.pdf)')
    parser.add_argument('-d', '--dataset', default='moons',
                        help='Dataset (default: %(default)s)')
    parser.add_argument('-n', '--n_cal', type=int, default=1000,
                        help='Number of calibration items (default: 1000)')
    parser.add_argument('-c', '--clf_type', default='DT',
                        help='Classifier type (default: %(default)s)')
    parser.add_argument('-r', '--prob_radius', type=float, default=0.1,
                        help='Radius in the probability simplex to define hidden heterogeneity'
                        ' neighborhood (default: %(default)s)')
    parser.add_argument('-m', '--sim_method', default='RFprox',
                        choices=['cosine', 'rbf', 'RFprox', 'Isoprox', 'all_one',
                                 'cosine-1NN', 'rbf-1NN', 'RFprox-1NN', 'Isoprox-1NN'],
                        help='SWC similarity method (default: %(default)s)')
    parser.add_argument('--n_rotate', type=int, default=0,
                        help=('Number of test set items to rotate'
                              '(default: %(default)s);'
                              'only works for mnist/fashion-mnist'))
    parser.add_argument('-p', '--plot_results', action='store_false',
                        default=True,
                        help='Omit reliability and performance plots (default: save)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed (default: %(default)s)')

    args = parser.parse_args()
    main(**vars(args))
