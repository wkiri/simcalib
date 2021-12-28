#!/usr/bin/env python3
# Plot results from proximity-based calib experiments (multiple trials)
#
# Kiri Wagstaff
# April 15, 2021

import sys
import os
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from utils import get_dataset, brier
from calib import calib_info
from eval_calib import calib_methods


# Read in the data set to obtain the per-item test set labels
def read_test_labels(dataset, clf_type, seed):

    np.random.seed(seed)

    if (dataset in ['imagenet', 'msl'] or
        dataset.startswith('starcraft-formations')):
        (_, y, _) = get_dataset(dataset, -1, model_name=clf_type,
                                seed=seed, verbose=False)
    else:
        (_, y, _) = get_dataset(dataset, 10000, model_name=clf_type,
                                seed=seed, verbose=False)

    # Split into training and test sets - we only need test set labels
    if dataset.startswith('cifar') or dataset == 'imagenet':
        te_size = 10000 if dataset == 'imagenet' else 5000
        sss = StratifiedShuffleSplit(n_splits=1, test_size=te_size,
                                     random_state=seed)
        # There is only one but this generator wants a loop
        for _, test_idx in sss.split(y, y):
            y_test = y[test_idx]
    elif dataset == 'msl':
        # Pre-defined cal/test sets
        y_test = y[300:]
    elif dataset.startswith('starcraft-formations'):
        # Pre-defined cal/test sets
        y_test = y[1800:]
    else:
        trte_size = {'letter': 2000,
                     'mnist10': 1000}
        te_size = trte_size[dataset] if dataset in trte_size else 500
        # Create a fixed set of te_size test items
        _, y_test = train_test_split(y, test_size=te_size, random_state=seed)

    return y_test


# Read in results from pickled file(s) (one per seed/trial)
def read_results(metrics, resfilebase, hh_thresh=0, verbose=True):
    res = {}
    # Add SWC so we can get test_HH
    for c in calib_methods:
        res[c] = {}
        for m in metrics + ['test_probs', 'test_HH', 'high_HH',
                            'all_test_HH', 'sim_mass']:
            res[c][m] = []
    # Just in case
    for c in ['ts', 'platt']:
        if c not in calib_methods:
            res[c] = {}
            for m in metrics + ['test_probs']:
                res[c][m] = []
    if 'swc' not in calib_methods:
        res['swc'] = {}
        res['swc']['test_HH'] = []

    max_seed = 100
    for seed in range(max_seed):
        resfile = resfilebase % (seed)
        if not os.path.exists(resfile):
            #print('Could not find results file %s' % resfile)
            break
        with open(resfile, 'rb') as inf:
            if verbose:
                print(' Loading results from %s' % resfile)
            r = pickle.load(inf)
            for c in calib_methods:
                # Skip missing entries
                if c not in r:
                    continue
                for m in metrics:
                    res[c][m] += [r[c][m]]
                # These aren't metrics; store one value per item per seed
                res[c]['test_probs'] += [r[c]['test_probs']]
                if c in ['swc']:
                    if 'test_HH' not in r[c]:
                        print('No test_HH in %s' % c)
                    else:
                        # Only store the mean, since for some data sets
                        # (like covid) the test set size isn't fixed
                        # and ragged arrays cause problems later.
                        res[c]['all_test_HH'] += [r[c]['test_HH']]
                        res[c]['test_HH'] += [np.mean(r[c]['test_HH'])]
                        res[c]['high_HH'] += [np.mean(r[c]['test_HH']
                                                      >= hh_thresh)]
                if c in ['swc', 'swc-hh']:
                    res[c]['sim_mass'] += [r[c]['sim_mass']]
            # Grab SWC test_HH if not already included
            if 'swc' not in calib_methods:
                if 'swc' in r:  # skip incomplete results
                    res['swc']['test_HH'] += [np.mean(r['swc']['test_HH'])]

    return res


# Specify y axis limits based on the data set
def get_axis_limits(dataset):
    params = {}
    if dataset == 'credit':
        params = {'acc': ('Accuracy', 0.75, 0.82),
                  'brier': ('Brier score (MSE)', 0.0, 0.35)}
    elif dataset == 'ctg':
        params = {'acc': ('Accuracy', 0.80, 1.0),
                  'brier': ('Brier score (MSE)', 0.0, 0.2)}
    elif dataset == 'letter':
        params = {'acc': ('Accuracy', 0.64, 0.94),
                  'brier': ('Brier score (MSE)', 0.0, 0.60)}
    elif dataset == 'moons':
        params = {'acc': ('Accuracy', 0.75, 0.95),
                  'brier': ('Brier score (MSE)', 0.0, 0.20)}
    elif dataset.startswith('mnist-'):
        params = {'acc': ('Accuracy', 0.60, 1.0),
                  'brier': ('Brier score (MSE)', 0.0, 0.2)}
    elif dataset == 'mnist10':
        params = {'acc': ('Accuracy', 0.65, 0.95),
                  'brier': ('Brier score (MSE)', 0.0, 0.4)}
    elif dataset == 'starcraft':
        params = {'acc': ('Accuracy', 0.80, 0.90),
                  'brier': ('Brier score (MSE)', 0.0, 0.30)}
    elif dataset == 'cifar10':
        params = {'acc': ('Accuracy', 0.90, 1.0),
                  'brier': ('Brier score (MSE)', 0.0, 0.15)}
    elif dataset == 'cifar100':
        params = {'acc': ('Accuracy', 0.65, 0.85),
                  'brier': ('Brier score (MSE)', 0.0, 0.5)}
    elif dataset == 'imagenet':
        params = {'acc': ('Accuracy', 0.65, 0.85),
                  'brier': ('Brier score (MSE)', 0.0, 0.5)}
    elif dataset == 'msl':
        params = {'acc': ('Accuracy', 0.65, 0.75),
                  'brier': ('Brier score (MSE)', 0.0, 0.5)}
    return params


# Generate and save a bar plot for the specified metric,
# yl (y axis label), ymin, and ymax range.
def plot_bar(dataset, clf_type, calib_methods_use, res,
             metric, res_dir, file_basename):

    labels = [calib_info[c][0] for c in calib_methods_use]
    colors = [calib_info[c][1] for c in calib_methods_use]
    params = get_axis_limits(dataset)
    (yl, ymin, ymax) = params[metric]

    pl.clf()
    xvals = range(1, len(calib_methods_use) + 1)
    pl.bar(xvals, [res[c]['avg_%s' % metric] for c in calib_methods_use],
           yerr=[res[c]['std_%s' % metric] for c in calib_methods_use],
           color=colors)
    pl.xticks(xvals, labels, fontsize=12, rotation=30)
    pl.yticks(fontsize=14)
    pl.ylim((ymin, ymax))
    # This assumes all methods have the same number of trials
    pl.title('%s: %s, %s (%d trials)' %
             (dataset, clf_type, yl, len(res['uncal']['brier'])),
             fontsize=18)
    figname = os.path.join(res_dir, 'all-%s-cmp-%s-%s.pdf' %
                           (dataset, metric, file_basename))
    pl.savefig(figname, bbox_inches='tight')
    print('Saved %s' % figname)


# Generate and save a per-class bar plot for the specified metric,
# yl (y axis label), ymin, and ymax range.
def plot_bar_perclass(dataset, clf_type, calib_methods_use, res,
                      metric, res_dir, file_basename):

    pl.clf()

    n_trials = len(res['uncal']['test_probs'])

    # Compute avg/std metric value across seeds
    mval = {}
    hh = {}
    width = 0.95 / len(calib_methods_use)
    for seed in range(n_trials):
        np.random.seed(seed)

        # Load in the test data so we can compute per-class metrics
        y_test = read_test_labels(dataset, clf_type, seed)
        classes = np.unique(y_test)

        for k in classes:
            in_this_class = y_test == k
            y_test_cl = y_test[in_this_class]
            if k not in mval:
                mval[k] = {}
            if k not in hh:
                hh[k] = {}
            for c in calib_methods_use:
                if c not in res:
                    continue
                if c not in mval[k]:
                    mval[k][c] = []
                if seed >= len(res[c]['test_probs']):
                    continue
                tp_cl = res[c]['test_probs'][seed][in_this_class]
                if metric == 'brier':
                    mval[k][c] += [brier(tp_cl, y_test_cl, classes)]
                elif metric == 'acc':
                    tpr_cl = np.argmax(tp_cl, axis=1)
                    mval[k][c] += [accuracy_score(y_test_cl, tpr_cl)]
                if c not in hh[k]:
                    hh[k][c] = []
                if c in ['swc', 'platt-swc', 'ts-swc']:
                    hh[k][c] += [np.mean(res[c]['all_test_HH'][seed][in_this_class])]
    for i, k in enumerate(classes):
        h = pl.bar([i - 0.4 + pos * width for pos in range(len(calib_methods_use))],
                   [np.mean(mval[k][c]) for c in calib_methods_use],
                   yerr=[scipy.stats.sem(mval[k][c]) for c in calib_methods_use],
                   color=[calib_info[c][1] for c in calib_methods_use],
                   width=width)

    pl.yticks(fontsize=14)
    if 'swc' in hh[k]:
        xl = ['%s\n%.2f' % (k, np.mean(hh[k]['swc'])) for k in classes]
    else:
        xl = classes
    pl.xticks(range(len(classes)), xl, fontsize=12)
    #pl.ylim((ymin, ymax))
    pl.ylabel(metric, fontsize=14)
    pl.xlabel('Classes', fontsize=14)
    pl.legend(h, [calib_info[c][0] for c in calib_methods_use],
              fontsize=8)
    # This assumes all methods have the same number of trials
    pl.title('%s: %s (%d trials)' % (dataset, clf_type, n_trials), fontsize=18)
    figname = os.path.join(res_dir, 'perclass-%s-%s-cmp-%s-%s.pdf' %
                           (dataset, clf_type, metric, file_basename))
    pl.savefig(figname, bbox_inches='tight')
    print('Saved %s' % figname)


# Generate and save a rejection curve: accuracy as a function of
# percent of the predictions that are filtered (abstained)
def plot_rejection_curve(dataset, clf_type, calib_methods_use, res,
                         res_dir, file_basename):

    pl.clf()
    # Compute accuracy as a fn of percent rejected per trial, then average
    thresh = np.arange(0.0, 1.0, 0.02)
    n_trials = len(res['uncal']['test_probs'])
    for c in calib_methods_use:
        acc_rej = np.zeros((n_trials, len(thresh)))  # fn of perc rejected
        for seed in range(n_trials):
            np.random.seed(seed)
            y_test = read_test_labels(dataset, clf_type, seed)
            n_test = len(y_test)

            for t in range(len(thresh)):
                # Calc prob threshold as the thresh[t]-th value in sorted max probs
                # Note: this approach groups all items with same posterior together,
                # creating flat ranges in rejection curve (instead of wiggles
                # induced by arbitrary order of items with same posterior in sort)
                #min_thresh = np.sort(np.max(res[c]['test_probs'][seed], axis=1))[
                #    int(thresh[t] * n_test)]
                #print(thresh[t], n_test, int(thresh[t] * n_test), min_thresh)
                #sel = np.max(res[c]['test_probs'][seed], axis=1) >= min_thresh
                n_keep = int((1 - thresh[t]) * n_test)
                ind = np.argsort(np.max(res[c]['test_probs'][seed], axis=1))[-n_keep:]
                sel = np.array([False] * n_test)
                sel[ind] = True
                if np.sum(sel) == 0:
                    acc_rej[seed][t] = np.nan
                else:
                    acc_rej[seed][t] = accuracy_score(
                        np.argmax(res[c]['test_probs'][seed][sel], axis=1), y_test[sel])

        # Note: some algs may have np.nan (zero entries) for all trials
        # with a given threshold.   This throws a RuntimeWarning even with nanmean.
        pl.plot(thresh * 100, np.nanmean(acc_rej, axis=0),
                color=calib_info[c][1],
                label=calib_info[c][0])

    pl.xlabel('Percent filtered', fontsize=14)
    pl.ylabel('Accuracy', fontsize=14)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=12)
    pl.legend()
    figname = os.path.join(res_dir, 'reject-%s-%s.pdf' %
                           (dataset, file_basename))
    pl.savefig(figname, bbox_inches='tight')
    print('Saved %s' % figname)


# Generate and save a rejection curve that uses confidence thresholds
def plot_rejection_curve2(dataset, clf_type, calib_methods_use, res,
                          res_dir, file_basename):

    pl.clf()
    # Compute accuracy as a fn of confidence threshold, then average
    #thresholds = np.arange(0.0, 1.0, 0.01)
    thresholds = np.concatenate((np.arange(0.0, 0.5, 0.1),
                                 np.arange(0.5, 0.7, 0.05),
                                 np.arange(0.7, 0.9, 0.02),
                                 np.arange(0.9, 1.01, 0.001)))
    n_trials = len(res['uncal']['test_probs'])
    for c in calib_methods_use:
        acc_rej = np.zeros((n_trials, len(thresholds)))  # fn of perc rejected
        perc_rej = np.zeros((n_trials, len(thresholds)))
        for seed in range(n_trials):
            np.random.seed(seed)
            y_test = read_test_labels(dataset, clf_type, seed)
            n_test = len(y_test)

            for (t, thresh) in enumerate(thresholds):
                if len(res[c]['test_probs']) <= seed:
                    print('Warning: %s is incomplete' % c)
                    acc_rej[seed][t] = np.nan
                    perc_rej[seed][t] = np.nan
                    continue
                sel = np.max(res[c]['test_probs'][seed], axis=1) >= thresh
                if np.sum(sel) == 0:
                    acc_rej[seed][t] = np.nan
                    perc_rej[seed][t] = 1.0
                else:
                    acc_rej[seed][t] = accuracy_score(
                        np.argmax(res[c]['test_probs'][seed][sel], axis=1), y_test[sel])
                    perc_rej[seed][t] = (n_test - np.sum(sel)) / n_test
                    #print(thresh, np.sum(sel), n_test, perc_rej[seed][t])

        # Note: some algs may have np.nan (zero entries) for all trials
        # with a given threshold.   This throws a RuntimeWarning even with nanmean.
        pl.plot(np.nanmean(perc_rej, axis=0) * 100,
                np.nanmean(acc_rej, axis=0),
                color=calib_info[c][1],
                label=calib_info[c][0])

    pl.xlabel('Percent rejected', fontsize=14)
    pl.ylabel('Accuracy', fontsize=14)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=12)
    pl.legend()
    figname = os.path.join(res_dir, 'reject-perc-%s-%s.pdf' %
                           (dataset, file_basename))
    pl.savefig(figname, bbox_inches='tight')
    print('Saved %s' % figname)


# Plot results from multiple trials
def main(res_dir, dataset, n_cal, clf_type, prob_radius, per_class):

    # Check arguments
    if not os.path.isdir(res_dir):
        print('Could not find directory %s' % res_dir)
        sys.exit(1)

    file_basename = 'n%d_clf%s_r%s' % (n_cal, clf_type, prob_radius)
    resfilebase = os.path.join(res_dir, 'res-%s-%s_seed%%s.pkl' %
                               (dataset, file_basename))

    metrics = ['brier', 'acc']

    # Read in results
    res = read_results(metrics, resfilebase)
    if len(res['uncal']['brier']) == 0:
        print('Did not find any results for %s (%s) '
              'with %d samples and prob. radius %s in %s.' %
              (dataset, clf_type, n_cal, prob_radius, res_dir))
        sys.exit(1)

    # Prune calib_methods to those with at least one Brier score result
    calib_methods_use = [c for c in calib_methods
                         if len(res[c]['brier']) > 0]

    # Average across seeds
    for c in calib_methods_use:
        if len(res[c]['brier']) == 0:
            continue
        print('%15s: ' % c, end='')
        for m in metrics:
            res[c]['avg_%s' % m] = np.mean(res[c][m])
            res[c]['std_%s' % m] = scipy.stats.sem(res[c][m])
            print('%s: %.4f (%.4f), ' % (m, res[c]['avg_%s' % m],
                                         res[c]['std_%s' % m]), end='')
        if c in ['swc', 'platt-swc', 'ts-swc']:
            # Avg across all seeds
            print('avg HH: %.4f' % np.mean(np.array(res[c]['test_HH'])), end='')
        print()

    # Generate and save bar plot for each metric
    for metric in metrics:
        plot_bar(dataset, clf_type, calib_methods_use, res,
                 metric, res_dir, file_basename)

        # If desired, output a per-class plot
        if per_class:
            plot_bar_perclass(dataset, clf_type, calib_methods_use, res,
                              metric, res_dir, file_basename)

    # Plot rejection curve
    plot_rejection_curve(dataset, clf_type, calib_methods_use, res,
                         res_dir, file_basename)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('res_dir',
                        help='Where to save plots (.pdf)')
    parser.add_argument('-d', '--dataset', default='synth',
                        help='Dataset (default: %(default)s)')
    parser.add_argument('-n', '--n_cal', type=int, default=1000,
                        help='Number of calibration items to use'
                        ' (default: %(default)s)')
    parser.add_argument('-c', '--clf_type', default='DT',
                        help='Classifier type (default: %(default)s)')
    parser.add_argument('-r', '--prob_radius', type=float, default=0.1,
                        help='Radius in the probability simplex to define'
                        ' hidden heterogeneity neighborhood'
                        ' (default: %(default)s)')
    parser.add_argument('-p', '--per_class', default=False, action='store_true',
                        help='Generate per-class plot (default: %(default)s)')

    args = parser.parse_args()
    main(**vars(args))
