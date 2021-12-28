#!/usr/bin/env python3
# Create a reliability diagram using KDE estimates for accuracy.
#
# Kiri Wagstaff
# August 17, 2021

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as pl
from calib import calib_info
from eval_calib import calib_methods
from plot_results import read_test_labels


# Read in predicted probabilities from pickled file
def read_probs(resfile):
    res = {}
    for c in calib_methods:
        res[c] = {}
    # Just in case
    for c in ['ts', 'platt']:
        if c not in calib_methods:
            res[c] = {}

    if not os.path.exists(resfile):
        print('Could not find results file %s' % resfile)
        sys.exit(1)

    with open(resfile, 'rb') as inf:
        print(' Loading results from %s' % resfile)
        r = pickle.load(inf)
        for c in calib_methods:
            # Skip missing entries
            if c not in r:
                continue
            res[c]['test_probs'] = r[c]['test_probs']
            for m in ['k_acc', 'k_ece']:
                if m in r[c]:
                    res[c][m] += [r[c][m]]

    return res


# Written by zhang64: https://github.com/zhang64-llnl/Mix-n-Match-Calibration
def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin + xmax) / 2
        return np.concatenate(((2 * xmin - d[d < xmed]).reshape(-1,1),
                               d,
                               (2 * xmax - d[d >= xmed]).reshape(-1,1)))
    elif xmin is not None:
        return np.concatenate((2 * xmin - d, d))
    elif xmax is not None:
        return np.concatenate((d, 2 * xmax - d))
    else:
        return d


# Compute the kernel ECE as described by Zhang et al. (2020)
# https://github.com/zhang64-llnl/Mix-n-Match-Calibration
# Kernel = triweight
# bandwidth (h) = 1.06 * (std(prob vals)*2)^(1/5)
# Updates include:
# 1) Support for binary problems, in which case the
#    reliability diagram is based on prob(class 1) instead of
#    prob(most likely class).
# 2) Evenly-spaced probability grid that ensures 0 and 1 endpoints
#    are included.
# 3) If calc_acc is specified, return estimated accuarcy
#    for each item as well as its density (z).
def kernel_ece(probs, labels, classes, calc_acc=False, order=1,
               binary=False, verbose=True):
    from KDEpy import FFTKDE

    # X values for KDE evaluation points
    # Ensure that 0.0 and 1.0 are included
    # Grid has to be evenly spaced
    step = 0.0001
    x1 = np.arange(-0.6, 0.0, step)
    x2 = np.arange(0.0, 1.0, step)
    x3 = np.arange(1.0, 1.6, step)
    x = np.concatenate((x1, x2, x3))
    N = len(labels)

    kernel = 'triweight'

    # 1. Do KDE for accuracy using only correct predictions
    max_pred = np.argmax(probs, axis=1)
    if binary:
        if probs.shape[1] != 2:
            print('Error: kernel ECE with binary=True requires nx2 probs.')
            sys.exit(1)

        # Store the indicator of presence of class 1
        correct = [l == 1 for l in labels]
        # Store the probability of class 1 instead of the argmax prob
        max_prob = probs[:, 1]
    else:
        correct = [classes[p] == l for (p, l) in zip(max_pred, labels)]
        max_prob = np.max(probs, axis=1)
    probs_correct = max_prob[correct]
    if verbose:
        print('  %d accurate of %d preds' % (probs_correct.shape[0], N))
    # Specify a minimum value so kbw doesn't go to 0
    n_correct = np.sum(correct)
    kbw = max(1.06 * np.std(probs_correct) * (n_correct * 2) ** -0.2,
              1e-4)
    if verbose:
        print('  bandwidth based on std of %d accurate preds x %f: %.4f' %
              (int(n_correct), 1.06 * (n_correct * 2) ** -0.2, kbw))

    # Mirror the data about the domain boundary to avoid edge effects
    low_bound = 0.0
    up_bound = 1.0
    probs_correct_m = mirror_1d(probs_correct.reshape(-1, 1),
                                low_bound, up_bound)
    if verbose:
        print('  mirror changes range from %.2f-%.2f to %.2f-%.2f' %
              (np.min(probs_correct), np.max(probs_correct),
               np.min(probs_correct_m), np.max(probs_correct_m)))
    # Compute KDE using the bandwidth found, and twice as many grid points
    kde1 = FFTKDE(bw=kbw, kernel=kernel).fit(probs_correct_m)
    pp1 = kde1.evaluate(x)
    pp1[x < low_bound] = 0 # Set the KDE to zero outside of the domain
    pp1[x > up_bound] = 0  # Set the KDE to zero outside of the domain
    if verbose:
        print('  integral: %.2f -> %.2f' % (np.sum(pp1) / sum(pp1 > 0),
                                            np.sum(pp1 * 2) / sum(pp1 > 0)))
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1

    # 2. Do KDE for all predictions
    preds_m = mirror_1d(max_prob.reshape(-1, 1), low_bound, up_bound)
    if verbose:
        print('  mirror changes range from %.2f-%.2f to %.2f-%.2f' %
              (np.min(max_prob), np.max(max_prob),
               np.min(preds_m), np.max(preds_m)))
    # Compute KDE using the bandwidth found, and twice as many grid points
    kde2 = FFTKDE(bw=kbw, kernel=kernel).fit(preds_m)
    pp2 = kde2.evaluate(x)
    pp2[x < low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x > up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    # Avg prob of being correct
    perc = np.mean(correct)
    # Sum the differences between confidence and accuracy
    # to get the (empirical) ECE for this data set,
    # using the closest grid point (x) to each prediction (pr)
    closest = [np.abs(x - pr).argmin() for pr in max_prob]
    est_acc = [perc * pp1[c] / pp2[c] for c in closest]
    ece = np.sum(np.abs(max_prob - est_acc) ** order) / N

    if calc_acc:
        # Return accuracy and estimated mass at each test point
        z = [np.sum(pp2[c]) for c in closest]
        return ece, est_acc, z

    return ece


# Plot reliability diagram for one trial
def main(res_dir, dataset, n_cal, clf_type, prob_radius, seed):

    # Check arguments
    if not os.path.isdir(res_dir):
        print('Could not find directory %s' % res_dir)
        sys.exit(1)

    file_basename = 'n%d_clf%s_r%s' % (n_cal, clf_type, prob_radius)
    resfile = os.path.join(res_dir, 'res-%s-%s_seed%d.pkl' %
                           (dataset, file_basename, seed))

    # Read in results
    res = read_probs(resfile)
    if len(res['uncal']['test_probs']) == 0:
        print('Did not find any results for %s (%s, trial %d)'
              ' with %d samples and prob. radius %s in %s.' %
              (dataset, clf_type, seed, n_cal, prob_radius, res_dir))
        sys.exit(1)

    # Prune calib_methods to those with at least one test_probs entry
    calib_methods_use = [c for c in calib_methods
                         if 'test_probs' in res[c] and len(res[c]['test_probs']) > 0]

    binary = res['uncal']['test_probs'].shape[1] == 2

    # Compute the KDE ECE and accuracy if they aren't already there
    for c in calib_methods_use:
        # Compute the KDE ECE and accuracy if they aren't already there
        if 'k_acc' not in res[c] or 'k_ece' not in res[c]:
            y_test = read_test_labels(dataset, clf_type, seed)

            # Compute the KDE ECE and accuracy
            #classes = np.unique(y_test)
            classes = range(res[c]['test_probs'].shape[1])
            res[c]['k_ece'], res[c]['k_acc'], _ = \
                kernel_ece(res[c]['test_probs'], y_test, classes,
                           calc_acc=True, binary=binary, verbose=False)
        print('%15s: KECE %.4f' % (c, res[c]['k_ece']))

    # Generate reliability diagram
    pl.figure()
    pl.plot([0, 1], [0, 1], '--', color='gray')
    if binary:
        for c in calib_methods_use:
            pl.plot(res[c]['test_probs'][:, 1], res[c]['k_acc'], ls='',
                    label='%s' % (calib_info[c][0]),
                    #label='%s: %.3f' % (calib_info[c][0], res[c]['k_ece']),
                    marker=calib_info[c][2], color=calib_info[c][1])
        pl.xlabel('P(class 1)', fontsize=14)
        pl.ylabel('True P(class 1)', fontsize=14)
    else:
        for c in calib_methods_use:
            pl.plot(np.max(res[c]['test_probs'], axis=1),
                    res[c]['k_acc'], ls='',
                    #label='%s: %.3f' % (calib_info[c][0], res[c]['k_ece']),
                    label='%s' % (calib_info[c][0]),
                    marker=calib_info[c][2], color=calib_info[c][1])
        pl.xlabel('max P(y)', fontsize=14)
        pl.ylabel('Accuracy', fontsize=14)
    pl.legend()
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=12)
    outdir = os.path.dirname(resfile)
    outfn = resfile.replace('.pkl', '.pdf')
    outfn = os.path.join(outdir,
                         os.path.basename(outfn).replace('res-', 'probs-'))
    pl.savefig(outfn, bbox_inches='tight')
    print('Saved %s' % outfn)


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
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed (default: %(default)s)')

    args = parser.parse_args()
    main(**vars(args))
