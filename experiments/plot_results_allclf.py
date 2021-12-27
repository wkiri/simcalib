#!/usr/bin/env python3
# Plot results for multiple classifiers on the same data set,
# with error bar for multiple trials.
#
# Kiri Wagstaff
# June 18, 2021

import sys
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as pl
from calib import calib_info, ds_name
from eval_calib import calib_methods
from plot_results import read_results


# Plot results from multiple classifiers and trials
def main(res_dir, dataset, n_cal, prob_radius, write_tables=False):

    # Check arguments
    if not os.path.isdir(res_dir):
        print('Could not find directory %s' % res_dir)
        sys.exit(1)

    if dataset in ['cifar10', 'cifar100']:
        clf_types = ['pretrained-resnet20', 'pretrained-resnet56-resnet20',
                     'pretrained-repvgg_a2-resnet20',
                     'pretrained-resnet56', 'pretrained-resnet20-resnet56',
                     'pretrained-repvgg_a2-resnet56',
                     'pretrained-repvgg_a2', 'pretrained-resnet20-repvgg_a2',
                     'pretrained-resnet56-repvgg_a2']
    elif dataset == 'imagenet':
        clf_types = ['pretrained-resnet18', 'pretrained-resnet152']
    elif dataset == 'msl':
        clf_types = ['pretrained-resnet34']
    elif dataset.startswith('starcraft-formations'):
        clf_types = ['pretrained-resnet18']
    else:
        clf_types = ['DT', 'GBT', 'NB', 'RF', 'SVM', 'RBFSVM']
    if (dataset in ['cifar10', 'cifar100', 'imagenet', 'letter',
                    'fashion-mnist', 'mnist10', 'ctg', 'mnist3', 'msl'] or
        dataset.startswith('starcraft-formations')):
        for m in ['platt', 'platt-swc']:
            if m in calib_methods:
                calib_methods.remove(m)
    else: # binary data set
        for m in ['ts', 'ts-swc']:
            if m in calib_methods:
                calib_methods.remove(m)

    # Read in results
    metrics = ['brier', 'acc']
    res = {}
    for clf_type in clf_types:
        file_basename = 'n%d_clf%s_r%s' % (n_cal, clf_type, prob_radius)
        resfilebase = os.path.join(res_dir, 'res-%s-%s_seed%%s.pkl' %
                                   (dataset, file_basename))
        if not os.path.exists(resfilebase % 0):
            print('Could not find %s' % (resfilebase % 0))
            continue
        res[clf_type] = read_results(metrics, resfilebase)
        # assuming the number of trials is the same
        # for all classifiers, methods, and seeds
        n_trials = len(res[clf_type]['uncal']['acc'])
        n_test_samples = len(res[clf_type]['uncal']['test_probs'][0])

        # Average across seeds
        for c in calib_methods:
            print('%15s: ' % c, end='')
            if c not in res[clf_type]:
                continue
            for m in metrics:
                res[clf_type][c]['avg_%s' % m] = np.mean(res[clf_type][c][m])
                res[clf_type][c]['std_%s' % m] = scipy.stats.sem(res[clf_type][c][m])
                print('%s, %s: %.4f (%.4f) [t=%d, ' %
                      (clf_type, m,
                       res[clf_type][c]['avg_%s' % m],
                       res[clf_type][c]['std_%s' % m],
                       len(res[clf_type][c][m])), end='')
                if len(res[clf_type][c]['test_probs']) > 0:
                    print('n=%d], ' % len(res[clf_type][c]['test_probs'][0]),
                          end='')
            if c == 'swc':
                # Avg across all seeds
                if 'test_HH' in res[clf_type][c]:
                    print('avg HH: %.4f, ' %
                          np.mean(np.array(res[clf_type][c]['test_HH'])),
                          end='')
            print()
        print()
    if len(res) == 0:
        print('Did not find any results for %s with %d samples and prob. radius %s in %s.' %
              (dataset, n_cal, prob_radius, res_dir))
        sys.exit(1)

    # Create bar plots and output .tex tables
    if dataset == 'ctg':
        n_cal = min(n_cal, 2126)

    # Create one table file for all metrics
    if write_tables:
        file_basename = 'n%d_r%s' % (n_cal, prob_radius)
        texfn = os.path.join(res_dir, 'table-%s-%s.tex' %
                             (dataset, file_basename))
        texf = open(texfn, 'w')
        texf.write('\\begin{table}[!ht]\n')
        texf.write('  \caption{Results for %s\n' % ds_name[dataset])
        texf.write('    ($n_{cal}=%d$, %d trials).\n' % (n_cal, n_trials))
        texf.write('    The best result(s) for each model (within'
                   ' 1 standard error) are in bold.')
        if clf_type.startswith('pretrained'):
            texf.write('    $\\rightarrow$ indicates the use of a'
                       ' different representation to calibrate'
                       ' the previous model.')
        texf.write('}\n')
        texf.write('  \label{tab:%s}\n' % (dataset))
        texf.write('  \small\n')
        # Figure out number of columns
        n_sc_methods = len([c for c in calib_methods if ('swc' in c or
                                                         'sba' in c)])
        texf.write('  \\begin{tabular}{l|c|%s|%s} \n' %
                   ('c' * (len(calib_methods) - n_sc_methods - 1),
                    'c' * n_sc_methods))
        
    for m in metrics:
        ylab = 'Brier score' if m == 'brier' else 'Accuracy'
        if write_tables:
            # Write table header
            texf.write('  \hline\n')
            texf.write('  \\multicolumn{%d}{c}{%s}\\\\ \hline\n' %
                       (len(calib_methods) + 1, ylab))
            texf.write('    Model')
            for c in calib_methods:
                texf.write(' & %s' % calib_info[c][3])
            texf.write(' \\\\ \hline\n')

        if dataset.startswith('cifar'):
            pl.figure(figsize=((6,2.5)))
        else:
            pl.figure(figsize=((5,2.5)))
        pl.clf()
        alg = []
        h = []
        # Reverse sort for Brier; normal sort for Accuracy
        for i, clf_type in enumerate(sorted(res, reverse=m == 'brier',
                                            key=lambda key:
                                            res[key]['uncal']['avg_%s' % m])):
            if write_tables:
                if clf_type.startswith('pretrained'):  # CNN models
                    # Add a separator
                    if i > 0 and clf_type.count('-') == 1:
                        texf.write('   \hline \n')
                    # Replace underscore with dash for LaTeX
                    clf_write = clf_type.replace('_', '\\_')
                    if clf_write.count('-') == 1:
                        texf.write('    \\footnotesize{%s}' %
                                   clf_write.split('-')[1])
                    else:
                        texf.write('    \\footnotesize{%s$\\rightarrow$}' %
                                   clf_write.split('-')[1])
                else:
                    texf.write('    \\footnotesize{%s}' % clf_type)

            n_clf = len(res[clf_type])
            if 'sim' in res[clf_type]:
                n_clf -= 1 # Don't leave room for 'sim' because it's a line
            width = 0.95 / n_clf
            if len(h) == 0:
                h = [0] * n_clf
            pos = 0
            best_method = 'uncal'
            best_res = res[clf_type][best_method]['avg_%s' % m]
            for j, c in enumerate(calib_methods):
                if c not in res[clf_type]:
                    continue
                if 'avg_%s' % m not in res[clf_type][c]:
                    continue
                avg_val = res[clf_type][c]['avg_%s' % m]
                if ((m == 'brier' and avg_val < best_res) or
                    (m == 'acc' and avg_val > best_res)):
                    best_res = avg_val
                    best_method = c
                h[j] = pl.bar([i - 0.4 + pos * width], avg_val,
                              yerr=res[clf_type][c]['std_%s' % m],
                              color=calib_info[c][1], width=width)
                pos += 1

            if write_tables:
                # Output to .tex file
                for j, c in enumerate(calib_methods):
                    if c not in res[clf_type]:
                        continue
                    if 'avg_%s' % m not in res[clf_type][c]:
                        continue
                    avg_val = res[clf_type][c]['avg_%s' % m]
                    std_val = res[clf_type][c]['std_%s' % m]
                    if (c == best_method or
                        (m == 'brier' and avg_val < best_res + std_val) or
                        (m == 'acc' and avg_val > best_res - std_val)):
                        texf.write(' & \\footnotesize{\\bf %.4f}'
                                   ' \\scriptsize{(%.3f)}' %
                                   (res[clf_type][c]['avg_%s' % m],
                                    res[clf_type][c]['std_%s' % m]))
                    else:
                        texf.write(' & \\footnotesize{%.4f}'
                                   ' \\scriptsize{(%.3f)}' %
                                   (res[clf_type][c]['avg_%s' % m],
                                    res[clf_type][c]['std_%s' % m]))
                texf.write(' \\\\\n')

            if dataset.startswith('cifar'):
                alg += ['%s\n(%.2f)' % ('$\\rightarrow$'.join(clf_type.split('-')[1:]),
                                        np.mean(np.array(res[clf_type]['swc']['test_HH'])))]
            else:
                alg += ['%s\n(%.2f)' % (clf_type, 
                                        np.mean(np.array(res[clf_type]['swc']['test_HH'])))]

        pl.xticks(range(len(res)), alg, fontsize=10.5
                  if dataset.startswith('cifar') else 12)
        pl.legend(h, [calib_info[c][0] for c in calib_methods],
                  fontsize=7,
                  #loc='lower right')
                  loc='lower right' if (m == 'acc' or
                                        dataset.startswith('cifar') or
                                        dataset == 'covid')
                  else 'best')
        pl.xlabel('Classifier')
        if m == 'acc':
            if dataset == 'cifar10':
                pl.ylim((0.90, 0.96))
            elif dataset == 'cifar100':
                pl.ylim((0.60, 0.80))
            elif dataset.startswith('starcraft-formations'):
                pl.ylim((0.90, 1.00))
            elif dataset.startswith('sc-perturb') or dataset == 'covid':
                pl.ylim((0.80, 1.00))
            elif dataset.startswith('mnist-'):
                pl.ylim((0.70, 1.00))
            elif dataset in ['fashion-mnist', 'mnist10', 'letter']:
                pl.ylim((0.50, 1.00))
        pl.ylabel(ylab, fontsize=14)

        pl.title('%s data set (n=%d)' % (dataset, n_cal))
        outfn = os.path.join(res_dir, 'summary-%s-%s-%s.pdf' % (dataset, m, file_basename))
        pl.savefig(outfn, bbox_inches='tight')
        print('Saved figure to %s' % outfn)

    if write_tables:
        texf.write('    \\hline\n')
        texf.write('  \\end{tabular}\n')
        texf.write('\\end{table}\n')
        texf.close()
        print('Saved table to %s' % texfn)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('res_dir',
                        help='Where to save plots (.pdf)')
    parser.add_argument('-d', '--dataset', default='moons',
                        help='Dataset (default: %(default)s)')
    parser.add_argument('-n', '--n_cal', type=int, default=1000,
                        help='Number of calibration items to use'
                        ' (default: %(default)s)')
    parser.add_argument('-r', '--prob_radius', type=float, default=0.1,
                        help='Radius in the probability simplex to define'
                        ' hidden heterogeneity neighborhood'
                        ' (default: %(default)s)')
    parser.add_argument('-t', '--write_tables', action='store_true',
                        default=False,
                        help='Write results in LaTeX form to .tex files'
                        ' (default: False)')

    args = parser.parse_args()
    main(**vars(args))
