#!/usr/bin/env python3
# Assess the amount of 'hidden heterogeneity'
# for a classifier and data set and plot it (2D).
#
# Kiri Wagstaff
# May 9, 2021

import os
import sys
import pickle
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from simcalib import calib_sim, platt_scaling, \
    temp_scaling_probs, hidden_hetero
from utils import get_dataset, get_classifier, brier
from calib import calib_info
# Max 3 classes
cm_bright = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# Assuming 2D data, plot it along with mesh-based estimates
# of posterior for class 1
def plot_heatmap(X, y, xx, yy, Z, title, vmin=0.0, vmax=1.0,
                 cm=pl.cm.RdBu):  # for red/blue membership

    alpha = 0.5
    pl.imshow(Z.reshape(xx.shape), interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              vmin=vmin, vmax=vmax,
              cmap=cm, aspect='auto', origin='lower', alpha=alpha)
    pl.colorbar()

    # Show the data
    pl.scatter(X[:, 0], X[:, 1], s=1, c=y, cmap=cm_bright)
    pl.xlim(np.min(xx), np.max(xx))
    pl.ylim(np.min(yy), np.max(yy))
    pl.title(title)


def main(res_dir, dataset, n_samples, clf_type, sim_method,
         prob_radius, cal_method, seed=0, plot_results=False):

    np.random.seed(seed)

    # Check arguments
    if not os.path.isdir(res_dir):
        print('Creating output directory %s' % res_dir)
        os.mkdir(res_dir)

    (X, y, _) = get_dataset(dataset, 10000, model_name=clf_type, seed=seed)

    # Split into training and test sets
    trte_size = {'letter': 2000,
                 'mnist10': 1000}
    te_size = trte_size[dataset] if dataset in trte_size else 500
    # Create a fixed set of te_size test items
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=te_size, random_state=seed)
    #train_test_split(X, y, test_size=.25, random_state=seed)

    # Create a fixed set of tr_size training items
    tr_size = trte_size[dataset] if dataset in trte_size else 500
    # Everything else (up to n_samples) is calibration
    X_cal, X_train, y_cal, y_train = \
        train_test_split(X_train, y_train, test_size=tr_size, random_state=seed)
    X_cal = X_cal[:n_samples]
    y_cal = y_cal[:n_samples]

    print('Train/val/test size:', len(y_train), len(y_cal), len(y_test))

    file_basename = 'n%d_%s_clf%s_sim%s_r%s_seed%d' % \
        (n_samples, cal_method, clf_type, sim_method, prob_radius, seed)
    resfile = os.path.join(res_dir, 'assess_HH_res-%s-%s.pkl' %
                           (dataset, file_basename))

    # -1. Read in previous results, if available
    if os.path.exists(resfile):
        with open(resfile, 'rb') as inf:
            res = pickle.load(inf)
    else:
        res = {}

    # 0. Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cal_orig = np.copy(X_cal)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)
    # Train the base classifier
    (clf, _) = get_classifier(clf_type)
    clf.fit(X_train, y_train)

    # 1. Evaluate uncalibrated probabilities
    if 'uncal' in res:
        cal_probs = res['uncal']['cal_probs']
        test_probs = res['uncal']['test_probs']
        classes = res['classes']
        bs = res['uncal']['brier']
    else:
        res['uncal'] = {}
        cal_probs = clf.predict_proba(X_cal)
        test_probs = clf.predict_proba(X_test)
        classes = clf.classes_
        bs = brier(test_probs, y_test, classes)

        # Save output for next time
        res['uncal']['cal_probs'] = cal_probs
        res['uncal']['test_probs'] = test_probs
        res['uncal']['brier'] = bs
        res['classes'] = classes
        with open(resfile, 'wb') as outf:
            pickle.dump(res, outf)

    print('Uncal. Brier: %.3f' % bs)

    if plot_results:
        # Plot heatmap of posteriors
        pl.figure(figsize=(15,2.5))

        # First generate the mesh
        # Thanks: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
        h = 0.1 # mesh resolution
        #h = 0.8 # mesh resolution
        #h = 1 # mesh resolution
        x_min, x_max = X_cal_orig[:,0].min(), X_cal_orig[:,0].max()
        y_min, y_max = X_cal_orig[:,1].min(), X_cal_orig[:,1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                             np.arange(y_min, y_max + h, h))
        mesh_data = np.c_[xx.ravel(), yy.ravel()]

        # Generate (or read back in) probs for the mesh
        mesh_data = scaler.transform(mesh_data)
        mesh_probs = res['uncal']['mesh_probs'] if 'mesh_probs' in res['uncal'] \
            else clf.predict_proba(mesh_data)
        res['uncal']['mesh_probs'] = mesh_probs
        print('mesh (n=%d): %f - %f' % (len(mesh_probs),
                                        np.min(mesh_probs[:, 1]),
                                        np.max(mesh_probs[:, 1])))

        pl.subplot(1, 4, 1)
        # Plot probability of class 1
        plot_heatmap(X_cal_orig, y_cal, xx, yy, mesh_probs[:, 1],
                     'Uncalibrated (%s), Brier %.3f' % (clf_type, bs))
        # Plot prob of most-confident class
        #plot_heatmap(X_cal, y_cal, xx, yy, np.max(mesh_probs, axis=1),
        #             'Uncalibrated (%s), Brier %.3f' % (clf_type, bs),
        #             cm='plasma')

    # 2. Apply Platt/TS
    # Although we can only plot 2D data sets,
    # we might be running this in non-plotting mode just for the stats,
    # so support multiclass options as well.
    if len(classes) == 2:
        cname = 'platt'
        new_test_probs = res[cname]['test_probs'] if cname in res \
            else platt_scaling(cal_probs, y_cal, test_probs)
        if plot_results:
            new_mesh_probs = res[cname]['mesh_probs'] if (cname in res
                                                          and 'mesh_probs' in res[cname])\
                else platt_scaling(cal_probs, y_cal, mesh_probs)
    else:
        cname = 'ts'
        # temp_scaling_probs returns T and the probs; select the latter
        new_test_probs = res[cname]['test_probs'] if cname in res \
            else temp_scaling_probs(cal_probs, y_cal,
                                    len(classes), test_probs)[1]
        if plot_results:
            new_mesh_probs = res[cname]['mesh_probs'] if cname in res \
                else temp_scaling_probs(cal_probs, y_cal,
                                        len(classes), mesh_probs)[1]

    bs = res[cname]['brier'] if cname in res \
        else brier(new_test_probs, y_test, classes)
    print('%s Brier: %.3f' % (cname, bs))

    # Save results
    if cname not in res:
        res[cname] = {}
    res[cname]['brier'] = bs
    res[cname]['test_probs'] = new_test_probs
    if plot_results:
        res[cname]['mesh_probs'] = new_mesh_probs
    with open(resfile, 'wb') as outf:
        pickle.dump(res, outf)

    if plot_results:
        print('mesh: %f - %f' % (np.min(new_mesh_probs[:, 1]),
                                 np.max(new_mesh_probs[:, 1])))
        # Plot heatmap of posteriors for test set
        pl.subplot(1, 4, 2)
        # Plot probability of class 1
        plot_heatmap(X_cal_orig, y_cal, xx, yy, new_mesh_probs[:, 1],
                     '%s, Brier %.3f' % (calib_info[cname][0], bs))
        # Plot prob of most-confident class
        #plot_heatmap(X_cal, y_cal, xx, yy, np.max(new_mesh_probs, axis=1),
        #             '%s calib., Brier %.3f' % (cname, bs),
        #             cm='plasma')

    print('%s, n=%d' % (dataset, n_samples))

    # 3. Use the calibration method on the test set
    if cal_method not in res:
        res[cal_method] = {}

    if 'test_probs' in res[cal_method]:
        test_probs = res[cal_method]['test_probs']
    else:
        if cal_method == 'swc':
            scaler = StandardScaler()
            scaler.fit(cal_probs)
            cal_probs_scaled = scaler.transform(cal_probs)
            test_probs_scaled = scaler.transform(test_probs)

            test_probs, _ = \
                calib_sim(np.hstack((X_cal, cal_probs_scaled)), y_cal,
                          np.hstack((X_test, test_probs_scaled)), test_probs,
                          nn=-1, hh=None, sim_method='RFprox')
        else:
            print('Error: unsupported cal method %s' % cal_method)
            sys.exit(1)
        res[cal_method]['test_probs'] = test_probs

    cal_bs = res[cal_method]['brier'] if 'brier' in res[cal_method] \
        else brier(test_probs, y_test, classes)
    res[cal_method]['brier'] = cal_bs

    print('%s Brier: %.3f' % (cal_method, cal_bs))

    with open(resfile, 'wb') as outf:
        pickle.dump(res, outf)

    if plot_results:
        # Compute HH for visualization
        # At this point, cal_method is in res but may not be complete
        mesh_hh = res['mesh_hh'] if 'mesh_hh' in res \
            else hidden_hetero(X_cal, y_cal, cal_probs,
                               mesh_probs, r=prob_radius)[0]
        res['mesh_hh'] = mesh_hh
        with open(resfile, 'wb') as outf:
            pickle.dump(res, outf)

        # Plot HH results
        pl.subplot(1, 4, 3)
        plot_heatmap(X_cal_orig, y_cal, xx, yy, mesh_hh,
                     'Hidden heterogeneity', vmin=0,
                     #vmax=0.05,
                     vmax=max(max(mesh_hh), 1),
                     cm='jet')

        # Use cal_method on the mesh
        if 'mesh_probs' in res[cal_method]:
            cal_mesh_probs = res[cal_method]['mesh_probs']
        else:
            if cal_method == 'swc':
                scaler = StandardScaler()
                scaler.fit(cal_probs)
                cal_probs_scaled = scaler.transform(cal_probs)
                mesh_probs_scaled = scaler.transform(mesh_probs)

                cal_mesh_probs, _ = \
                    calib_sim(np.hstack((X_cal, cal_probs_scaled)), y_cal,
                              np.hstack((mesh_data, mesh_probs_scaled)), mesh_probs,
                              nn=-1, hh=mesh_hh, sim_method='RFprox')
            else:
                print('Error: unsupported cal method %s' % cal_method)
                sys.exit(1)

            res[cal_method]['mesh_probs'] = cal_mesh_probs

            with open(resfile, 'wb') as outf:
                print(' Saving results to %s' % resfile)
                pickle.dump(res, outf)

        # Plot SWC results
        pl.subplot(1, 4, 4)
        # Plot probability of class 1
        plot_heatmap(X_cal_orig, y_cal, xx, yy, cal_mesh_probs[:, 1],
                     '%s, Brier %.3f' % (calib_info[cal_method][0], cal_bs))
        # Plot prob of most-confident class
        #plot_heatmap(X_cal, y_cal, xx, yy, np.max(cal_mesh_probs, axis=1),
        #             '%s, Brier %.3f' % (cal_method, cal_bs),
        #             cm='plasma')

    if plot_results:
        outfn = os.path.join(res_dir, 'probs_map-%s-%s.pdf' %
                             (dataset, file_basename))
        pl.savefig(outfn, bbox_inches='tight')
        print('Saved figure to %s' % outfn)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('res_dir',
                        help='Where to save plots (.pdf)')
    parser.add_argument('-d', '--dataset', default='moons',
                        help='Dataset (default: %(default)s)')
    parser.add_argument('-n', '--n_samples', type=int, default=1000,
                        help='Number of items to generate (default: 1000)')
    parser.add_argument('-c', '--clf_type', default='DT',
                        choices=['DT', 'MC', 'RF', 'SVM', 'NB', 'GBT'],
                        help='Classifier type (default: %(default)s)')
    parser.add_argument('-r', '--prob_radius', type=float, default=0.1,
                        help='Radius in the probability simplex to define hidden heterogeneity'
                        ' neighborhood (default: %(default)s)')
    parser.add_argument('-m', '--sim_method', default='RFprox',
                        choices=['cosine', 'rbf', 'RFprox'],
                        help='Similarity measure (default: %(default)s)')
    parser.add_argument('-l', '--cal_method', default='swc',
                        help='Calibration method to plot (default: %(default)s)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('-p', '--plot_results', action='store_true',
                        default=False, help='Generate plots')
    args = parser.parse_args()
    main(**vars(args))
