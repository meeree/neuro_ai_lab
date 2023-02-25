# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:13:11 2023

@author: jhazelde
"""

import numpy as np 
import torch
from matplotlib import pyplot as plt
import os

# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',]
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]

def fit(net, folder_name, toy_params, net_params, train_params, trainData, validData, trainOutputMask, validOutputMask, override_data=False):    
    if not os.path.isdir('SAVES'):
        os.mkdir('SAVES')
        
    folder = 'SAVES/' + folder_name
    if os.path.isdir(folder):
        if not override_data:
            raise Exception(f'Error in fit: folder {folder} already exists. Overriding existing folder can be enabled by setting parameter override_data to True in fit.')
    else:
        os.mkdir(folder) # Make folder to store everything for run.
    
    # Save parameters to file. THIS IS IMPORTANT since the words are random binary vectors. If we run again without same seed
    # will have different random binary vectors!
    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
        
    for params, name in zip([toy_params, net_params, train_params], ['toy_params', 'net_params', 'train_params']):
        param_txt = json.dumps(params, cls=NumpyEncoder)
        with open(f'{folder}/{name}.json', 'w') as f:
            f.write(param_txt)
        
    net.verbose = True
    _ = net.fit('sequence', epochs=train_params['epochs'], 
                trainData=trainData, batchSize=train_params['batch_size'],
                validBatch=validData[:,:,:], learningRate=train_params['lr'],
                gradientClip=train_params['gradient_clip'],
                monitorFreq=train_params['monitorFreq'], 
                trainOutputMask=trainOutputMask, validOutputMask=validOutputMask, 
                scheduler=train_params.get('scheduler', None),
                filename = f'{folder}/save')
    return net

def plot_accuracy(hist):
    plt.figure(figsize = (8, 3))
    plt.subplot(1,2,1)
    ax1 = plt.gca()
    ax1.plot(hist['iters_monitor'], hist['train_loss'], color=c_vals[0], label='Train')
    ax1.plot(hist['iters_monitor'], hist['valid_loss'], color=c_vals[1], label='Test')
    ax1.plot(hist['iters_monitor'], hist['avg_valid_loss'], color=c_vals[2], label='Avg Test')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss (XE)')
    
    ax1.set_yscale('log')
    
    plt.subplot(1,2,2)
    ax1 = plt.gca()
    ax1.plot(hist['iters_monitor'], hist['train_acc'], color=c_vals[0], label='Train')
    ax1.plot(hist['iters_monitor'], hist['valid_acc'], color=c_vals[1], label='Test')
    ax1.plot(hist['iters_monitor'], hist['avg_valid_acc'], color=c_vals[2], label='Avg Test')
    max_acc = max(hist['avg_valid_acc'])
    ax1.axhline(max_acc, color='k', linestyle='dashed', c = 'red')
    ax1.text(0, max_acc*1.03, f'{max_acc:.3f}', c = 'red')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    plt.show()

def eval_on_test_set(net, testData):
    labels = np.array(testData[:,:,:][1][:, -1, 0].cpu())
    db = net.evaluate(testData[:,:,:], debug=True)
    return db

def sliding_window_states(len_filter, states):
    from scipy.signal import fftconvolve
    smoothing_filter = 1/len_filter * np.ones((1, len_filter, 1))
    convolve_mode = 'valid' # 'full’, ‘valid’, ‘same’
    
     # Adds initial state
    h0 = np.zeros((states.shape[0], 1, states.shape[2]))
    spk_hidden_fit = np.concatenate((h0, states), axis=1)

    hidden_conv = fftconvolve(smoothing_filter, spk_hidden_fit, axes=1, mode=convolve_mode)
    return hidden_conv

def PCA_dim(states, verbose = False):
    from sklearn.decomposition import PCA
    # Perform PCA projection on states and determine approximate dimensionality.
    n_components = 100
    pca_handler = PCA(n_components=n_components)
    
    pca_handler.fit(states.reshape((-1, states.shape[-1])))
    
    pca = np.zeros((states.shape[0], states.shape[1], n_components,))
    for batch_idx in range(states.shape[0]):
        pca[batch_idx] = pca_handler.transform(states[batch_idx])
    
    def participation_ratio_vector(C):
        """Computes the participation ratio of a vector of variances."""
        return np.sum(C) ** 2 / np.sum(C*C)
    
    variances = pca_handler.explained_variance_
    PR = participation_ratio_vector(variances)
    if verbose:
        print('PCA shape:', pca.shape)
        print('Variances:', ['{:.2f}'.format(var) for var in variances[:10]])
        print('PR: {:.2f}'.format(PR))
    return PR, pca, pca_handler

def plot_pca(hs_pca, labels):
    color_type = 'labels' # 'labels', 'words', 'seq_idx', 'labels_1', 'labels_2'
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
    
    pcxs = (0, 0, 1)
    pcys = (1, 2, 2)
    ax_pair_lims = (None, None, None)
    
    cmap = plt.cm.get_cmap('plasma')
    # phrase_len = hs_fit.shape[1]
    
    for ax, pcx, pcy, ax_pair_lim in zip((ax1, ax2, ax3), pcxs, pcys, ax_pair_lims):
        for batch_idx in range(labels.shape[0]):
            if color_type == 'labels':
                color_labels_l = c_vals_l[labels[batch_idx]]
                color_labels = c_vals[labels[batch_idx]]
    
            ax.scatter(hs_pca[batch_idx, :, pcx], hs_pca[batch_idx, :, pcy], linewidth=0.0,
                       marker='o', color=color_labels_l, zorder=0, alpha=0.5)
            ax.scatter(hs_pca[batch_idx, -1, pcx], hs_pca[batch_idx, -1, pcy], marker='o',
                       color=color_labels, zorder=5, alpha=0.5)
    
        # for ro_idx in range(net_params['n_outputs']):
        #     ax.plot([zero_matrix_pca[ro_idx, pcx], ro_matrix_pca[ro_idx, pcx]], 
        #             [zero_matrix_pca[ro_idx, pcy], ro_matrix_pca[ro_idx, pcy]],
        #              color=c_vals_d[ro_idx], linewidth=3.0, zorder=10)
    
        # Some sample paths
        for batch_idx in range(3):
            ax.plot(hs_pca[batch_idx, :, pcx], hs_pca[batch_idx, :, pcy], marker='.',
                    color=c_vals_d[labels[batch_idx]], zorder=5)
    
        ax.set_xlabel('PC{}'.format(pcx))
        ax.set_ylabel('PC{}'.format(pcy))
        # Sets square limits if desired
        if ax_pair_lim is not None:
            ax.set_xlim((-ax_pair_lim, ax_pair_lim))
            ax.set_ylim((-ax_pair_lim, ax_pair_lim))
    
    if color_type == 'labels':
        ax2.set_title('Hidden state PC plots (color by phrase label)')
    elif color_type == 'words':
        ax2.set_title('Hidden state PC plots (color by input word)')
    elif color_type == 'seq_idx':
        ax2.set_title('Hidden state PC plots (color by sequence location)')
    plt.show()
