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
    
def plot_lr_decay(hist):
    plt.figure()
    plt.plot(hist['iters_monitor'], hist['lr'], color=c_vals[0])
    plt.yscale('log')
    plt.show()

def eval_on_test_set(net, testData):
    db = net.evaluate(testData[:,:,:], debug=True)
    return db

def to_dataset(data, labels, device='cuda'):
    ''' Convert data and labels in numpy format to tensorDataset.'''
    from torch.utils.data import TensorDataset
    # Make labels same shape as data: [B] -> [B, T, 1].
    labels_full = labels.reshape(-1, 1, 1) # [Batch size, 1, 1]
    labels_full = labels_full.repeat(data.shape[1], 1) # [Batch Size, Tsteps, 1]
    data, labels_full = torch.from_numpy(data), torch.from_numpy(labels_full)
    data, labels_full = data.to(device), labels_full.to(device)
    return TensorDataset(data, labels_full)

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

def plot_pca(hs_pca, labels, zero_mat_pca = None):
    color_type = 'labels' # 'labels', 'words', 'seq_idx', 'labels_1', 'labels_2'
    
    spread = np.zeros((3, hs_pca.shape[1]))
    plt.figure(dpi=500)
    for i in range(3):
        spread[i] = np.var(hs_pca[:, :, i], 0)
        plt.plot(spread[i], color=c_vals[i], label=str(i))
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Variance Along PCA Dimension')
    plt.show()
    
    masks = [labels == 0, labels == 1, labels == 2]
    clusters = [np.mean(hs_pca[np.where(mask), :, :][0], 0) for mask in masks]
    plt.figure(figsize=(16,4))
    for d in range(3):
        plt.subplot(1,3,d+1)
        for i in range(len(masks)):
            plt.plot(clusters[i][:, d], c=c_vals[i])
        plt.xlabel('Timestep')
        plt.ylabel('Cluster Mean')
        plt.title(f'PC{d}')
    plt.tight_layout()
    plt.show()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
    
    pcxs = (0, 0, 1)
    pcys = (1, 2, 2)
    ax_pair_lims = (None, None, None)
    
    for ax, pcx, pcy, ax_pair_lim in zip((ax1, ax2, ax3), pcxs, pcys, ax_pair_lims):
        n_per_label = [0, 0, 0]
        for batch_idx in range(labels.shape[0]):
            if color_type == 'labels':
                color_labels_l = c_vals_l[labels[batch_idx]]
                color_labels = c_vals[labels[batch_idx]]
    
            ax.scatter(hs_pca[batch_idx, :, pcx], hs_pca[batch_idx, :, pcy], linewidth=0.0,
                        marker='.', color=color_labels_l, zorder=0, alpha=0.5)
            n_per_label[labels[batch_idx]] += 1
            ax.scatter(hs_pca[batch_idx, -1, pcx], hs_pca[batch_idx, -1, pcy], marker='.',
                       color=color_labels, zorder=5, alpha=0.5)
    
        # Plot coordinate axes for PCA.
        if zero_mat_pca is not None:
            for ro_idx in range(zero_mat_pca.shape[0]):
                ax.plot([zero_mat_pca[ro_idx, pcx], zero_mat_pca[ro_idx, pcx]], 
                        [zero_mat_pca[ro_idx, pcy], zero_mat_pca[ro_idx, pcy]],
                          color=c_vals_d[ro_idx], linewidth=3.0, zorder=10)
    
        # Some sample paths
        for batch_idx in range(0):
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

def input_vector_to_words(sample, toy_params):
    ''' Convert an input which is of size [tsteps, input size] to a word 
    consisting of indices corresponding to a dictionary of words (shape [tsteps]). '''
    
    # Measure which word in the word_list (vectors) has minimal distance for each timstep.
    words = np.array([toy_params['word_to_input_vector'][word] for word in toy_params['words']])
    dists = words.reshape(words.shape[0], 1, -1) - sample.reshape(1, *sample.shape)
    dists = np.linalg.norm(dists, axis=2) # Shape [# words, tsteps].
    return np.argmin(dists, axis=0) # Shape [tsteps].

def get_percents_sample(evid_vec, n_labels):
    ''' Get percent of sample that corresponds to each label (not including null/eos/etc).
    Sample should be an array of shape [tsteps] with indices corresponding to words. '''
    sums = np.zeros(n_labels)
    for l in range(n_labels):
        sums[l] = np.sum(evid_vec == l) 
    return sums / np.sum(sums)

def get_extreme_data(data, labels, toy_params, above_thresh = 0.0, below_thresh = 1.0, debug=False):
    ''' Determine which data samples are 'extreme', 
    in the sense that there is a lot or very little evidence for correct label. 
    Will check the percent of label in each sample and select ones above 
    above_thresh and below below_thresh.'''
    percent_label = np.zeros(data.shape[0]) # Percent of input that corresponds to label.
    for i in range(data.shape[0]):
        # Convert input sample from string of random binary vectors to indices for evidence.
        label, sample = labels[i], data[i] # sample is shape [T, word len].
        evid_vec = input_vector_to_words(sample, toy_params)
        percent_label[i] = get_percents_sample(evid_vec, toy_params['n_classes'])[label]
        
    if debug:
        plt.figure()
        plt.hist(percent_label, bins = 30, zorder=5)
        if above_thresh > 0:
            plt.axvline(above_thresh, color='red', zorder=10)
        if below_thresh < 1:
            plt.axvline(below_thresh, color='red', zorder=10)
        plt.show()
        
    matches = np.logical_and(percent_label > above_thresh, percent_label < below_thresh)
    return np.where(matches)

def cutoff_data(data, labels, toy_params, cutoff_ts = -1):
    ''' Cutoff samples so that at and after cutoff_ts all the input is just null (except for EOS).'''
    null_vec = toy_params['word_to_input_vector']['null']
    for i in range(data.shape[0]):
        sample = data[i]
        sample[cutoff_ts:-1] = null_vec
        
        # Relabel based on new input sample.
        evid_vec = input_vector_to_words(sample, toy_params)
        percents = get_percents_sample(evid_vec, toy_params['n_classes'])
        labels[i] = np.argmax(percents)
    return data, labels
        