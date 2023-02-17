# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:55:53 2023

@author: jhazelde
"""

import torch
import sys
sys.path.append('../../mpn') # Replace with your own relative path.
import int_data as syn
from networks import VanillaBNN
from utils import fit, eval_on_test_set, sliding_window_states, PCA_dim, plot_pca, plot_accuracy, c_vals
from matplotlib import pyplot as plt

toy_params = {
    'data_type': 'retro_context_int', # int, context, retro_context, context_int, retro_context_int, retro_context_anti, cont_int, 
    
    'phrase_length': 50,
    'n_classes': 3,
    'input_type': 'binary',    # one_hot, binary, binary1-1
    'input_size': 50,          # defaults to length of words
    'include_eos': True,

    'stack_phrases': False,
    'n_stack': 1,
    'include_sos': False,
    'n_delay': 0, # Inserts delay words (>0: at end, <0: at beginning)
    'delay_word': '<delay>', # '<delay>' or 'null'

    'uniform_score': True, # Uniform distribution over scores=
}

net_params = {
    # Network Architecture
    'n_inputs': toy_params['input_size'],
    'n_hidden': 128,
    'n_outputs': toy_params['n_classes'],

    'snn_beta': 0.95,
    'filter_length': 50,

    'cuda': True,
}

train_params = {
    'epochs': 300,

    'batch_size': 64,
    'train_set_size': 3200,
    'valid_set_size': 100,
    'gradient_clip': 10,

    'monitorFreq': 5,
}

if net_params['cuda']:
    print('Using CUDA...')
    device = torch.device('cuda')
else:
    print('Using CPU...')
    device = torch.device('cpu')

# Generate datasets.
trainData, trainOutputMask, toy_params = syn.generate_data(
    train_params['train_set_size'], toy_params, net_params['n_outputs'], 
    verbose=True, auto_balance=False, device=device)
validData, validOutputMask, _ = syn.generate_data(
    train_params['valid_set_size'], toy_params, net_params['n_outputs'], 
    verbose=False, auto_balance=False, device=device)

test_set_size = 100
testData, testOutputMask, _ = syn.generate_data(
    test_set_size, toy_params, net_params['n_outputs'], 
    verbose=False, auto_balance=False, device=device)

import time
import numpy as np
import os

def sweep_2_params(param1, param2, p1_vals, p2_vals, n_trials = 1, force_eval = False):
    ''' Vary two parameters and evaluate fitting for each case '''
    global toy_params, net_params, train_params
    
    # Figure out which dict the parameters correspond to.
    dicts = []
    for p in param1, param2:
        for param_dict in [toy_params, net_params, train_params]:
            if p in param_dict:
                dicts.append(param_dict)
                break

    grid_name, acc_name = f'pca_grid_{param1}_{param2}.npy', f'acc_grid_{param1}_{param2}.npy'
    if os.path.exists(grid_name) and os.path.exists(acc_name) and not force_eval:
        with open(grid_name, 'rb') as f:
            grid = np.load(f)
        with open(acc_name, 'rb') as f:
            acc = np.load(f)
        return grid, acc
    
    grid = np.zeros((len(p1_vals), len(p2_vals)))
    acc = np.zeros_like(grid)
    for i1, p1 in enumerate(p1_vals):
        dicts[0][param1] = p1
        
        # ReGenerate datasets.
        torch.manual_seed(0)
        trainData, trainOutputMask, toy_params = syn.generate_data(
            train_params['train_set_size'], toy_params, net_params['n_outputs'], 
            verbose=True, auto_balance=False, device=device)
        validData, validOutputMask, _ = syn.generate_data(
            train_params['valid_set_size'], toy_params, net_params['n_outputs'], 
            verbose=False, auto_balance=False, device=device)
        
        for i2, p2 in enumerate(p2_vals):
            dicts[1][param2] = p2
            
            t0 = time.time()
            acc_avg = 0.0
            pca_avg = 0.0
            
            # Run multiple trials to filter out noise associated with randomization of network.
            for trial in range(n_trials):
                torch.manual_seed(trial)
                net = fit(toy_params, net_params, train_params, trainData, validData, trainOutputMask, validOutputMask)
                
                db = eval_on_test_set(net, testData)
                spk_hidden = db['spk_hidden'].detach().cpu().numpy()
                
                hidden_conv = sliding_window_states(net_params['filter_length'], spk_hidden)
                PR, hs_pca, pca_handler = PCA_dim(hidden_conv)
                pca_avg += PR / n_trials
                acc_avg += net.hist['valid_acc'][-1] / n_trials
                plot_pca(hs_pca, labels = np.array(testData[:,:,:][1][:, -1, 0].cpu()))
                plot_accuracy(net)
                
            grid[i1, i2] = pca_avg
            acc[i1, i2] = acc_avg
            print(f'{i1}, {i2}; Time Spent on Trials: {time.time() - t0:.2f}s; Accuracy: {acc_avg}')
            
        # Save intermediate results.
        with open(grid_name, 'wb') as f:
            np.save(f, grid)
            
        with open(acc_name, 'wb') as f:
            np.save(f, acc)
            
    return grid, acc
 
def single_window_plot():
    ''' Plot results after varying parameters with a fixed window size of 1 '''
    train_params['epochs'] = 5
    net_params['n_hidden'] = 500
    filter_lens = [1, 2, 5, 10]
    phrase_lens = np.linspace(2, 300, 5).astype(int)
    phrase_lens = [200]
    grid, acc = sweep_2_params('phrase_length', 'filter_length', phrase_lens, filter_lens, force_eval=True)
    grid, acc = grid[0, :], acc[0, :]
    plt.subplot(2,1,1)
    plt.plot(grid)
    plt.subplot(2,1,2)
    plt.plot(acc)
    plt.show()

# def set_ticks_imshow(xinc, yinc, xvals, yvals):
#     xt = list(range(0, len(xvals), 2))
#     yt = list(range(0, len(yvals), 2))
#     plt.xticks(xt, [xvals[i] for i in xt])
#     plt.yticks(yt, [yvals[i] for i in yt])  
  
# def plot_sweep():
#     timeframes, filter_percent, grid, acc = time_sweep()
#     plt.subplot(1,2,1)
#     plt.imshow(grid, aspect='auto', interpolation='sinc', cmap='gist_rainbow')
#     set_ticks_imshow(2, 2, filter_percent, timeframes)
#     plt.colorbar()
#     plt.ylabel('Timeframe')
#     plt.xlabel('Filter length (%)')
#     plt.title('PR Dimension')

#     plt.subplot(1,2,2)    
#     plt.imshow(acc, aspect='auto', interpolation='sinc', cmap='hot')
#     plt.colorbar()
#     set_ticks_imshow(2, 2, filter_percent, timeframes)
#     plt.title('Accuracy')
#     plt.xlabel('Filter length (%)')
#     plt.tight_layout()
#     plt.savefig('PR_snn.pdf')
#     plt.show()
    
#     plt.plot(timeframes, grid[:, 0])
#     plt.show()

def hh_properties_test():
    ''' Test hodgkin-huxley model with different inputs, etc. '''
    from networks import HH
    hh = HH(1, 'cpu')
    hh.reset_state(1)
    # hh.gk = 0 
    # I_{Na,t} model with oscillation (pg. 133 of Izhikevich DSN)
    # hh.gk = 0.0
    # hh.offs[2] = 62
    # hh.divs[2] = 6
    # hh.muls[2] = 0.2
    # hh.Ena = 60.0
    # hh.El = -70.0
    # hh.gl = 1.5
    # hh.gna = 15.0
    
    
    z = torch.zeros((1, 1))
    V_out = torch.zeros(2000)
    hh.Iapp = 0.5
    K_out = torch.zeros((2000, 3))
    for i in range(2000):
        # hh.Iapp = np.sin(i / 2000.0 * 2 * np.pi * 10) * 1.5
        # if i > 700 or i > 1000:
        #     hh.Iapp = 5.0
        # if (i > 720 and i < 1000) or i > 1020:
        #     hh.Iapp = 0.0
        z = torch.ones((1, 1)) * 0
        V_out[i] = hh.V.item()
        K_out[i, :] = hh.K[:, 0, 0].detach()
        _ = hh(z)
        
    plt.plot(V_out, K_out[:, 1])
    plt.show()
    
    plt.plot(V_out, c='black')
    ax2 = plt.gca().twinx()
    # ax2.plot(K_out)
    plt.show()
    
# hh_properties_test()
# exit()

    
def analyze_network_discrete(fl = '', toy_params_fl = ''):
    global toy_params
    import json
    if len(toy_params_fl) > 0:
        with open(toy_params_fl) as f:
            toy_params = json.load(f)    
            for key, val in toy_params.get('base_word_vals', {}).items():
                toy_params['base_word_vals'][key] = np.array(val)
            for key, val in toy_params.get('word_to_input_vector', {}).items():
                toy_params['word_to_input_vector'][key] = np.array(val)
        
    def plot(net):          
        for i, W in enumerate([net.W_inp, net.W_rec, net.W_ro]):
            plt.subplot(1,3,1+i)
            plt.imshow(W.weight.data.cpu(), aspect='auto', interpolation='none', cmap='seismic')
            plt.xticks([])
            plt.yticks([])
        plt.show()
      
    net_params['cuda'] = True
    net = VanillaBNN(net_params, device='cuda').to('cuda')
    init_W_rec = net.W_rec.weight.data.clone().cpu().detach().numpy()
    
    if len(fl) > 0:
        sd = torch.load(fl)
        sd.pop('hist')
        net.load_state_dict(sd)
        net = net.to('cuda')
        print(net.hh_hidden.gna)
    else:
        net = fit(net, 'TEST', toy_params, net_params, train_params, trainData, validData, trainOutputMask, validOutputMask)    
    
    testData, testOutputMask, _ = syn.generate_data(
            test_set_size, toy_params, net_params['n_outputs'], 
            verbose=False, auto_balance=False, device='cuda')
    labels = np.array(testData[:,:,:][1][:, -1, 0].cpu())
    db = eval_on_test_set(net, testData)
    # plot_accuracy(net)
    accuracy = net.accuracy(testData[:,:,:], outputMask=testOutputMask)
    print('Accuracy: ', accuracy.item())
    spk_hidden = db['spk_hidden'].detach().cpu().numpy()
    spk_out = db['spk_out'].detach().cpu().numpy()
    spk_out= spk_out[:, 10:, :]
    
    def svd_analysis():
        W_ro = net.W_ro.weight.data.detach().cpu().numpy()
        u, s, v = np.linalg.svd(W_ro)
        v = v.T
        v = v[:, :3] # Only care about first 3 columns of V for basis. 
        
        fig = plt.figure(figsize = (3,3))
        gs = fig.add_gridspec(3,3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[:, 2])
        ax1.imshow(u, cmap='seismic', interpolation='none')
        ax1.set_axis_off()
        ax1.set_title('U')
        ax2.imshow(s * np.identity(3), cmap='seismic', interpolation='none')
        ax2.set_axis_off()
        ax2.set_title('$\sigma$')
        ax3.imshow(v, cmap='seismic', interpolation='none', aspect='auto')
        ax3.set_axis_off()
        ax3.set_title('$V$')
        plt.show()
    svd_analysis()
    
    means_out = np.zeros((3, spk_out.shape[1], 3))
    n_hit = np.zeros(3)
    for i in range(len(labels)):
        label = labels[i]
        means_out[label] += spk_out[i, :, :]
        n_hit[label] += 1
    means_out /= n_hit.reshape((3, 1, 1)) 
    
    # Plot output weights
    plt.figure(dpi=300)
    plt.imshow(net.W_ro.weight.data.detach().cpu() > 0, aspect='auto')
    plt.show()

    animate_phase = False
    if animate_phase:
        # Animate phase space solution for label = 0.           
        from matplotlib.animation import FuncAnimation
        others = means_out[0, :, 1]
        correct = means_out[0, :, 0]
    
        fig = plt.figure(dpi=300)
        ln, = plt.plot(correct[:10], others[:10], 'o', zorder=10)
        ln2, = plt.plot([], [], 'o', alpha=0.1, zorder=1)
        plt.xlim(np.min(correct), np.max(correct))
        plt.ylim(np.min(others), np.max(others))
        plt.xlabel('Output 0 Over Time')
        plt.ylabel('Mean of Output 1 and 2 Over Time')
        plt.title('Label = 0 Mean Activity Phase Space')
        
        def update(t):
            ln2.set_data(correct[t-200:t], others[t-200:t])
            ln.set_data(correct[t:t+10], others[t:t+10])
            ln.set_color([t / means_out.shape[1], 0.0, 0.0])
            return ln,
        
        ani = FuncAnimation(
            fig, update,
            frames=range(0, means_out.shape[1], 10), blit=True)
        ani.save('phase_animation.gif', fps=60)
        plt.show()
    
    plt.figure(figsize=(15,9))
    for i in range(3):
        for j in range(3):
            plt.subplot(3,3,3*i+1)
            plt.plot(means_out[i, :, j], c = c_vals[j], linestyle = '--' if j != i else '-')
            plt.ylabel(f'Label = {i}')
            if i == 0:
                plt.title('Neuron Output')

            plt.subplot(3,3,3*i+3)            
            fft = np.fft.fft(means_out[i, :, j]) / means_out.shape[1]
            fft = fft[range(means_out.shape[1] // 2)] # Exclude sampling frequency.
            freq = np.arange(means_out.shape[1] // 2) 
            fft, freq = fft[1:50], freq[1:50]
            plt.plot(freq, fft.real, c = c_vals[j], linestyle = '--' if j != i else '-')
            if i == 0:
                plt.title('Fourier Domain')
                        
        for j in range(3):  
            trace = means_out[i, :, j]
            running_sum = np.zeros(net.filter_len)
            for k in range(-net.filter_len+1,0):
                running_sum[k+net.filter_len] = running_sum[k - 1] + trace[k]
            running_sum /= net.filter_len
            plt.subplot(3,3,3*i+2)
            plt.plot(running_sum, c = c_vals[j], linestyle = '--' if j != i else '-')
            
        if i == 0:
            plt.title('Running Output (in Window of Consideration)')
    plt.tight_layout()
    plt.show()
    
    smooth_means_out = sliding_window_states(200, means_out)
    plt.figure(figsize=(9,9))
    for i in range(3):
        for j in range(3):
            j2 = (j+1) % 3
            plt.subplot(3,3,3*i+j+1)
            for t in range(0, smooth_means_out.shape[1], 5):
                plt.plot(smooth_means_out[i, t:t+5, j], smooth_means_out[i, t:t+5, j2], c=[t / smooth_means_out.shape[1], 0, 0])
            plt.title(f'{j}, {j2}, label={i}')
    plt.tight_layout()
    plt.show()
        
        
    # Plot eigenvalue spectra before and after training.
    final_W_rec = net.W_rec.weight.data.clone().cpu().detach().numpy()
    
    plt.figure(dpi=600)
    for W_rec, color in zip([init_W_rec, final_W_rec], ['red', 'black']):
        ev, _ = np.linalg.eig(W_rec)
        x, y = [e.real for e in ev], [e.imag for e in ev]
        plt.scatter(x, y, c = color)
        W_rec = final_W_rec
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Eigenvalues')
    plt.show()
   
    spk_hidden = net.z1.detach().cpu().numpy()
    hidden_conv = sliding_window_states(200, spk_hidden)
    PR, hs_pca, pca_handler = PCA_dim(hidden_conv)
    print(PR)
    plot_pca(hs_pca, labels = np.array(testData[:,:,:][1][:, -1, 0].cpu()))
    

analyze_network_discrete()    
# analyze_network_discrete('SAVES/TRAIN_GNA_NOISY_SHORT_40_save_No_Delay_unregulated_3001.pt', 'SAVES/TRAIN_GNA_NOISY_SHORT_40_toy_params_No_Delay_unregulated.json')
# analyze_network_discrete('SAVES/REGULARIZED_NOISY_SHORT_40_save_No_Delay_unregulated_2524.pt', 'SAVES/REGULARIZED_NOISY_SHORT_40_toy_params_No_Delay_unregulated.json')
analyze_network_discrete('SAVES/NOISY_SHORT_40_save_No_Delay_unregulated_2830.pt', 'SAVES/NOISY_SHORT_40_toy_params_No_Delay_unregulated.json')