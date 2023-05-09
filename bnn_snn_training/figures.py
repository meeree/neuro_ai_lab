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
from utils import fit, to_dataset, cutoff_data, get_extreme_data, eval_on_test_set, sliding_window_states, plot_lr_decay, PCA_dim, plot_pca, plot_accuracy, c_vals
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    'epochs': 1000,

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

def neural_properties_test(hh):
    ''' Test hodgkin-huxley model with different inputs, etc. '''
    hh.reset_state(1, False)
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
    hh.Iapp = 0.0
    K_out = torch.zeros((2000, 3))
    T_out = torch.zeros(2000)
    for i in range(2000):
        # hh.Iapp = np.sin(i / 2000.0 * 2 * np.pi * 10) * 1.5
        # if i > 700 or i > 1000:
        #     hh.Iapp = 5.0
        # if (i > 720 and i < 1000) or i > 1020:
        #     hh.Iapp = 0.0
        z = torch.ones((1, 1)) * 0
        # K_out[i, :] = hh.K[:, 0, 0].detach()
        _ = hh(z)
        V_out[i] = hh.V.item()
        T_out[i] = hh.T.item()
        
    # plt.plot(V_out, K_out[:, 1])
    # plt.show()
    
    plt.plot(V_out, c='black')
    ax2 = plt.gca().twinx()
    ax2.plot(T_out)
    plt.show()
    

# from networks import HH, HH_Fast, MorrisLecar
# hh =  MorrisLecar(1, 'cpu', phi = 2./30.0, V3 = 12., V4 = 17.)
# hh = HH_Fast(1, 'cpu')
# neural_properties_test(hh)
# exit()

def network_properties_test():
    from networks import VanillaBNN
    net_params['softmax'] = False
    net_params['noise_std'] = 0.0
    net_params['n_hidden'] = 128
    net = VanillaBNN(net_params, device='cuda').to('cuda')
    net.hidden_neurons.Iapp = 0.1
    
    # Initialize all weights to 1 and all biases to 0 
    for W in[net.W_ro, net.W_rec, net.W_inp]:
        W.weight.data.fill_(1e-1)
        W.bias.data.fill_(0.0)
        
    net.W_rec.weight.data.fill_(1e-3)
    
    # sparse = torch.rand(net.W_rec.weight.shape)
    # sparse = torch.where(sparse > 0.7, 1.0, 0.0)
    # net.W_rec.weight.data = sparse.cuda()

    tsteps = 4000
    inp = torch.ones((1, tsteps, net.n_inputs)).cuda() * 0
    batch = (inp, inp)
    out = net.evaluate(batch).detach().cpu().numpy()[0]
    
    raster = net.z1.detach().cpu().numpy()[0]
    raster = raster > -20
    plt.figure(dpi=500)
    plt.imshow(raster.T, aspect = 'auto', cmap = 'binary')
    plt.show()
    
    plt.figure()
    plt.plot(out[:, 0])
    plt.show()
    
# network_properties_test()
# exit()

def get_fit_lif(I0, freq, fudge = 0.99, debug = False):
    ''' Fit LIF model so that it fires at a given frequency given a fixed input current I0. '''
    from snntorch import Leaky
    beta = (1 - fudge) ** freq
    thresh = fudge * I0 / (1 - beta)
    lif = Leaky(beta, threshold = thresh)
    
    if debug:
        mems = torch.zeros(2000)
        for i in range(1,2000):
            _, mems[i] = lif(torch.ones(1)*I0, mems[i-1])
        plt.plot(mems.detach())
        plt.show()
    
    return lif

## Should give beta = 0.95, thresh = 1.0
# lif_95 = get_fit_lif(0.05005, 1.0 / 134.6717, 0.999, True) 
# print(lif_95.beta, lif_95.threshold) 
# lif_10x_slower = get_fit_lif(0.05005, 0.1 / 134.6717, 0.999, True) 
# print(lif_10x_slower.beta, lif_10x_slower.threshold) 


def fit_lif_hh_fi_curve(T=5000):
    ''' Fit an LIF neuron to an HH neuron using F-I curves. '''
    from networks import HH
    S = 500 # Number of samples for F-I curve (batch size).
    I = torch.linspace(0, 2, S) # Applied currents.
    hh = HH(1, 'cpu')
    hh.reset_state(S, randomize=False)
    hh.Iapp = 0.0
    Ts_hh = torch.zeros((S, T))
    for i in range(T):
        Ts_hh[:, i] = hh(I.unsqueeze(-1)).squeeze()

    def get_fi_curve(Ts):
        spikes = torch.logical_and(Ts[:, 1:] > 0.1, Ts[:, :-1] <= 0.1).float()
        return torch.mean(spikes, 1) * (1000 / hh.dt)
        
    fi_hh = get_fi_curve(Ts_hh)
    
    # Fit LIF to HH fi-curve.
    max_I0 = torch.max(I)
    max_freq = torch.max(fi_hh)
    max_freq /= (1000 / hh.dt) # Convert from Hz to timesteps.
    print(max_freq, 1/max_freq)
    
    # Try out a bunch of fudge values and use one that gives least L2 error.
    fudges = np.linspace(0.05, 0.1, 20)
    best_err, best_Ts, best_fudge, errs = 1e10, None, None, []
    for fudge in tqdm(fudges):
        lif = get_fit_lif(max_I0, max_freq, fudge)
        mem = lif.init_leaky()
        Ts_lif = Ts_hh.clone()
        for i in range(T):
            Ts_lif[:, i], mem = lif(I, mem)
            
        fi_lif = get_fi_curve(Ts_lif)
        err = torch.mean((fi_hh - fi_lif)**2).item()
        if err < best_err:
            best_err, best_Ts, best_fudge = err, Ts_lif, fudge
        errs.append(err)
    
    plt.figure()
    plt.plot(fudges, errs)
    plt.xscale('log')
    plt.xlabel('Fudge (%)')
    plt.ylabel('FI-Curve Error (L2)')
    plt.show()
    
    best_fi_lif = get_fi_curve(best_Ts)
    plt.figure()
    plt.plot(I, best_fi_lif)
    plt.plot(I, fi_hh)
    plt.legend(['LIF', 'HH'])
    plt.title('Fit FI-Curves')
    plt.xlabel('Input')
    plt.ylabel('Firing Frequency (Hz)')
    plt.show()
    
    plt.figure()
    plt.imshow(Ts_hh.detach() - best_Ts.detach(), aspect='auto', cmap='seismic', vmin=-1, vmax=1, interpolation = 'sinc')
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['LIF', 'None', 'HH'])
    plt.show()

    return max_I0, max_freq, best_fudge
    
# print(fit_lif_hh_fi_curve())

def load_network(net, toy_params, folder_full, specific_epoch = -1):
    import json, glob
    with open(folder_full + '/toy_params.json') as f:
        toy_params = json.load(f)    
        for key, val in toy_params.get('base_word_vals', {}).items():
            toy_params['base_word_vals'][key] = np.array(val)
        for key, val in toy_params.get('word_to_input_vector', {}).items():
            toy_params['word_to_input_vector'][key] = np.array(val)
        
        # Get index of best network and load it
        fl_names = [os.path.basename(fl) for fl in glob.glob(folder_full + '/save_*.pt')]
        fl_names.sort(key = lambda fl: int(fl[5:-3]))
        sd = torch.load(folder_full + '/' + fl_names[-1])
        hist = sd['hist']
        plot_accuracy(hist)
        plot_lr_decay(hist)
        
        if specific_epoch >= 0:
            net_idx = specific_epoch
        else:
            net_idx = np.argmax(hist['avg_valid_acc'])
            
        fl = folder_full + '/save_' + str(net_idx) + '.pt'
        sd = torch.load(fl)
        sd.pop('hist')
        net.load_state_dict(sd)
        net = net.to('cuda')
        
    return net, toy_params
       
def ablation_testing(folder = ''):
    global toy_params
    global trainData, validData, trainOutputMask, validOutputMask
    import utils

    folder_full = 'SAVES/' + folder    
    if not os.path.exists(folder_full):
        print(f"Folder doesn't exist: {folder_full}")
        return

    toy_params['phrase_length'] = 50
    net_params['filter_length'] = 100
    net_params['cuda'] = True
    net_params['use_snn'] = False
    net_params['n_per_step'] = 40
    toy_params['n_classes'] = 3
    net_params['n_outputs'] = toy_params['n_classes']
    net_params['softmax'] = True
    
    net = VanillaBNN(net_params, device='cuda').to('cuda')
    net.hidden_neurons.Iapp = 0.1
        
    net, toy_params = load_network(net, toy_params, folder_full)
    net.toy_params = toy_params
    
    # Evaluation and analysis of model.
    torch.manual_seed(0)
    test_set_size = 10
    testData, testOutputMask, _ = syn.generate_data(
            test_set_size, toy_params, net_params['n_outputs'], 
            verbose=False, auto_balance=False, device='cuda')
    
    labels = np.array(testData[:,:,:][1][:, -1, 0].cpu())    
    test_inputs = testData[:, :, :][0].cpu().numpy()
    
    evid0 = toy_params['word_to_input_vector']['evid0']
    null = toy_params['word_to_input_vector']['null']
    print(test_inputs.shape, evid0.shape)
    
    test_inputs[:, :-1, :] = null # Fill with no evidence
    cutoffs = np.linspace(0, test_inputs.shape[1], test_set_size+1)
    cutoffs = cutoffs[1:].astype(int)
    
    for i in range(test_set_size-1):
        # test_inputs[i, :cutoffs[i], :] = evid0
        test_inputs[i, cutoffs[i]:cutoffs[i+1]] = evid0
    
    labels.fill(0)
    testData = to_dataset(test_inputs, labels)
    
    word_grid = []
    for inp, label in zip(test_inputs, labels):
        word_grid.append(utils.input_vector_to_words(inp, toy_params))
    plt.imshow(word_grid, cmap = 'gist_rainbow', aspect='auto', interpolation='none')
    cbar = plt.colorbar(ticks=[0,1,2,3,4])
    cbar.ax.set_yticklabels(toy_params['words'])
    plt.show()

    db = eval_on_test_set(net, testData)
    accuracy = net.accuracy(testData[:,:,:], outputMask=testOutputMask)
    print('Accuracy: ', accuracy.item())
    spk_hidden = db['spk_hidden'].detach().cpu().numpy()
    spk_out = db['spk_out'].detach().cpu().numpy()
    
    out_conv = np.sum(spk_out[:, -net.filter_len:], 1)
    predicted = np.array(np.argmax(out_conv, 1))
        
    for case in range(test_set_size):
        for n_back in [40]:
            plt.figure(dpi=600)
            plt.subplot(3,1,1)
            scaled_back = net_params['n_per_step'] * n_back
            plt.plot(spk_out[case, -scaled_back:, :])
            plt.title(f'Cutoff = {cutoffs[case]}, label = {labels[0]}, predicted = {predicted[0]}')
            plt.legend([0,1,2])
            
            plt.subplot(3,1,2)
            weights = net.W_ro.weight[0, :].detach().cpu().numpy()
            plt.imshow(spk_hidden[case, -scaled_back:, :].T * weights.reshape(-1, 1), aspect='auto', cmap='seismic')
            plt.colorbar()
            
            plt.subplot(3,1,3)
            plt.plot(word_grid[case][-n_back:])
            plt.yticks([0, 1, 2, 3, 4], toy_params['words'])
            plt.show()        

# ablation_testing('TEST_50ev_NO_RANDOM_5e-4_CORRECT_LR')
# exit()

def gru_test():
    from networks import VanillaGRU
    global toy_params
    global trainData, validData, trainOutputMask, validOutputMask

    toy_params['phrase_length'] = 5
    train_params['epochs'] = 10
    net_params['filter_length'] = 100
    # net_params['random_start'] = 10
    net_params['cuda'] = True
    net_params['n_per_step'] = 30
    toy_params['n_classes'] = 3
    net_params['n_outputs'] = toy_params['n_classes']
    train_params['lr'] = 5e-4
    train_params['batch_size'] = 150
    # net_params['softmax'] = True
    device = 'cuda' if net_params['cuda'] else 'cpu' 
    
    net = VanillaGRU(net_params, device=device).to(device)
    
    trainData, trainOutputMask, toy_params = syn.generate_data(
        train_params['train_set_size'], toy_params, net_params['n_outputs'], 
        verbose=True, auto_balance=False, device=device)
    
    validData, validOutputMask, _ = syn.generate_data(
        train_params['valid_set_size'], toy_params, net_params['n_outputs'], 
        verbose=False, auto_balance=False, device=device)
        
    net = fit(net, 'GRU_TEST', toy_params, net_params, train_params, trainData, validData, trainOutputMask, validOutputMask, override_data=True) 
    plot_accuracy(net.hist)
    
    out = net.out.cpu().detach()
    for b in range(out.shape[0]):
        plt.plot(out[b])
        plt.show()
        
# gru_test()
# exit()
        
def analyze_network_discrete(folder = '', train = False, specific_epoch = -1):
    global toy_params
    global trainData, validData, trainOutputMask, validOutputMask
    import json
        
    def plot(net):          
        for i, W in enumerate([net.W_inp, net.W_rec, net.W_ro]):
            plt.subplot(1,3,1+i)
            plt.imshow(W.weight.data.cpu(), aspect='auto', interpolation='none', cmap='seismic')
            plt.xticks([])
            plt.yticks([])
        plt.show()

    # TODO: SET TRUNC, DELAY, ETC FOR LONG TIME FRAME TASK        
    toy_params['phrase_length'] = 50
    net_params['filter_length'] = 100
    # net_params['random_start'] = 10
    net_params['cuda'] = True
    net_params['use_snn'] = False
    net_params['n_per_step'] = 40
    toy_params['n_classes'] = 3
    net_params['n_outputs'] = toy_params['n_classes']
    # net_params['loss_fn'] = 'mse'
    train_params['lr'] = 5e-3
    train_params['batch_size'] = 50
    # train_params['scheduler'] = 'reducePlateau'
    net_params['softmax'] = False
    # net_params['n_hidden'] = 256
    
    # SNN setup
    # net_params['filter_length'] = 20
    # net_params['cuda'] = False 
    # net_params['use_snn'] = True                print(end)

    # net_params['n_per_step'] = 20
    # train_params['lr'] = 1e-3
    
    net = VanillaBNN(net_params, device='cuda').to('cuda')
    
    for param in net.params:
        param.set_params(100, 1.0)
    
    
    for W in[net.W_ro, net.W_rec, net.W_inp]:
        W.weight.data.fill_(1e-1)
        # W.bias.data.fill_(0.0)
        
        stdv = 1. / (W.weight.size(1) ** 0.5)
        W.weight.data.uniform_(-stdv + 1e-1, stdv + 1e-1)
        
    stdv = 1. / (net.W_rec.weight.size(1) ** 0.5)   
    net.W_rec.weight.data.fill_(1e-3)
    net.hidden_neurons.Iapp = 0.1
    net.W_rec.weight.data.uniform_(1e-3 - stdv, 1e-3 + stdv)
    
    # Sparse recurrent weights
    for i in range(net.W_rec.weight.shape[0]):
        for j in range(net.W_rec.weight.shape[1]):
            rand = torch.rand(())
            if rand > 0.8:
                net.W_rec.weight.data[i,j] = 1e-3

    override = False
    
    # net.trunc = 10
    init_W_rec = net.W_rec.weight.data.clone().cpu().detach().numpy()
    init_W_ro = net.W_ro.weight.data.clone().cpu().detach().numpy()
    init_W_inp = net.W_inp.weight.data.clone().cpu().detach().numpy()
    
    folder_full = 'SAVES/' + folder
    # for name, param in net.named_arameters():
    #     param.data *= 5
    
    # lif_10x_slower = get_fit_lif(0.05005, 0.1 / 134.6717, 0.999) 
    # net.hidden_neurons = lif_10x_slower
    if not override and len(folder) > 0 and os.path.exists(folder_full):
        net, toy_params = load_network(net, toy_params, folder_full)
        
    net.toy_params = toy_params

    if train:
        # Regenerate the train/validation sets and train. 
        trainData, trainOutputMask, toy_params = syn.generate_data(
            train_params['train_set_size'], toy_params, net_params['n_outputs'], 
            verbose=True, auto_balance=False, device=device)
        
        validData, validOutputMask, _ = syn.generate_data(
            train_params['valid_set_size'], toy_params, net_params['n_outputs'], 
            verbose=False, auto_balance=False, device=device)
            
        net = fit(net, 'SMOOTHGRAD_TESTS', toy_params, net_params, train_params, trainData, validData, trainOutputMask, validOutputMask, override_data=True) 
        #'TEST_50ev_NO_RANDOM_5e-4_CORRECT_LR'
    # Swap out LIF model instead of HH.
    # net.use_snn = True
    # net.hidden_neurons = get_fit_lif(2.0, 0.0038, 0.05) # Use parameters to fit LIF to HH
    
    # Evaluation and analysis of model.
    testData, testOutputMask, _ = syn.generate_data(
            test_set_size, toy_params, net_params['n_outputs'], 
            verbose=False, auto_balance=False, device='cuda')
    

    labels = np.array(testData[:,:,:][1][:, -1, 0].cpu())
    test_inputs = testData[:, :, :][0].cpu().numpy()
    db = eval_on_test_set(net, testData)
    accuracy = net.accuracy(testData[:,:,:], outputMask=testOutputMask)
    print('Accuracy: ', accuracy.item())
    spk_hidden = db['spk_hidden'].detach().cpu().numpy()
    spk_out = db['spk_out'].detach().cpu().numpy()
    
    W_rec = net.W_rec.weight.data.clone().cpu().detach().numpy()
    W_ro = net.W_ro.weight.data.clone().cpu().detach().numpy()
    W_inp = net.W_inp.weight.data.clone().cpu().detach().numpy()
    
    for fl, arr in zip(['spk_hidden.pt', 'spk_out.pt', 'labels_out.pt', 'W_ro.pt'], [spk_hidden, spk_out, labels, W_ro]):
        print(fl)
        with open(fl, 'wb') as f:
            np.save(f, arr)
    
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
    
    diff_rec = np.linalg.norm(W_rec - init_W_rec) / np.linalg.norm(init_W_rec)
    diff_ro = np.linalg.norm(W_ro - init_W_ro) / np.linalg.norm(init_W_ro)
    diff_inp = np.linalg.norm(W_inp - init_W_inp) / np.linalg.norm(init_W_inp)
    print(f'Relative weight change: W_rec {diff_rec:.2f}, W_ro {diff_ro:.2f}, W_inp {diff_inp:.2f}')
        
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

    for i in range(3):
        plt.subplot(3, 1, 1+i)
        plt.specgram(means_out[0, :, i], cmap="seismic")
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
            running_sum = np.cumsum(trace[-net.filter_len:]) / net.filter_len
            plt.subplot(3,3,3*i+2)
            plt.plot(running_sum, c = c_vals[j], linestyle = '--' if j != i else '-')
            
        if i == 0:
            plt.title('Running Output (in Window of Consideration)')
    plt.tight_layout()
    plt.show()
    
    smooth_means_out = sliding_window_states(net_params['n_per_step'] * 20, means_out)
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
    
    windows = np.linspace(220, 220, 1).astype(int)
    prs = []
    for w in tqdm(windows):
        hidden_conv = sliding_window_states(w, spk_hidden)
        prs.append(PCA_dim(hidden_conv)[0])
    plt.plot(windows, prs)
    plt.xlabel('Window Size')
    plt.ylabel('Estimated PCA Dimensionality')
    plt.show()
    
    min_w, _ = min(zip(windows, prs), key=lambda wp: wp[1])
         
    # TODO: ISOLATE PCA ON EXTREME CASES. ALSO TURN OFF NOISE!
    # TODO: WHAT IF WE REMOVE SOME TRANSIENT TIME?? no change :(
    
    # Isolate extreme cases in data
    # percents = np.linspace(0.0, 1.0, 0, endpoint=False)
    # prs = []
    # for percent in tqdm(percents):
    #     inds = get_extreme_data(test_inputs, labels, toy_params, 
    #                             percent, debug=True)
    #     spk_hidden, labels, test_inputs = spk_hidden[inds], labels[inds], test_inputs[inds]
    #     hidden_conv = sliding_window_states(min_w, spk_hidden)
    #     prs.append(PCA_dim(hidden_conv)[0])
        
    # plt.figure()
    # plt.plot(percents, prs)
    # plt.xlabel('Extremeness Percent Cutoff')
    # plt.ylabel('Estimated PCA Dimensionality')
    # plt.show()
    
    # Analyze PCA. We use more test data here. 
    # This requires iterated evaluates since the data is too big for the GPU in one pass.
    spk_hidden_total = np.zeros((10, *spk_hidden.shape))
    test_inputs_total = np.zeros((10, *test_inputs.shape))
    labels_total = np.zeros((10, *labels.shape)).astype(int)
       
    for cutoff in [toy_params['phrase_length']]:
        accuracy = 0.0
        for i in tqdm(range(10)):
            testData, testOutputMask, _ = syn.generate_data(
                test_set_size, toy_params, net_params['n_outputs'], 
                verbose=False, auto_balance=False, device='cuda')
            test_inputs, labels = testData[:, :, :][0].cpu().numpy(), testData[:, :, :][1][:, -1, 0].cpu().numpy()  
         
            test_inputs, labels = cutoff_data(test_inputs, labels, toy_params, cutoff)
            testData = to_dataset(test_inputs, labels)
            test_inputs_total[i], labels_total[i] = test_inputs, labels
            
            db = eval_on_test_set(net, testData)
            accuracy += net.accuracy(testData[:,:,:], outputMask=testOutputMask) / 10
            spk_hidden_total[i] = db['spk_hidden'].detach().cpu().numpy()
        print('Accuracy: ', accuracy.item(), 'Cutoff: ', cutoff)
          
        spk_hidden = spk_hidden_total.reshape((-1, *spk_hidden.shape[1:]))
        labels = labels_total.reshape(-1)
        test_inputs = test_inputs_total.reshape((-1, *test_inputs.shape[1:]))
        hidden_conv = sliding_window_states(min_w, spk_hidden)
                
        # Plot hidden states with imshow
        for spks in [spk_hidden, hidden_conv]:
            plt.figure(dpi=600)
            n_per_label = [0, 0, 0]
            vmin, vmax = np.min(spks), np.max(spks)
            for i in range(spk_hidden.shape[0]):
                label = labels[i]
                if n_per_label[label] < 5:
                    plt.subplot(5, 3, n_per_label[label] * 3 + label + 1)
                    plt.imshow(spks[i, :, :].T, vmin=vmin, vmax = vmax, aspect='auto', cmap='seismic')
                    n_per_label[label] += 1
            plt.show()
                
        # PCA analysis
        PR, hs_pca, pca_handler = PCA_dim(hidden_conv)
        print('Window size', min_w, 'PCA Dimensionality Estimate', PR)
        with open('pca.npy', 'wb') as f:
            np.save(f, hs_pca)
            
        with open('labels.npy', 'wb') as f:
            np.save(f, labels)
    
        zero_mat_pca = pca_handler.transform(np.zeros_like(W_ro))    
        plot_pca(hs_pca, labels, zero_mat_pca)
 
analyze_network_discrete('', True)
analyze_network_discrete('TEST_50ev_NO_RANDOM_5e-4_CORRECT_LR', False)
# analyze_network_discrete('CONTINUE_TWICE_SCHEDULE', False)
# analyze_network_discrete('MORRIS_LECAR_FIT_5e-4_RANDOM_LENGTH_CONTINUE_1e-4_BSIZE_4', False)
# analyze_network_discrete('HH_FAST_OUT_OF_SYNC_1e-3', True)
analyze_network_discrete('RANDOM_EL_MNH_HH_5e-4_KYLE_SETUP_NO_SCHEDULE', False)
exit()
analyze_network_discrete('ML_FIT_5e-4_RANDOM_LENGTH_FIXED_CONTINUE_1e-5', False)
exit()
analyze_network_discrete('MORRIS_LECAR_FIT_5e-4_CONTINUE_5e-5', False)
analyze_network_discrete('CONTINUE_FAST_HH', False)

# analyze_network_discrete('SNN_5e-4_MSE_LOSS_PLATEAU0.5', False)
analyze_network_discrete('CONTINUE_TWICE_SCHEDULE', False)
# analyze_network_discrete('TRAIN_5e-3_MSE_LOSS_PLATEAU0.5', False)
# analyze_network_discrete('TRAIN_5e-4_MSE_LOSS_PLATEAU0.5', False)
exit()

analyze_network_discrete('TRAIN_5e-3_PLATEAU_SCHEDULE0.5/', False)
analyze_network_discrete('TRAIN_1e-2_PLATEAU_SCHEDULE', False)
analyze_network_discrete('OLD_WORKING_NOISY_SHORT_40_1e-3/', True, specific_epoch = 2830)    
analyze_network_discrete('BNN_NO_RANDOM_NOISY_1e-2/')    