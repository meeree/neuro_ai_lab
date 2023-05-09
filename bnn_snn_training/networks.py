# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:52:34 2023

@author: jhazelde
"""

# imports
import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F
import sys
sys.path.append('mpn') 
from net_utils import StatefulBase, xe_classifier_accuracy
from utils import plot_accuracy
import os

class HH(torch.jit.ScriptModule):
    def __init__(self, L, device):
        super().__init__()
        self.L = L
        
        # State varaibles. V is the voltage B, [L], B batch size, L layer count, 
        # K is the gating variables [3, B, L], T is the output [B, L].
        self.V = self.K = self.T = torch.ones(()) 
        
        # HH parameters. These can be tweaked after initializing the HH model.
        self.gna = 40.0; self.gk = 35.0; self.gl = 0.3;
        self.Ena = 55.0; self.Ek = -77.0; self.El = -65.0; 
        self.Iapp = 0.5; self.Vt = -3.0; self.Kp = 8.0; 
        self.dt = 0.1;
        self.device = device
        
        # Offsets and divisors for gating variable update rates.
        # For their uses, see below.
        self.offs = torch.tensor([35.0, -25.0, 90.0, 35.0, -25.0, 34.0]).view(-1, 1, 1).to(self.device)
        self.divs = torch.tensor([-9.0, -9.0, -12.0, 9.0, 9.0, 12.0]).view(-1, 1, 1).to(self.device)
        self.muls = torch.tensor([0.182, 0.02, 0.0, -0.124, -0.002, 0.0]).view(-1, 1, 1).to(self.device)
        
        if os.path.exists(f'EL_random_{L}.pt'):
            self.El = torch.load(f'EL_random_{L}.pt').to(self.device)
        else:
            self.El = torch.ones((1, L)).to(self.device) * self.El
            self.El = torch.normal(self.El, 1.5)
            torch.save(self.El, f'EL_random_{L}.pt')
            
        # self.Iapp = torch.ones((1, L)) * self.Iapp
        # self.Iapp = nn.Parameter(self.Iapp).to(self.device)

         # Add some random noise to each nueron mul
        # self.muls = self.muls.reshape((-1, 1, 1)).repeat(1, 1, L)
        # self.muls = self.muls + torch.rand(self.muls.shape).to(self.device) * 5e-3
        
    def reset_state(self, B):
        self.V = torch.ones((B, self.L)).to(self.device) * self.El
        self.T = torch.zeros_like(self.V).to(self.device)
                
        # Gating variables
        if os.path.exists(f'gating_variables_random_{B}_{self.L}.pt'):
            self.K = torch.load(f'gating_variables_random_{B}_{self.L}.pt').to(self.device)
        else:
            self.K = torch.normal(torch.zeros((3, B, self.L)) + 0.5, 0.1).to(self.device)
            torch.save(self.K, f'gating_variables_random_{B}_{self.L}.pt')
        # self.y = torch.zeros_like(self.V).to(self.device)

    @torch.jit.script_method
    def forward(self, z):
        ''' Pass input through HH for one timestep. z should be shape [batch size, neuron count]'''
        ''' Input k is the NEXT timestep we are considering. Should start at 1. '''
        B = z.shape[0] # Batch size
        
        aK = torch.zeros((3, B, self.L)).to(self.device)
        bK = torch.zeros((3, B, self.L)).to(self.device)
        # Optimization: concatenate all gating variables in one big tensor since their updates are very similar.
        m = self.K[0, :, :]
        n = self.K[1, :, :]
        h = self.K[2, :, :]

        # Calculate V intermediate channel quantities.
        pow1 = self.gna * (m ** 3) * h
        pow2 = self.gk * n ** 4
        G_scaled = (self.dt / 2) * (pow1 + pow2 + self.gl)
                                    # + self.y * self.gs * self.Es)
        E = pow1 * self.Ena + pow2 * self.Ek + self.gl * self.El
        # + self.y * self.gs * self.Es

        # V update.
        self.V = (self.dt * (E + self.Iapp + z) +  (1 - G_scaled) * self.V.clone()) / (1 + G_scaled)

        # Calculate gating variable intermediate rate quantities.
        v_off = self.V + self.offs
        EXP = torch.exp(v_off / self.divs) # Optimization: do all exponentials at once. I've found this to shave ~20% time off.           
        scaled_frac = self.muls / (1 - EXP) # Optimization: compute these terms in a batch. MOST HAVE FORM: k * (v + off) / (1 - exp).

        aK[:2] = v_off[:2] * scaled_frac[:2]
        aK[2] = 0.25 * EXP[2]
        bK[:2] = v_off[3:5] * scaled_frac[3:5]          
        bK[2] = 0.25 * EXP[5]
        aK = torch.nan_to_num(aK, 0.0) # Handle division by zero in EXP.
        bK = torch.nan_to_num(bK, 0.0)

        # Gating Variable update.
        sum_scaled = self.dt/2 * (aK+bK)
        self.K = (self.dt * aK + (1 - sum_scaled) * self.K.clone()) / (1 + sum_scaled) # Note similarity with V update above
        
        # sum_scaled2 = self.dt / 2 * (self.a_d * z + self.a_r)
        # self.y = (self.dt * self.a_d * z + (1 - sum_scaled2) * self.y.clone()) / (1 + sum_scaled2)
        
        self.T = torch.sigmoid((self.V - self.Vt) / self.Kp)
        return self.T
    
class HH_Fast(torch.jit.ScriptModule):
    def __init__(self, L, device):
        super().__init__()
        self.L = L
        
        # State varaibles. V is the voltage B, [L], B batch size, L layer count, 
        # K is the gating variables [3, B, L], T is the output [B, L].
        self.V = self.K = self.T = torch.ones(()) 
        
        # HH parameters. These can be tweaked after initializing the HH model.
        self.gna = 120.0; self.gk = 36.0; self.gl = 0.3;
        self.Ena = 55.0; self.Ek = -77.0; self.El = -65;
        self.Iapp = 0.0; self.Vt = -3.0; self.Kp = 8.0; 
        self.dt = 0.1;
        self.device = device
        
        self.phase_shift = torch.linspace(0.0, 10.0, L).reshape(1, L).to(self.device)
        
        # Offsets and divisors for gating variable update rates.
        # For their uses, see below.
        self.inf_V12 = torch.tensor([-40., -53., -62.]).view(-1, 1, 1).to(self.device) 
        self.inf_k = torch.tensor([15., 15., -7.]).view(-1, 1, 1).to(self.device)
        self.tau_base = torch.tensor([0.04, 1.1, 1.2]).view(-1, 1, 1).to(self.device)
        self.tau_amp = torch.tensor([0.46, 4.7, 7.4]).view(-1, 1, 1).to(self.device)
        self.tau_Vmax = torch.tensor([-38., -79., -67.]).view(-1, 1, 1).to(self.device)
        self.tau_var = torch.tensor([30., 50., 20.]).view(-1, 1, 1).to(self.device) # STD NOT VARIANCE

        # DONT DO THIS STEP YOURSELF ! 
        self.tau_var = self.tau_var ** 2 # Variance, not STDDEV!
        
    def reset_state(self, B, randomize=True):
        self.V = torch.ones((B, self.L)).to(self.device) * -65.0
        if randomize:
            self.V += torch.normal(torch.zeros_like(self.V), 5.0) # Add some noise to initial conditions.
        self.T = torch.zeros_like(self.V).to(self.device)
                
        # Gating variables
        self.K = torch.zeros((3, B, self.L)).to(self.device)

    @torch.jit.script_method
    def forward(self, z):
        ''' Pass input through HH for one timestep. z should be shape [batch size, neuron count]'''
        ''' Input k is the NEXT timestep we are considering. Should start at 1. '''
        # Optimization: concatenate all gating variables in one big tensor since their updates are very similar.
        m = self.K[0, :, :]
        n = self.K[1, :, :]
        h = self.K[2, :, :]

        # Calculate V intermediate channel quantities.
        pow1 = self.gna * (m ** 3) * h
        pow2 = self.gk * n ** 4
        G_scaled = (self.dt / 2) * (pow1 + pow2 + self.gl)
        E = pow1 * self.Ena + pow2 * self.Ek + self.gl * self.El

        # V update.
        I_in = self.Iapp + z + self.phase_shift
        self.V = (self.dt * (E + I_in) +  (1 - G_scaled) * self.V.clone()) / (1 + G_scaled)

        # Calculate gating variable intermediate rate quantities.
        inf = torch.sigmoid((self.V - self.inf_V12) / self.inf_k)
        tau = self.tau_base + self.tau_amp * torch.exp(-(self.tau_Vmax - self.V) / self.tau_var)

        # Gating Variable update.
        self.K = (inf * self.dt + (tau - self.dt/2) * self.K) / (tau + self.dt/2)
        self.T = torch.sigmoid((self.V - self.Vt) / self.Kp)
        return self.T
    
class MorrisLecar(torch.jit.ScriptModule):
    def __init__(self, L, device, phi = 0.04, V1 = -1.2, V2 = 18., V3 = 2., V4 = 30., C = 20.):
        super().__init__()
        self.L = L
        
        # State varaibles. V is the voltage B, [L], B batch size, L layer count, 
        # K is the gating variables [3, B, L], T is the output [B, L].
        self.V = self.w = self.T = torch.ones(()) 
        
        # ML Parameters.
        self.phi = phi; self.V1 = V1; self.V2 = V2; self.V3 = V3; self.V4 = V4;
        self.C = C
        self.dt = 0.1; self.Vt = -3.0; self.Kp = 8.0; 
        self.device = device
        self.Iapp = 0.0
        
    def reset_state(self, B, randomize=True):
        self.V = torch.ones((B, self.L)).to(self.device) * -20.0 
        if randomize:
            self.V += torch.normal(torch.zeros_like(self.V), 5.0) # Add some noise to initial conditions.
        self.T = torch.zeros_like(self.V).to(self.device)
                
        # Gating variables
        self.w = torch.ones((B, self.L)).to(self.device) * 0.014173

    @torch.jit.script_method
    def forward(self, z):
        minf = 0.5 * (1 + torch.tanh((self.V - self.V1) / self.V2))
        winf = 0.5 * (1 + torch.tanh((self.V - self.V3) / self.V4))
        tauw = 1. / (self.phi * torch.cosh((self.V - self.V3) / (2 * self.V4)))
        self.w = (winf * self.dt + (tauw - self.dt/2) * self.w) / (tauw + self.dt/2)
        
        G = minf * 4. + self.w * 8. + 2.
        E = minf * 4. * 120. + self.w * 8. * -84. + 2. * -60.
        inp = self.Iapp + z
        inp = inp * 36.75 + 39.9 # This makes HH and ML range of inputs consistent.
        inp = inp / self.C
        G, E = G / self.C, E / self.C
        self.V = (self.V * (1 - self.dt/2 * G) + self.dt * (E + inp)) / (1 + self.dt/2 * G)
        self.T = torch.sigmoid((self.V - self.Vt) / self.Kp)
        return self.T
   
class WeightSampler(torch.jit.ScriptModule):
    '''Class supporting adding S offsets to weight matrix'''
    def __init__(self, input_dim, output_dim, device='cpu', trainable = True):
        super().__init__()
        self.weight = nn.Linear(output_dim, input_dim, bias=False).weight.data # note transpose
        self.weight = self.weight.to(device)
        if trainable:
            self.weight = nn.Parameter(self.weight) # Make a trainable param.
        else:
            self.weight.requires_grad = False
            
        self.noise = torch.zeros(()).to(device)
        self.W_noisy = torch.zeros(()).to(device)
        self.set_params(1, 0.0)
        self.noisify()
        self.active = False
        
    def set_params(self, S, stddev):
        self.S = S
        self.stddev = stddev
    
    def noisify(self):
        # Copies of weight for noisy sampling over batches and S samples.
        self.W_noisy = self.weight.data
        self.W_noisy = self.W_noisy
        self.W_noisy = self.W_noisy.repeat(self.S, 1, 1, 1) # [S, 1, IN, OUT], 1 is for batch dim.

        self.noise = torch.normal(0.0, self.stddev * torch.ones_like(self.W_noisy))
        self.noise[0] *= 0.0 # No noise added to first sample. This is needed for gradient calculation!
        
        # Make sampling sparse
        # mask = torch.rand_like(self.noise) > 0.8
        # self.noise = self.noise * mask 
        
        self.W_noisy += self.noise
        
    def set_active(self, active):
        self.active = active
        
    def compute_grad(self, losses):
        # Sum 1/S * loss(v_i) * (v_i - mu_i) / sigma_i^2 
        smpls = self.noise.reshape(self.S, -1) # Reshape for easier multiplication
        grad = torch.mean(losses.reshape(self.S, 1) * smpls, 0) / (self.stddev ** 2)
        return grad.reshape(self.weight.shape)

    def forward(self, x):
        if not self.active:
            return torch.matmul(x, self.weight) # Normal weight vector multiplication.
        
        # Reshape dims to extract sample dim and batch dim.
        dims = (self.S, -1, 1, x.shape[-1])
        z = torch.matmul(x.view(dims), self.W_noisy)
        return z.reshape((*x.shape[:-1], -1)) 
   
class VanillaBNN(StatefulBase):
    ''' Implements a network of biological (or spiking) neurons with a specified neuron model type. '''
    def __init__(self, net_params, device = 'cpu'):
        super(VanillaBNN,self).__init__()

        self.n_inputs = net_params['n_inputs']
        self.n_hidden = net_params['n_hidden']
        self.n_outputs = net_params['n_outputs']
        
        self.loss_fn = F.cross_entropy # Reductions is mean by default
        if net_params.get('loss_fn', '') == 'mse':
            def mse_loss(x, y):
                one_hot = F.one_hot(y, num_classes = self.n_outputs).float()
                return F.mse_loss(x, one_hot)
            self.loss_fn = mse_loss
        self.acc_fn = xe_classifier_accuracy
        
        # Uses a given filter to convolve the the spike counts over time. 
        # This is a retrospective padding, so will have edge effects at start 
        self.filter_len = net_params['filter_length']

        # Initialize layers
        self.W_inp = WeightSampler(self.n_inputs, self.n_hidden, device)
        self.W_rec = WeightSampler(self.n_hidden, self.n_hidden, device, trainable=False)
        self.W_ro =  WeightSampler(self.n_hidden, self.n_outputs, device)
        self.params = [self.W_inp, self.W_ro]
                
        self.use_snn = net_params.get('use_snn', False)
        if self.use_snn:
            # Surrogate gradients
            spike_grad = surrogate.fast_sigmoid(slope=25)
            self.hidden_neurons = snn.Leaky(beta=net_params['snn_beta'], spike_grad=spike_grad)
        else:
            self.hidden_neurons = HH(self.n_hidden, device)
            # self.hidden_neurons = MorrisLecar(self.n_hidden, device, phi = 2./30.0, V3 = 12., V4 = 17.)
            # self.hidden_neurons = HH_Fast(self.n_hidden, device)

        self.reset_state()
        self.n_per_step = net_params.get('n_per_step', 1) # For how many timesteps should the same batch input be fed in?
        self.random_start = net_params.get('random_start', 0) # Random start data so we can be robust to sequence length.
        self.softmax = net_params.get('softmax', True)
        self.noise_std = net_params.get('noise_std', 0.0)
        
        self.active_param = 0

    def reset_state(self, batchSize=1):
        if self.use_snn:
            # Initialize hidden states of LIFs
            # (shape of the hidden states are automatically initialized based on 
            # the input data dimensions during the first forward pass)
            self.mem_hidden = self.hidden_neurons.init_leaky()
        else:
            self.hidden_neurons.reset_state(batchSize)
            
        self.spk_hidden = torch.zeros((batchSize, self.W_rec.weight.data.shape[-1],), 
                                      device=self.W_rec.weight.data.device) #shape=[B,Nh]
        
        self.timers = torch.zeros_like(self.spk_hidden)

    def forward(self, cur_inp, t):
        """
        Runs a single forward pass of the network 
        (iterated over in self.evaluate() for sequential data)

        z.shape: (B, num_inputs)
        """            
        cur_rec = self.W_rec(self.spk_hidden)
        z1 = cur_inp + cur_rec
        z1 = z1 + torch.normal(torch.zeros_like(z1), self.noise_std)
        
        # Pass input through hidden neurons.
        if self.use_snn:
            spk_hidden, mem_hidden = self.hidden_neurons(z1, self.mem_hidden)
            # self.timers = torch.where(spk_hidden > 0.1, 20, self.timers - 1)
            # spk_hidden = (self.timers > 0).float()
        else:
            spk_hidden = self.hidden_neurons(z1)
            mem_hidden = self.hidden_neurons.V.clone()
        
        z2 = self.W_ro(spk_hidden)
        spk_output = z2
        if self.softmax:
            spk_output = F.softmax(z2, dim = -1) # No output neurons!
            
        # Saves all internal states
        self.spk_hidden = spk_hidden
        self.mem_hidden = mem_hidden
        return spk_output

    def evaluate(self, batch, debug=False): 
        W = self.filter_len # We only need to store the last W timesteps for task.
        
        T = batch[1].shape[1] * self.n_per_step # Simulate for longer than the batch input. 
        B = batch[1].shape[0]
        self.reset_state(batchSize=batch[0].shape[0])
        
        plot = False
        if B == 100:
            plot = True

            # Things to plot
            n_samples = 1
            sample_freq = 10
            hidden_record = np.zeros((n_samples, T // sample_freq, self.n_hidden))
            out_record = np.zeros((n_samples, T // sample_freq, self.n_outputs)) 
        
        # Record the final layer
        shape = (B, W, self.n_outputs) # [B, W, Ny]
        spk_out = torch.zeros(shape, dtype = torch.float, layout = batch[1].layout)
        spk_out = spk_out.to(batch[1].device)
        
        # Simulate population of neurons over time.
        zin = self.W_inp(batch[0])
        for time_idx in range(T):
            bidx = time_idx // self.n_per_step
            out = self(zin[:, bidx, :], time_idx)
            
            if plot:
                if time_idx % sample_freq == 0:
                    hidden_record[:, time_idx // sample_freq, :] = self.mem_hidden[:n_samples].detach().cpu().numpy()
                    out_record[:, time_idx // sample_freq, :] = out[:n_samples].detach().cpu().numpy()
            
            # Only write values in final window.
            write_idx = time_idx - (T - W)
            if write_idx >= 0:
                spk_out[:, write_idx, :] = torch.nan_to_num(out)
                
        if plot:
            # hidden_record = hidden_record.detach().cpu().numpy()
            # out_record = out_record.detach().cpu().numpy()
            
            plt.figure(dpi=400)
            plt.subplot(2,1,1)
            for b in range(n_samples):
                plt.plot(hidden_record[b], linewidth=1)
            plt.title('Hidden Layer')
                
            plt.subplot(2,1,2)
            for b in range(n_samples):
                plt.plot(out_record[b])
            plt.title('Output Layer')
            plt.tight_layout()
            plt.show()
            
        
        self.counter = self.__dict__.get("counter", -1) + 1
        if self.counter % 10 == 0:
            if self.hist is not None and len(self.hist['train_loss']) > 0:
                plot_accuracy(self.hist)
                plt.show()        

        # Return mean of output window
        return torch.mean(spk_out, 1)
        
    def _train_epoch(self, trainData, validBatch=None, batchSize=1, earlyStop=True, earlyStopValid=False, validStopThres=None, 
                   trainOutputMask=None, validOutputMask=None, minMaxIter=(-2, 1e7)):
        for b in range(0, trainData.tensors[0].shape[0], batchSize):  #trainData.tensors[0] is shape [B,T,Nx]
            if 'optims' not in self.__dict__:
                self.optims = []
                lr = self.optimizer.param_groups[0]['lr']
                self.optims.append(torch.optim.Adam(self.W_inp.parameters(), lr=lr))
                self.optims.append(torch.optim.Adam(self.W_ro.parameters(), lr=lr))

            S = self.params[self.active_param].S
            
            inp, target = trainData[b:b+batchSize,:,:] 
            inp = inp.repeat(S, 1, 1)
            target = target.repeat(S, 1, 1)
            trainBatch = (inp, target)
            
            if trainOutputMask is not None: 
                trainOutputMaskBatch = trainOutputMask[b:b+batchSize,:,:] 
            else:
                trainOutputMaskBatch = None
      
            AUTODIFF = False
            
            self.optimizer.zero_grad()
            self.optims[self.active_param].zero_grad()
            with torch.set_grad_enabled(AUTODIFF):
                if not AUTODIFF:
                    self.params[self.active_param].set_active(True)
                    self.params[self.active_param].noisify()
                
                out = self.evaluate(trainBatch) #expects shape [B,T,Nx], out: [B,Ny]
                loss_fn = nn.CrossEntropyLoss(reduction = 'none')
                
                def mse_loss(x, y):
                    one_hot = F.one_hot(y, num_classes = self.n_outputs).float()
                    print(x.shape, one_hot.shape)
                    return F.mse_loss(x, one_hot, reduction='none')
                loss_fn = mse_loss

                losses = loss_fn(out, target[:, 0, 0]) # Shape [B*S, 3]
                losses = torch.mean(losses.reshape((S, -1)), -1) # Shape [S]
                loss = losses[0]
                
            if AUTODIFF:  
   #             loss = self.average_loss(trainBatch, out=out, outputMask=trainOutputMaskBatch)
                loss.backward()
            
                # Zero NaNs in gradients. This can sometimes pop up with BNNs.
                total, corrupted = 0, 0
                grad_norm = 0.0
                for name, param in self.named_parameters():
                    print(name)
                    if torch.any(param.grad.isnan()):
                        corrupted += 1
                    param.grad = torch.nan_to_num(param.grad, 0.0)
                    total += 1
                    
                    grad_norm += param.grad.norm()
                    
                print(f'Percentage of corrupted parameters: {100 * corrupted / total}%')
                print(f'Grad norm: {grad_norm}')
          
                if self.gradientClip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradientClip)          
           
                self.optimizer.step() 
            else:
                grad = self.params[self.active_param].compute_grad(losses)
                print(f'Grad norm: {grad.norm()}')
                self.params[self.active_param].weight.grad = grad
                self.optims[self.active_param].step() 
            
            # Disable samplers.
            for param in self.params:
                param.set_active(False)
                
            self.active_param = (self.active_param + 1) % len(self.params)

            # Note: even though this is called every batch, only runs validation batch every monitor_freq batches
            self._monitor(trainBatch, validBatch=validBatch, out=out, loss=loss, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask) 
            # self._monitor(trainBatch, validBatch=validBatch, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask)  
       
            # if earlyStopValid and len(self.hist['valid_loss'])>1 and self.hist['valid_loss'][-1] > self.hist['valid_loss'][-2]:
            STEPS_BACK = 10
            # Stop if the avg_valid_loss has asymptoted (or starts to increase)
            # (since rolling average is 10 monitors, 2*STEPS_BACK makes sure there are two full averages available for comparison,
            #  subtracting self.hist['monitor_thresh'][-1] prevents this threshold from being tested when a network continues training,)
            if self.hist['iter'] > minMaxIter[0]: # Only early stops when above minimum iteration count
                if earlyStopValid and (len(self.hist['iters_monitor']) - self.hist['monitor_thresh'][-1]) > 2*STEPS_BACK:
                    if 1.0 * self.hist['avg_valid_loss'][-STEPS_BACK] < self.hist['avg_valid_loss'][-1]:
                        print('  Early Stop: avg_valid_loss saturated, current (-1): {:.2e}, prev (-{}): {:.2e}, acc: {:.2f}'.format(
                            self.hist['avg_valid_loss'][-1], STEPS_BACK, self.hist['avg_valid_loss'][-STEPS_BACK], self.hist['avg_valid_acc'][-1]))
                        return True
                if validStopThres is not None and self.hist['avg_valid_acc'][-1] > validStopThres:
                    print('  Early Stop: valid accuracy threshold reached: {:.2f}'.format(
                        self.hist['avg_valid_acc'][-1]
                    ))
                    return True
                if self.hist['iter'] > minMaxIter[1]: # Early stop if above maximum numbers of iters
                    print('  Early Stop: maximum iterations reached, acc: {:.2f}'.format(
                        self.hist['avg_valid_acc'][-1]
                    ))
                    return True
            # if earlyStop and sum(self.hist['train_acc'][-5:]) >= 4.99: #not a proper early-stop criterion but useful for infinite data regime
            #     return True
            
        return False  
        
    
class VanillaGRU(StatefulBase):
    ''' Implements a network of biological (or spiking) neurons with a specified neuron model type. '''
    def __init__(self, net_params, device = 'cpu'):
        super(VanillaGRU,self).__init__()

        self.n_inputs = net_params['n_inputs']
        self.n_hidden = net_params['n_hidden']
        self.n_outputs = net_params['n_outputs']
        
        self.loss_fn = F.cross_entropy # Reductions is mean by default
        if net_params.get('loss_fn', '') == 'mse':
            def mse_loss(x, y):
                one_hot = F.one_hot(y, num_classes = self.n_outputs).float()
                return F.mse_loss(x, one_hot)
            self.loss_fn = mse_loss
        self.acc_fn = xe_classifier_accuracy
        
        self.gru = nn.GRU(self.n_inputs, self.n_hidden, batch_first = True)
        self.linear1 = nn.Linear(self.n_hidden, self.n_outputs)
        
        # Uses a given filter to convolve the the spike counts over time. 
        # This is a retrospective padding, so will have edge effects at start 
        self.filter_len = net_params['filter_length']
        
        self.n_per_step = net_params.get('n_per_step', 1) # For how many timesteps should the same batch input be fed in?
        self.random_start = net_params.get('random_start', 0) # Random start data so we can be robust to sequence length.
        self.softmax = net_params.get('softmax', True)

    def evaluate(self, batch, debug=False):
        inp = torch.repeat_interleave(batch[0], self.n_per_step, dim=1)
        out, _ = self.gru(inp)
        out = self.linear1(out)
        
        if self.random_start > 0:
            for b in range(batch[0].shape[0]):
                end = np.random.randint(batch[1].shape[1] - self.random_start, batch[1].shape[1])
                end = end * self.n_per_step
                out[b, end:] = out[b, end]
                
        self.out = out

        filter = torch.tensor(np.ones((1, 1, self.filter_len,)), dtype=torch.float).to(batch[0].device)
        out_conv = torch.transpose(out, 1, 2) # B, T, Ny -> B, Ny, T (for convolve along dim=2)
        out_conv = out_conv.reshape(-1, out_conv.shape[-1]).unsqueeze(1) # B, Ny, T -> B*Ny, 1, T

        out_cum = F.conv1d(out_conv, filter, padding=self.filter_len-1)[:, :, :-self.filter_len+1] #output: B*Ny, 1, T
        out_cum = torch.transpose(out_cum.reshape(out.shape[0], out.shape[2], out.shape[1]), 1, 2) # B*Ny, 1, T -> B, Ny, T -> B, T, Ny
        out_cum = out_cum[:, -batch[1].shape[1]:, :] # Make output compatible with mask. Really only care about final timestep value. 
        return out_cum