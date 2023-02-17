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
        
    def reset_state(self, B):
        self.V = torch.ones((B, self.L)).to(self.device) * -65.0
        self.V += torch.normal(torch.zeros_like(self.V), 5.0) # Add some noise to initial conditions.
        self.T = torch.zeros_like(self.V).to(self.device)
                
        # Gating variables
        self.K = torch.zeros((3, B, self.L)).to(self.device)

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
        E = pow1 * self.Ena + pow2 * self.Ek + self.gl * self.El

        # V update.
        self.V = (self.dt * (E + self.Iapp + z) +  (1 - G_scaled) * self.V.clone()) / (1 + G_scaled)

        # Calculate gating variable intermediate rate quantities.
        v_off = self.V[:, :] + self.offs
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
        self.T = torch.sigmoid((self.V - self.Vt) / self.Kp)
        return self.T
    
class VanillaBNN(StatefulBase):
    ''' Implements a network of biological (or spiking) neurons with a specified neuron model type. '''
    def __init__(self, net_params, device = 'cpu'):
        super(VanillaBNN,self).__init__()

        self.n_inputs = net_params['n_inputs']
        self.n_hidden = net_params['n_hidden']
        self.n_outputs = net_params['n_outputs']
        
        self.loss_fn = F.cross_entropy # Reductions is mean by default
        self.acc_fn = xe_classifier_accuracy
        
        # Uses a given filter to convolve the the spike counts over time. 
        # This is a retrospective padding, so will have edge effects at start 
        self.filter_len = net_params['filter_length']

        # Initialize layers
        self.W_inp = nn.Linear(self.n_inputs, self.n_hidden)
        self.W_rec = nn.Linear(self.n_hidden, self.n_hidden)
        self.W_ro = nn.Linear(self.n_hidden, self.n_outputs)
        self.W_inp.weight.data *= 10
        self.W_rec.weight.data *= 10

        self.z1 = torch.zeros(())
        
        self.use_snn = net_params.get('use_snn', False)
        if self.use_snn:
            # Surrogate gradients
            spike_grad = surrogate.fast_sigmoid(slope=25)
            self.hidden_neurons = snn.Leaky(beta=net_params['snn_beta'], spike_grad=spike_grad)
        else:
            self.hidden_neurons = HH(self.n_hidden, device)

        self.reset_state()
        self.trunc = net_params.get('trunc', -1) # Truncation for TBTT.

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

    def forward(self, x, t, debug=False):
        """
        Runs a single forward pass of the network 
        (iterated over in self.evaluate() for sequential data)

        x.shape: (B, num_inputs)
        """            
        cur_inp = self.W_inp(x)
        cur_rec = self.W_rec(self.spk_hidden)
        z1 = cur_inp + cur_rec
        z1 = z1 + torch.normal(torch.zeros_like(z1), 1.0)
        
        # Pass input through hidden neurons.
        if self.use_snn:
            spk_hidden, mem_hidden = self.hidden_neurons(z1, self.mem_hidden)
        else:
            spk_hidden = self.hidden_neurons(z1)
            mem_hidden = self.hidden_neurons.V.clone()

        self.z1[:, t, :] = spk_hidden
        
        z2 = self.W_ro(spk_hidden)
        spk_output = F.softmax(z2, dim = -1) # No output neurons!
            
        # Saves all internal states
        self.spk_hidden = spk_hidden
        self.mem_hidden = mem_hidden

        if debug:
            db_step = {
                'spk_hidden': spk_hidden,
                'mem_hidden': mem_hidden,
            }
            return spk_output, db_step
        else:
            return spk_output

    def evaluate(self, batch, debug=False): 
        if self.trunc < 0:
            self.trunc = batch[1].shape[1]
            
        n_per_step = 40 # For how many timesteps should the same batch input be fed in?
        T = batch[1].shape[1] * n_per_step # Simulate for longer than the batch input. 
        self.reset_state(batchSize=batch[0].shape[0])

        # Record the final layer
        out_size = torch.Size([batch[1].shape[0], T, self.n_outputs]) # [B, T, Ny]
        spk_out = torch.empty(out_size, dtype=torch.float, layout=batch[1].layout, device=batch[1].device)*0 # This has to be a float, otherwise causes gradient problems
        
        hidden_size = torch.Size([batch[1].shape[0], T, self.n_hidden]) # [B, T, Nh]
        self.z1 = torch.empty(hidden_size, dtype=torch.float, layout=batch[1].layout, device=batch[1].device)
        if debug:
            hidden_size = torch.Size([batch[1].shape[0], batch[1].shape[1], self.n_hidden]) # [B, T, Nh]
            db = {
                'spk_hidden': torch.empty(hidden_size, dtype=torch.float, device=batch[1].device),
                'mem_hidden': torch.empty(hidden_size, dtype=torch.float, device=batch[1].device),
            }

        random_start = np.random.randint(1, T // 10) # This makes us have variable length inputs!
        for time_idx in range(random_start, T):
            bidx = time_idx // n_per_step
            x = batch[0][:, bidx, :] # [B, Nx]
            
            grad_cap = batch[0].shape[1] - self.trunc
            with torch.set_grad_enabled(bidx >= grad_cap):
                spk_out[:, time_idx, :] = self(x, time_idx, debug=debug)
                
#             if debug:
#                 db['spk_hidden'][:, time_idx, :] = db_step[0]['spk_hidden']
#                 db['mem_hidden'][:, time_idx, :] = db_step[0]['mem_hidden']
                                
        self.counter = self.__dict__.get("counter", -1) + 1
        if self.counter % 5 == 0:
            plt.subplot(2,1,1)
            plt.plot(self.z1[0, :, :].detach().cpu())
            plt.subplot(2,1,2)
            plt.plot(spk_out[0, 1:, :].detach().cpu())
            plt.show()

        filter = torch.tensor(np.ones((1, 1, self.filter_len,)), dtype=torch.float).to(batch[0].device)
        spk_out_conv = torch.transpose(spk_out, 1, 2) # B, T, Ny -> B, Ny, T (for convolve along dim=2)
        spk_out_conv = spk_out_conv.reshape(-1, spk_out_conv.shape[-1]).unsqueeze(1) # B, Ny, T -> B*Ny, 1, T

        spk_out_cum = F.conv1d(spk_out_conv, filter, padding=self.filter_len-1)[:, :, :-self.filter_len+1] #output: B*Ny, 1, T
        spk_out_cum = torch.transpose(spk_out_cum.reshape(spk_out.shape[0], spk_out.shape[2], spk_out.shape[1]), 1, 2) # B*Ny, 1, T -> B, Ny, T -> B, T, Ny
        spk_out_cum = spk_out_cum[:, -batch[1].shape[1]:, :] # Make output compatible with mask. Really only care about final timestep value. 

        if debug:
            db['spk_out'] = spk_out
            return db
        else:
            return spk_out_cum