#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:23:16 2023

@author: ws3
"""

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

class HH_Fast(nn.Module):
    def __init__(self, L, device):
        super().__init__()
        self.L = L
        
        # State varaibles. V is the voltage B, [L], B batch size, L layer count, 
        # K is the gating variables [3, B, L], T is the output [B, L].
        self.T = torch.ones(()) 
        
        # HH parameters. These can be tweaked after initializing the HH model.
        self.gna = torch.tensor([120.0]); self.gk = torch.tensor([36.0]); self.gl = torch.tensor([0.3]);
        self.Ena = torch.tensor([55.0]); self.Ek = torch.tensor([-77.0]); self.El = torch.tensor([-65.0]);
        self.Iapp = torch.tensor([0.0]); self.Vt = torch.tensor([-3.0]); self.Kp = torch.tensor([8.0]); 
        self.device = device
        
        self.Iapp = nn.Parameter(torch.tensor([0.0]))
                
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
        
    def reset_state(self, B):
        V = torch.ones((B, self.L)).to(self.device) * -65.0
        self.T = torch.zeros_like(V).to(self.device)
        
        # Gating variables
        K = torch.zeros((3, B, self.L)).to(self.device)
        return V, K

    def forward(self, t, state):
        ''' Return dSdt where S is state '''
        V, K = state
        m, n, h = K
        
        self.T = torch.sigmoid((V - self.Vt) / self.Kp)
        
        # Calculate V intermediate channel quantities.
        pow1 = self.gna * (m ** 3) * h
        pow2 = self.gk * n ** 4
        G = pow1 + pow2 + self.gl
        E = pow1 * self.Ena + pow2 * self.Ek + self.gl * self.El
        dVdt = self.Iapp - G * V + E

        # Calculate gating variable intermediate rate quantities.
        inf = torch.sigmoid((V - self.inf_V12) / self.inf_k)
        tau = self.tau_base + self.tau_amp * torch.exp(-(self.tau_Vmax - V) / self.tau_var)
        dKdt = (inf - K) / tau
        return dVdt, dKdt

    def simulate(self, B, T):
        V, K = self.reset_state(B)
        state = (V, K)
        tt = torch.linspace(0.0, T / 10, T)
        solution = odeint(self, state, tt, atol=1e-5, rtol=1e-5)
        V, K = solution # Shapes [T, B, L], [T, 3, B, L]
        V, K = V.transpose(0, 1), K.transpose(0, 1).transpose(1, 2) # [B, T, L], [3, B, T, L]
        return V, K
    
hh = HH_Fast(100, 'cuda')
hh.Iapp = nn.Parameter(torch.linspace(-20.0, 20.0, 100).reshape(1, -1)).cuda()
V, _ = hh.simulate(1, 500)
losses = torch.mean(torch.sigmoid((V - hh.Vt) / hh.Kp), 1).detach().reshape(-1)
plt.plot(hh.Iapp.detach().reshape(-1), losses)
plt.show()

for i in range(0, 100, 10):
    plt.plot(V[0, :, i].detach())
    plt.title(f'Iapp = {hh.Iapp[0, i].item()}')
    plt.show()

exit()

optim = torch.optim.Adam(hh.parameters(), lr = 0.01)
Iapps = []
losses = []
for i in range(200):
    print(i)
    optim.zero_grad()
    V, K = hh.simulate(1, 500)
    loss = torch.mean(torch.sigmoid((V - hh.Vt) / hh.Kp))
    loss.backward()
    optim.step()
    losses.append(loss.item())
    Iapps.append(hh.Iapp.item())

plt.plot(Iapps)
plt.show()

plt.plot(losses)
plt.show()

print(hh.Iapp)

plt.plot(V.detach()[0, :, 0])
plt.show()