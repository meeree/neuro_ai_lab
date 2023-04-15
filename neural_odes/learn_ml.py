import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
import os
import time

torch.set_default_dtype(torch.float64)

class ML(nn.Module):
    def __init__(self, L, softmax = True, adjoint=True, bias = True):
        super().__init__()
        self.L = L
        self.V = torch.ones(L) * -40.0
        self.w = torch.ones(L) * 0.014173
        self.Vt = torch.tensor([20.0])
        self.Kp = torch.tensor([100.0])
        self.W_rec = nn.Linear(L, L, bias = bias)
        self.W_out = nn.Linear(L, 3, bias = bias)
        self.softmax = softmax
        self.backwards = False

        self.El = torch.normal(torch.zeros(L) - 55., 2.5)

        self.odeint = odeint_adjoint if adjoint else odeint
    
    def compute_output(self, V):
        ''' Compute "output" of neuron given its voltage '''
        return torch.sigmoid((V - self.Vt) * self.Kp)

    def forward(self, t, state):
        V, w = state
        T = self.compute_output(V)
        z = self.Iapp + self.W_rec(T)

        phi, V1, V2, V3, V4, c = 2./30., -1.2, 18., 12., 17., 20.
        gca, Eca, gk, Ek, gl = 4., 120., 8., -84., 2.
        El = self.El 

        minf = .5 * (1 + torch.tanh((V - V1) / V2))
        winf = .5 * (1 + torch.tanh((V - V3) / V4))
        tauw = 1. / (torch.cosh((V - V3) / (2 * V4)))
        dVdt = z - gca * minf * (V - Eca) - gk * w * (V - Ek) - gl * (V - El)
        dwdt = phi * (winf - w) / tauw
        if self.backwards:
            dVdt, dwdt = -dVdt, -dwdt # Backwards in time!
        return dVdt, dwdt

    def load_state_from_file(self, fl):
        fl_v, fl_w = fl + '_V.pt', fl + '_w.pt'
        if os.path.exists(fl_v) and os.path.exists(fl_w):
            V, w = torch.load(fl_v), torch.load(fl_w)
        else:
            V = torch.normal(torch.ones(self.L) * -40.0, 5.0)
            w = torch.normal(torch.ones(self.L) * 0.5, 0.1)
            torch.save(V, fl_v)
            torch.save(w, fl_w)
        return (V, w)

    def simulate(self, Iapp, tt, tol = 1e-5):
        self.Iapp = Iapp
        state = self.load_state_from_file(f'init_conds_{self.L}')

        # Simulate V and w ODE states over time.
        self.Vs, self.ws = self.odeint(self, state, tt, atol=tol, rtol=tol, method='rk4')

        # Compute outputs of neurons given voltages.
        self.Ts = self.compute_output(self.Vs)
        return self.Ts

        # Compute network output using linear transform and optional softmax. 
        out = self.W_out(self.Ts) 
        if self.softmax:
            out = torch.softmax(out, 1)
        return out 

    def sim_back_forth(self, Iapp, tt, tol = 1e-5):
        self.Iapp = Iapp
        state = self.load_state_from_file(f'init_conds_{self.L}')

        # Simulate forwards then simulate backwards form forwards.
        self.Vs, self.ws = self.odeint(self, state, tt, atol=tol, rtol=tol, method='rk4')
        state = (self.Vs[-1], self.ws[-1])
        tt = torch.flip(tt, [0])
        self.backwards = True
        self.Vs_back, self.ws_back = self.odeint(self, state, tt, atol=tol, rtol=tol, method='rk4')
        self.backwards = False
        return self.Vs, self.Vs_back

def limit_dynamics_Kp():
    tt = torch.linspace(0.0, 30.0, 300) # Samples to plot. Sample every ms.
    torch.manual_seed(0)

    Kps = np.array([10**k for k in np.linspace(-1, 3, 10)])
    grid = np.zeros((len(Kps), len(tt)))
    for Kp_idx, Kp in enumerate(Kps):
        with torch.no_grad():
            model = ML(1, adjoint = False, bias = False)
            model.Kp = torch.tensor([Kp])
            model.W_rec.weight.data[0,0] = 1.0

            Ts = model.simulate(100.0, tt, tol = 1e-2)
            Vs = model.Vs.clone()

            def loss_fn(out):
                return torch.sum(out)

            l0 = loss_fn(Ts)

            dw = 1e-4
            model.W_rec.weight.data[0,0] += dw
            Ts_perb = model.simulate(100.0, tt, tol = 1e-2)
            dLdw = (loss_fn(Ts_perb) - l0) / dw
            dZdw = (model.Vs - Vs) / dw

            grid[Kp_idx, :] = dZdw.squeeze()
            plt.subplot(2,1,1)
            plt.plot(Ts, c = [Kp_idx / len(Kps), 0.0, 0.0])

            plt.subplot(2,1,2)
            plt.plot(dZdw, c = [Kp_idx / len(Kps), 0.0, 0.0])

#    plt.imshow(grid, aspect='auto', interpolation='none')
    plt.show()

def single_neuron_approximation_error():
    tt = torch.linspace(0.0, 500.0, 5000) # Samples to plot. Sample every ms.
    torch.manual_seed(0)
    import tqdm

    K_vals = [10**k for k in np.linspace(1, 3, 10)]
    percents = []
    errs = []
    for K_ind, K in enumerate(tqdm.tqdm(K_vals)):
        with torch.no_grad():
            model = ML(1, adjoint = False, bias = False)
            model.Kp[0] = K
            model.W_rec.weight.data[0,0] = 1.0
            Ts = model.simulate(100.0, tt, tol = 1e-2)
            Vs = model.Vs.clone()
            Ts_deriv = model.Kp * Ts * (1 - Ts)

            def loss_fn(x):
                return torch.sum(x)

            dw = 1e-4
            model.W_rec.weight.data[0,0] += dw
            Ts_perb = model.simulate(100.0, tt, tol = 1e-2)
            dZdw = (model.Vs - Vs) / dw

            smpls = Ts_deriv > 1e-3
            dLdw = torch.sum(Ts_deriv * dZdw)
            approx_dLdw = torch.sum(Ts_deriv * dZdw * smpls)

            def rel_error(x, xapprox):
                return 100 * abs((x - xapprox) / x) # Percents in range [0, 100]

            errs.append(rel_error(dLdw, approx_dLdw))
            percents.append(torch.mean(smpls.float()) * 100)

    plt.subplot(1,2,1)
    plt.plot(K_vals, errs)
    plt.ylabel('Relative Error Percentage (%)')
    plt.xlabel('K')
    plt.xscale('log')
    
    plt.subplot(1,2,2)
    plt.plot(K_vals, percents)
    plt.xlabel('K')
    plt.xscale('log')
    plt.ylabel('Proportion of Points Used (%)')
    plt.show()

single_neuron_approximation_error()
exit()
    

def single_neuron_ml():
    tt = torch.linspace(0.0, 2000.0, 20000) # Samples to plot. Sample every ms.
    torch.manual_seed(0)

    w_vals = [1.0]

    for ind, w in enumerate(w_vals):
        model = ML(1, adjoint = False, bias = False)
        model.W_rec.weight.data[0,0] = w
        Ts = model.simulate(100.0, tt, tol = 1e-2)
        Vs = model.Vs.clone()

        def loss_fn(out):
            return torch.sum(out)

        l0 = loss_fn(Ts)
        dLdT = torch.zeros_like(Ts)
        for i in range(dLdT.shape[0]):
            dT = 1e-2
            Ts_perb = Ts.clone()
            Ts_perb[i] += dT
            dLdT[i] = (loss_fn(Ts_perb) - l0) / dT


        dw = 1e-4
        model.W_rec.weight.data[0,0] += dw
        Ts_perb = model.simulate(100.0, tt, tol = 1e-2)
        dLdw = (loss_fn(Ts_perb) - l0) / dw
        dZdw = (model.Vs - Vs) / dw

        spk_starts = torch.zeros(Ts.shape, dtype=bool)
        spk_ends = spk_starts.clone()
        spk_starts[:-1] = torch.logical_and(Ts[:-1] < 1e-2, Ts[1:] >= 1e-2)
        spk_ends[:-1] = torch.logical_and(Ts[:-1] > 1e-2, Ts[1:] <= 1e-2)
        spks = torch.logical_or(spk_starts, spk_ends)

        plt.subplot(3,len(w_vals),1 + ind)
        plt.title(f'w = {w}')

        Ts_deriv = model.Kp * Ts * (1 - Ts)

        plt.plot(tt, Ts.detach(), zorder=0)
        plt.plot(tt, Ts_deriv.detach(), zorder=20, linestyle='dotted')

        ax2 = plt.gca().twinx()
        ax2.plot(tt, dZdw.detach(), c='black', zorder=10, linestyle='--')
        ax2.plot(tt, (Ts_deriv * dZdw).detach(), c='pink', zorder=10)
        plt.xlim(tt[0], tt[-1])

        def fit(times):
            dZdw_heights = dZdw[times]
            tt_spikes = tt[times[:, 0]]
            m, b = np.polyfit(tt_spikes.numpy(), dZdw_heights.detach().numpy(), 1)
            return m,b,tt_spikes,dZdw_heights

        plt.subplot(3,len(w_vals),1 + len(w_vals) + ind)
        for times in [spk_ends]:
            m, b, tt_spikes, dZdw_heights = fit(times)
            plt.plot(tt_spikes, dZdw_heights.detach(), 'o')
            plt.plot(tt_spikes, m * tt_spikes.numpy() + b)

        plt.ylabel('dZdw at spikes')
        plt.xlabel('Time (ms)')
        plt.xlim(tt[0], tt[-1])

        m, b, tt_spikes, _ = fit(spk_ends)
        approx_dZdw = torch.zeros_like(dZdw)
        approx_dZdw[spk_ends] = m * tt_spikes + b

        plt.subplot(3,len(w_vals),1 + 2 * len(w_vals) + ind)
#        plt.plot(Vs.detach(), dZdw.detach(), linewidth=1.0, zorder=0)
#        colors = torch.tensor([[float(k) / len(Vs), 0.0, 0.0] for k in range(len(Vs))])
#        plt.scatter(Vs.detach(), dZdw.detach(), c=colors, zorder=10)
#
#        plt.axvline(model.Vt, c='green', linestyle='--', zorder=15) 
#        plt.xlabel('$V(t)$')
#        plt.ylabel('$d V(t) / d w$')

        plt.plot(Ts_deriv.detach(), dZdw.detach(), c='black', zorder=10, linestyle='--')

        approx_dLdw = torch.sum(Ts_deriv * dZdw)
        approx_dLdw_spikes_only = torch.sum(Ts_deriv * dZdw * spks)
        print(tt_spikes.shape, approx_dZdw.shape, Ts_deriv[spks].shape)
        print(torch.mean(approx_dZdw))
        approx_dLdw_linear_adjoint = torch.sum(Ts_deriv * approx_dZdw)
        print(approx_dLdw_linear_adjoint)
        def rel_error(x, xapprox):
            return 100 * abs((x - xapprox) / x) # Percents in range [0, 100]
        e1, e2, e3 = rel_error(dLdw, approx_dLdw), rel_error(approx_dLdw, approx_dLdw_spikes_only), rel_error(approx_dLdw, approx_dLdw_linear_adjoint)
        print(f'Finite difference {dLdw:.10f}, using dZdw {approx_dLdw:.10f}. Relative err: {e1:.2f}%')
        print(f'Using spikes only: {approx_dLdw_spikes_only:.10f}. Relative err: {e2:.2f}%, Percent used: {torch.mean(spks.float())*100:.3f}%')
        print(f'Using linear adjoint: {approx_dLdw_linear_adjoint:.10f}. Relative err: {e3:.2f}%')

    plt.show()

single_neuron_ml()
exit()

def ml_forward_back(L):
    tt = torch.linspace(0.0, 200.0, 2000) # Samples to plot. Sample every ms.
    torch.manual_seed(0)
    model = ML(L, adjoint = False)
    Vs, Vs_back = model.sim_back_forth(100.0, tt, 1e-2)

    Vs_back = torch.flip(Vs_back, [0]) # Reverse so same order as Vs.
    Vs, Vs_back = Vs[1000:].detach(), Vs_back[1000:].detach() # Transient.
    tt = tt[1000:]

    plt.subplot(1,2,1)
    plt.plot(model.Vs[:, 0].detach(), model.ws[:, 0].detach())
    plt.xlabel('V'); plt.ylabel('w')

    plt.subplot(1,2,2)
    plt.plot(torch.mean(model.Vs, 1).detach(), torch.mean(model.ws, 1).detach())
    plt.plot(torch.mean(model.Vs_back, 1).detach(), torch.mean(model.ws_back, 1).detach())
    plt.xlabel('V')
    plt.show()

    plt.plot(tt, Vs[:, 0])
    plt.plot(tt, Vs_back[:, 0], '--')
    plt.legend(['Forward', 'Backward'])
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Forward-Backward Evaluation with RK4')
    plt.show()

ml_forward_back(50)

def ml_test(L, adjoint):
    tt_long = torch.linspace(0.0, 100.0, 1000)
    tt_short = torch.linspace(0.0, 100.0, 2)
    for tt, name in zip([tt_short, tt_long], ['Endpoints', 'Linspace']):
        torch.manual_seed(0)
        model = ML(L, adjoint = adjoint)

        # Feedforward 
        t0 = time.time()
        Ts = model.simulate(100.0, tt, tol = 1e-2)
        plt.plot(Ts[:, 0].detach())
        plt.show()
        print(f'Time for {name} simulation: {time.time() - t0}s')

        # Backprop timing test.
        t0 = time.time()
        loss = torch.mean(Ts[-1])
        loss.backward()
        print(f'Time for {name} backprop {time.time() - t0}, W_rec mean grad = {torch.mean(model.W_rec.weight.grad)}')

ml_test(100, True) 
exit()

def train_to_turn_off(L):
    torch.manual_seed(0)
    model = ML(L)
    model.softmax = False

    losses = []
    optim = torch.optim.Adam(model.parameters(), lr = 1e-1)
    for e in range(1000):
        Ts = model.simulate(100.0)
        loss = torch.mean(Ts) 
        loss.backward()
        losses.append(loss.item())

        if e % 5 == 0:
            plt.plot(Ts.detach())
            plt.show()
        optim.step()
        print(f'{e}: {loss.item()}')

    plt.plot(losses)
    plt.ylabel('Loss')
    plt.show()

train_to_turn_off(1)
