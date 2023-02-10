import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from random import randint
from fast_sigmoid import fast_sigmoid

class Resonator(nn.Module):
    def __init__(self, beta, freq, thresh = 1.0, reset = -0.5, rest = 0.0, dt = 0.01):
        ''' beta is decay rate, freq is oscillation frequency '''
        super().__init__()

        self.params0 = beta + freq * 1.0j
        self.params = beta + freq * 1.0j # Complex valued parameter.
        self.params = nn.Parameter(torch.tensor(self.params, requires_grad=True))
        self.thresh = thresh
        self.reset = reset
        self.rest = rest
        self.dt = dt
        self.surrogate = fast_sigmoid()

    def init_resonator(self, N):
#        with torch.no_grad():
#              self.params.real = self.params0.real # ONLY TRAIN FREQUENCY 
        return torch.ones(N, dtype=torch.cfloat) * self.rest # Complex valued state.

    def forward(self, mem, inp):
        mem = mem + self.dt * (self.params * mem + inp)
        spk = self.surrogate(mem.imag)
        mem.real = mem.real * (1 - spk) # Reset.
        return spk, mem 

class OrNet(nn.Module):
    def __init__(self, freqOn = 64, freqOff = 32, scale = 20.0, T = 1000):
        super().__init__()
        self.scale = scale
        self.T = T
        self.res = Resonator(-1.0, 10.0)
        self.freqOn = freqOn
        self.freqOff = freqOff

        # Generate inputs for two different cases.
        self.inpOn = torch.zeros(T)
        self.inpOn[freqOn::freqOn] = scale # Periodic input with given frequency.
#        self.inpOn[1*freqOn+1:] *= 0
        self.inpOff = torch.zeros(T)
        self.inpOff[freqOff::freqOff] = scale
#        self.inpOff[1*freqOff+1:] *= 0

    def forward(self, x1, x2, debug=False):
        # Convert inputs to spiketrains with frequency based on value.
        B = x1.shape[0]
        z = torch.zeros(B, self.T)
        for x in [x1, x2]:
            for b in range(B):
                z[b, :] += self.inpOn.clone() if x[b] > 0 else self.inpOff.clone()
        z += torch.normal(torch.zeros_like(z), self.scale * 0.0) # White noise.

        self.spk_out = torch.zeros((B, self.T))
        self.mems = torch.zeros((B, self.T), dtype=torch.cfloat)
        mem = self.res.init_resonator(B)
        for t in range(self.T):
            self.mems[:, t] = mem
            self.spk_out[:, t], mem = self.res(mem, z[:, t])

        window = int(self.T * 0.5)
        means = torch.mean(self.spk_out[:, -window:], 1) / 0.004

        if debug:
            for i in range(4):
                plt.subplot(2, 2, 1 + i)
                plt.plot(self.mems[i, :].detach().imag)
                plt.plot(self.spk_out[i, :].detach())
                plt.plot(z[i, :].detach() / self.scale)
                mean_out = means[i]
                plt.title(f'{x1[i].item()}, {x2[i].item()}, out = {mean_out:.4f}')
            plt.tight_layout()
            plt.show()

            for i in range(4):
                plt.subplot(2, 2, 1 + i)
                for j in range(1, self.mems.shape[1]):
                    plt.plot([self.mems[i, j-1].detach().real, self.mems[i, j].detach().real], [self.mems[i, j-1].detach().imag, self.mems[i, j].detach().imag], c = (j / self.mems.shape[1], 0.0, 0.0))
                plt.title(f'{x1[i].item()}, {x2[i].item()}, out = {means[i]:.4f}')
            plt.tight_layout()
            plt.show()

        return mem.imag

def test_resonator_basic(T):
    res = Resonator(-1.0, 10.0)

    t = 70.0

    mem = res.init_resonator(1)
    mems = np.ones(T, dtype=complex)
    inp = torch.zeros(T)
    points = [0, 64]
    for p in points:
        inp[p] = t

    for i in range(T):
        mems[i] = mem[0].item()
        _, mem = res(mem, inp[i])
    
    plt.subplot(2,2,1)
    plt.plot(mems.imag)
    plt.subplot(2,2,2)
    plt.plot(mems.real)
    plt.subplot(2,2,(3,4))
    plt.plot(mems.real, mems.imag)
    plt.show()

#test_resonator_basic(1000)

def or_test():
    net = OrNet()
    optim = torch.optim.Adam([net.res.params], lr = 1e-2)
    losses = []
    for batch in range(2000):
        x1 = torch.IntTensor([0, 0, 1, 1])
        x2 = torch.IntTensor([0, 1, 0, 1])
        target = torch.IntTensor([0, 1, 1, 1])

        optim.zero_grad()
        out = net(x1, x2, debug = (batch % 200 == 0))
        loss = torch.mean((out - target)**2)
        loss.backward()
        optim.step()
        losses.append(loss.detach().item())
        print(batch, losses[-1])

        if batch % 200 == 0:
            plt.plot(losses)
            plt.show()
#        if batch == 49:
#            plt.subplot(2,1,1)
#            plt.plot(net.mems[0].detach().real, net.mems[0].detach().imag)
#            plt.title(f'{x1}, {x2}, {target}, {out}')
#            plt.subplot(2,1,2)
#            plt.plot(net.spk_out[0].detach())
#            plt.show()

    plt.plot(losses)
    plt.show()

or_test()
