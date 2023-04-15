import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float64)


class ML(nn.Module):
    def __init__(self, adjoint=True, winp = 1.):
        super().__init__()
        # self.t0 = nn.Parameter(torch.tensor([0.0]))
        # self.V = nn.Parameter(torch.tensor([-40.0]))
        # self.w = nn.Parameter(torch.tensor([0.014173]))
        self.t0 = torch.tensor([0.0])
        self.V = torch.tensor([-40.0])
        self.w = torch.tensor([0.014173])
        
        self.odeint = odeint_adjoint if adjoint else odeint
        self.winp = nn.Parameter(torch.tensor([winp]))

    def forward(self, t, state):
        V, w = state
        phi, V1, V2, V3, V4, c = 2./30., -1.2, 18., 12., 17., 20.
        gca, Eca, gk, Ek, gl, El = 4., 120., 8., -84., 2., -60.
        minf = .5 * (1 + torch.tanh((V - V1) / V2))
        winf = .5 * (1 + torch.tanh((V - V3) / V4))
        tauw = 1. / (torch.cosh((V - V3) / (2 * V4)))
        Iapp = self.winp * 100.0
        dVdt = (Iapp - gca * minf * (V - Eca) - gk * w * (V - Ek) - gl * (V - El))/c
        dwdt = phi * (winf - w) / tauw
        return dVdt, dwdt

    def simulate(self):
        state = (self.V, self.w)
        tt = torch.tensor([0.0, 100.0])
        tt = torch.linspace(0.0, 100.0, 1000)
        solution = self.odeint(self, state, tt, atol=1e-5, rtol=1e-5)
        return solution[0]
    
winps = torch.linspace(0.0, 2.5, 1)
losses = []
for winp in winps:
    model = ML(winp=winp)
    print(model.V)
    V = model.simulate()
    print(model.V)
    losses.append(V[-1].item())
plt.plot(winps, losses)
plt.show()

model = ML()
V = model.simulate()
plt.plot(V.detach())
loss = V[-1]
loss.backward()
for name, p in model.named_parameters():
    print(name, p.grad)
    
print(model.winp.item(), (model.winp - 0.1 * model.winp.grad).item())

model_perb_up = ML(winp=1.01)
loss_perb_up = model_perb_up.simulate()[-1]

model_perb_down = ML(0.99)
loss_perb_down = model_perb_down.simulate()[-1]

print((loss_perb_up - loss_perb_down) / 0.01)

exit()
