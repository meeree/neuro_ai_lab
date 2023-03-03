import numpy as np
from tqdm import tqdm

class HH_Like():
    def __init__(self, N, channels, gating_vars):
        import sympy as sp

        # Variables
        self.V = np.zeros(N)
        self.channel_names = list(channels.keys())
        self.channels_currents = np.zeros((len(channels), N))

        self.gating_var_names = list(gating_vars.keys())
        self.gating_vars = np.zeros((len(gating_vars), N))
        self.gating_alpha = np.zeros_like(self.gating_vars)
        self.gating_beta = np.zeros_like(self.gating_vars)

        # Constants
        self.conductances = np.array([np.ones_like(self.V) * vals['conductance'] for vals in channels.values()])
        self.reversals = np.array([np.ones_like(self.V) * vals['reversal'] for vals in channels.values()])

        # Update functions
        # Each update_gating_term function : all gating variables -> gating term.
        setup = []
        for name, vals in channels.items():
            expr_fn = lambda _: np.ones_like(self.V)
            if 'gating_term' in vals:
                # Convert string to a lambda that extracts gating variables by index and evaluates gating term.
                expr = sp.sympify(vals['gating_term'])
                gating_vars_used = list(expr.free_symbols)
                inds = [self.gating_var_names.index(str(g)) for g in gating_vars_used]
                expr_fn = sp.lambdify(gating_vars_used, expr)
            setup.append((expr_fn, inds))

        self.update_gating_terms = lambda G: np.array([fn(*G[inds]) for fn, inds in setup])

        # Each update_alpha/beta function : voltage -> alpha/beta term, respectively.
        update_alphas, update_betas = [], []
        for name, vals in gating_vars.items():
            for term, out_list in zip(['alpha', 'beta'], [update_alphas, update_betas]):
                expr = sp.sympify(vals[term])
                syms = list(expr.free_symbols)
                assert(len(syms) == 1 and str(syms[0]) == 'V')
                out_list.append(sp.lambdify(syms, expr))
        self.update_alphas = lambda V: np.array([fn(V) for fn in update_alphas])
        self.update_betas = lambda V: np.array([fn(V) for fn in update_betas])

    def trapezoid(self, dt, Iapp):
        self.alphas = self.update_alphas(self.V)
        self.betas = self.update_betas(self.V)

        # Trapezoid rule update for gating variables.
        self.gating_vars = (self.alphas * dt + (1 - dt/2 * (self.alphas + self.betas))*self.gating_vars) / (dt/2 * (self.alphas + self.betas) + 1)
        terms = self.update_gating_terms(self.gating_vars) * self.conductances
        self.channel_currents = terms * (self.V - self.reversals) # For analysis, not used.
        G = np.sum(terms, 0)
        E = np.sum(terms * self.reversals, 0)

        # Trapezoid rule update for voltage.
        self.V = (self.V * (1 - dt/2 * G) + dt * (E + Iapp)) / (1 + dt/2 * G)

    def dxdt(self, V, gating_vars, Iapp):
        ''' Compute derivative of states: voltage and gating variables. '''
        self.alphas = self.update_alphas(V)
        self.betas = self.update_betas(V)
        dGdt = self.alphas * (1 - gating_vars) - self.betas * gating_vars

        terms = self.update_gating_terms(gating_vars) * self.conductances
        self.channel_currents = terms * (V - self.reversals)
        dVdt = Iapp - np.sum(self.channel_currents, 0)
        return dVdt, dGdt

    def rk4(self, dt, Iapp):
        V, G = self.V, self.gating_vars
        K1V, K1G = self.dxdt(V, G, Iapp)
        K2V, K2G = self.dxdt(V + dt/2 * K1V, G + dt/2 * K1G, Iapp) 
        K3V, K3G = self.dxdt(V + dt/2 * K2V, G + dt/2 * K2G, Iapp) 
        K4V, K4G = self.dxdt(V + dt * K2V, G + dt * K2G, Iapp) 
        self.V = self.V + dt/6 * (K1V + K4V) + dt/3 * (K2V + K3V)
        self.gating_vars = self.gating_vars + dt/6 * (K1G + K4G) + dt/3 * (K2G + K3G)

    def simulate_and_record(self, T, Iapp, dt, method = 'trapezoid'):
        record = np.zeros((1+self.gating_vars.shape[0], T, self.V.shape[0]))
        def to_record(t):
            record[0, t, :] = self.V
            record[1:, t, :] = self.gating_vars

        # Simulation over time.
        sim_fn = self.trapezoid if method == 'trapezoid' else self.rk4
        for t in tqdm(range(T)):
            to_record(t)
            self.rk4(dt, Iapp[t])
#            self.trapezoid(dt, Iapp)

        return record

channels = \
{
        'Na': {'gating_term': 'm**3 * h', 'conductance': 40.0, 'reversal': 55.0}, 
        'K' : {'gating_term': 'n**4', 'conductance': 35.0, 'reversal': -77.0}, 
        'L': {'conductance': 0.3, 'reversal': -65.0}, 
}

gating_vars = \
{
    'm': 
       {'alpha': '0.182 * (V + 35) / (1 - exp((-V - 35) / 9))',
        'beta': '-0.124 * (V + 35) / (1 - exp((V + 35) / 9))'},
    'n': 
       {'alpha': '0.02 * (V - 25) / (1 - exp((-V + 25) / 9))',
        'beta': '-0.002 * (V - 25) / (1 - exp((V - 25) / 9))'},
    'h': 
       {'alpha': '0.25 * exp((-V - 90.0) / 12.0)',
        'beta': '0.25 * exp((V + 34) / 12.0)'},
}

from matplotlib import pyplot as plt
hh = HH_Like(1, channels, gating_vars)

record = []
for method, dt, T in zip(['trapezoid', 'rk4'], [0.1, 0.01], [10**3, 10**4]):
    Iapp = np.zeros(T)
    inp_len = 5 if method == 'trapezoid' else 50
    Iapp[5*T//10:5*T//10+inp_len] = 20.0
    Iapp[7*T//10:7*T//10+inp_len] = 20.0
    record.append(hh.simulate_and_record(T, Iapp, dt, method))
    print(record[-1].shape)

record[1] = record[1][:, ::10, :] # Subsample since RK4 has 10 times smaller dt.

for i in range(2):
    plt.subplot(2,1,1+i)
    r = record[i]
    r = r[:, 100:, :]
    plt.plot(r[0, :, 0], np.sum(r[1:, :, 0],0))

#    plt.plot(r[0, :, 0])
#    ax = plt.gca().twinx()
#    ax.plot(r[1:, :, 0].T)
plt.show()
