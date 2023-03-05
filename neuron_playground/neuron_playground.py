import numpy as np
from tqdm import tqdm

class HH_Like():
    def __init__(self, N, channels, gating_vars):
        import sympy as sp

        # Variables
        self.V = np.zeros(N) - 70
        self.channel_names = list(channels.keys())
        self.channels_currents = np.zeros((len(channels), N))

        self.gating_var_names = list(gating_vars.keys())
        self.gating_vars = np.zeros((len(gating_vars), N))
        self.infs = np.zeros_like(self.gating_vars)
        self.taus = np.zeros_like(self.gating_vars)

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
                gating_vars_used = [str(g) for g in list(expr.free_symbols)]
                use_inf = np.array([('inf' in g) for g in gating_vars_used])
                print(use_inf, name)
                gating_vars_used = [g.replace('inf', '') for g in gating_vars_used]
                inds = [self.gating_var_names.index(g) for g in gating_vars_used]
                expr_fn = sp.lambdify(gating_vars_used, expr)
            setup.append((expr_fn, inds, use_inf))

        self.update_gating_terms = lambda G, Ginf: np.array([fn(*G[inds]) for fn, inds, use_inf in setup])

        # Each update_inf/tau function : voltage -> inf/tau term for gating variable, respectively.
        update_infs, update_taus = [], []
        for name, vals in gating_vars.items():
            for term, out_list in zip(['inf', 'tau'], [update_infs, update_taus]):
                expr = sp.sympify(vals[term])
                syms = list(expr.free_symbols)
                assert(len(syms) == 1 and str(syms[0]) == 'V')
                out_list.append(sp.lambdify(syms, expr))
        self.update_infs = lambda V: np.array([fn(V) for fn in update_infs])
        self.update_taus = lambda V: np.array([fn(V) for fn in update_taus])

    def trapezoid(self, dt, Iapp):
        self.infs = self.update_infs(self.V)
        self.taus = self.update_taus(self.V)

        # Trapezoid rule update for gating variables.
        self.gating_vars = (self.infs * dt + (self.taus - dt/2) * self.gating_vars) / (self.taus + dt/2)

        # Update gating variable terms by computing g * m^a * h^b for each term (a, b vary).
        terms = self.update_gating_terms(self.gating_vars, self.infs) * self.conductances

        # Update V using channel currents and inputted current/noise.
        self.channel_currents = terms * (self.V - self.reversals) # For analysis, not used.
        G = np.sum(terms, 0)
        E = np.sum(terms * self.reversals, 0)

        # Trapezoid rule update for voltage.
        self.V = (self.V * (1 - dt/2 * G) + dt * (E + Iapp)) / (1 + dt/2 * G)

    def dxdt(self, V, gating_vars, Iapp):
        ''' Compute derivative of states: voltage and gating variables. '''
        self.infs = self.update_infs(V)
        self.taus = self.update_taus(V)
        dGdt = (self.infs - gating_vars) / self.taus

        terms = self.update_gating_terms(gating_vars, self.infs) * self.conductances
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
            sim_fn(dt, Iapp[t])

        return record

def vanilla_hh():
    channels = \
    {
            'Na': {'gating_term': 'm**3 * h', 'conductance': 40.0, 'reversal': 55.0}, 
            'K' : {'gating_term': 'n**4', 'conductance': 35.0, 'reversal': -77.0}, 
            'L': {'conductance': 0.3, 'reversal': -65.0}, 
    }
    gating_vars = \
    {
#        'm': 
#           {'alpha': '0.182 * (V + 35) / (1 - exp((-V - 35) / 9))',
#            'beta': '-0.124 * (V + 35) / (1 - exp((V + 35) / 9))'},
#        'n': 
#           {'alpha': '0.02 * (v - 25) / (1 - exp((-v + 25) / 9))',
#            'beta': '-0.002 * (V - 25) / (1 - exp((V - 25) / 9))'},
#        'h': 
#           {'alpha': '0.25 * exp((-V - 90.0) / 12.0)',
#            'beta': '0.25 * exp((V + 34) / 12.0)'},

        'm': 
           {'inf': '(0.182 * (V + 35) / (1 - exp((-V - 35) / 9))) / (0.182 * (V + 35) / (1 - exp((-V - 35) / 9)) + -0.124 * (V + 35) / (1 - exp((V + 35) / 9)))',
            'tau': '1 / (0.182 * (V + 35) / (1 - exp((-V - 35) / 9)) + -0.124 * (V + 35) / (1 - exp((V + 35) / 9)))'},
        'n': 
           {'inf': '(0.02 * (V - 25) / (1 - exp((-V + 25) / 9))) / (0.25 * exp((-V - 90.0) / 12.0) + -0.002 * (V - 25) / (1 - exp((V - 25) / 9)))',
            'tau': '1 / (0.25 * exp((-V - 90.0) / 12.0) + -0.002 * (V - 25) / (1 - exp((V - 25) / 9)))'},
        'h': 
           {'inf': '(0.25 * exp((-V - 90.0) / 12.0)) / (0.25 * exp((-V - 90.0) / 12.0) + 0.25 * exp((V + 34) / 12.0))',
            'tau': '1 / (0.25 * exp((-V - 90.0) / 12.0) + 0.25 * exp((V + 34) / 12.0))'},
    }
    return channels, gating_vars

def squid_giant_axon():
#    channels = \
#    {
#            'Na': {'gating_term': 'm**3 * h', 'conductance': 120.0, 'reversal': 120.0-65}, 
#            'K' : {'gating_term': 'n**4', 'conductance': 36.0, 'reversal': -12.0-65}, 
#            'L': {'conductance': 0.3, 'reversal': 10.6-65}, 
#    }
    channels = \
    {
            'Na': {'gating_term': 'm**3 * h', 'conductance': 40.0, 'reversal': 55.0}, 
            'K' : {'gating_term': 'n**4', 'conductance': 35.0, 'reversal': -77.0}, 
            'L': {'conductance': 0.3, 'reversal': -65}, 
    }
    gating_vars = \
    {
        'm': 
           {'inf': '1 / (1 + exp((-40 - V)/15))', 
            'tau': '0.04 + 0.46 * exp(-(-38-V)**2 / 30**2)'},
        'n': 
           {'inf': '1 / (1 + exp((-53 - V)/15))', 
            'tau': '1.1 + 4.7 * exp(-(-79-V)**2 / 50**2)'},
        'h': 
           {'inf': '1 / (1 + exp((-62 - V)/-7))', 
            'tau': '1.2 + 7.4 * exp(-(-67-V)**2 / 20**2)'},
#        'm': 
#           {'inf': '(0.1 * (25 - V) / (exp((25-V)/10) - 1)) / (0.1 * (25 - V) / (exp((25-V)/10) - 1) + 4 * exp(-V/18))', 
#            'tau': '1 / (0.1 * (25 - V) / (exp((25-V)/10) - 1) + 4 * exp(-V/18))'},
#        'n': 
#           {'inf': '(0.01 * (V - 10) / (1 - exp((10-V) / 10))) / (0.01 * (V - 10) / (1 - exp((10-V) / 10)) + 0.125 * exp(-V/80))', 
#            'tau': '1 / (0.01 * (V - 10) / (1 - exp((10-V) / 10)) + 0.125 * exp(-V/80))'},
#        'h': 
#           {'inf': '0.07*exp(-V/20) / (0.07*exp(-V/20) + 1/(exp((30-V)/10) + 1))', 
#            'tau': '1 / (0.07*exp(-V/20) + 1/(exp((30-V)/10) + 1))'},
    }
    return channels, gating_vars
#        'm': 
#           {'alpha': '0.1 * (25 - V) / (exp((25-V)/10) - 1)',
#            'beta': '4 * exp(-V/18)'},
#        'n': 
#           {'alpha': '0.01 * (V - 10) / (1 - exp((10-V) / 10))',
#            'beta': '0.125 * exp(-V/80)'},
#        'h': 
#           {'alpha': '0.07*exp(-V/20)',
#            'beta': '1/(exp((30-V)/10) + 1)'},


from matplotlib import pyplot as plt
channels, gating_vars = vanilla_hh()

record = []
Iapps = []
for method, dt, T in zip(['trapezoid', 'rk4'], [0.1, 0.01], [10**3, 10**4]):
    hh = HH_Like(1, channels, gating_vars)
    Iapp = np.ones(T) * 1
    Iapp[:int(30 / dt)] = 0.0
    record.append(hh.simulate_and_record(T, Iapp, dt, method))
    Iapps.append(Iapp)
    print(record[-1].shape)

record[1] = record[1][:, ::10, :] # Subsample since RK4 has 10 times smaller dt.

for i in range(2):
    r = record[i]
    r = r[:, :, :]

    plt.subplot(2,3,1+3*i)
    plt.plot(r[0, :, 0], c='black')
#    ax = plt.gca().twinx()
#    ax.plot(Iapps[i])
#    plt.yticks([])

    plt.subplot(2,3,2+3*i)
    plt.plot(r[1:, :, 0].T, linestyle='--')

    plt.subplot(2,3,3+3*i)
    plt.plot(r[0, :, 0], np.sum(r[1:, :, 0],0))

plt.tight_layout()
plt.show()
