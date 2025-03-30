import re
from statistics import mean
from gym import spaces
from turtle import done
import numpy as np
import copy

from ABM_plots import plot_emissions_over_time

param_range = {
    "num_vars": 17,
    "names": ['$N$', '$gamma$', '$delA_0$,$delB_0$', '$e^*$', '$m_0$', '$theta$', '$chi$', '$omg_1/omg_2$', '$delta$', '$delDelta$', '$alpha_{pot}$', '$alpha_{costs}$', '$delAlpha_{costs}$', '$eta$', '$delEta$', '$psi$', '$mu_1$,$mu_2$,$mu_3$'],
    "bounds": [[30, 50],  # 00 - N - Number of firms
               [0.1, 0.5],  # 01 - γ - Price sensitivity of demand
               [0, 0.4],  # 02 - ΔA_0,ΔB_0 - Heterogeneity of production factors
               [0.1, 0.2],  # 03 - e^* - Emission target
               [0.2, 0.4],  # 04 - m0 - Initial mark-up
               [0.04, 0.2],  # 05 - ϑ - Mark-up adaption rate
               [0.025, 0.15],  # 06 - χ - Market-share adaption rate
               [0.2, 5],  # 07 - ω_1/ω_2 - Market-share weight difference
               [0.05, 0.3],  # 08 - δ - Permit price adaption rate
               [0, 0.4],  # 09 - Δδ - Heterogeneity of above
               [0.17, 0.87],  # 10 - α_{pot} - Abatement potential
               [1, 10],  # 11 - α_{costs} - Abatement cost factor
               [0, 0.4],  # 12 - Δα_{costs} - Heterogeneity of above
               [0, 0.4],  # 13 - η - Investment profitability target
               [0, 0.4],  # 14 - Δη - Heterogeneity of above
               [0, 2],  # 15 - ψ - Auction mechanism
               [0, 3]]  # 16 - μ_1,μ_2,μ_3 - Expectation Rule
}

measure_names = [ "Emissions", "Abatement \n Costs", "Emissions \n Costs", "Technology \n Adoption", "Compositional \n Change",
                 "Product \n Sales", "Profit \n rate", "Market \n Concentration", "Sales \n Price", "Consumer \n Impact", "Emissions \n Price"]

class c_parameters:
    def __init__(self, variable_parameters):

        # Fixed Params
        self.TP = 100  # no. periods (30)
        self.t_start = 10  # delay until policy starts (11)
        self.t_period = 10  # length of regulation period
        self.t_impl = 30  # no. of implementation periods
        self.D0 = 1  # max. demand
        self.A0 = 1  # emission intensity
        self.B0 = 1  # prod. costs
        self.lamb_n = 20  # no. of technological steps
        self.I_d = 0.1  # desired inventory share

        self.mode = "Tax"
        self.emission_tax = True

        # Expectation factor ranges
        exp_x_trend = [0.5, 1.0]
        exp_x_adaptive = [0.25, 0.75]

        # Calibration parameters
        self.calibration_threshold = 10**(-3)
        self.calibration_max_runs = 20
        self.tax = 100  # upper bound

        # Variable parameters
        self.N, self.gamma, self.delAB, self.E_max, self.m0, self.theta, self.chi, dOmg, self.delta, self.delDelta, self.lamb_max, self.alpha, self.delAlpha, self.eta, self.delEta, ex_mode, exp_mode = variable_parameters
        self.N = int(round(self.N))

        # Parameter preparation
        self.T = self.t_period * self.TP
        self.m0 = [self.m0] * self.N
        self.omg = [dOmg/(dOmg + 1), 1/(dOmg + 1)]

        if ex_mode <= 1:
            self.ex_mode = "uniform"
        else:
            self.ex_mode = "discriminate"

        if exp_mode < 1:
            self.exp_mode = ["trend"] * self.N
            self.exp_x = exp_x_trend[0] + \
                (exp_x_trend[1] - exp_x_trend[0]) * (exp_mode - 1)
        elif exp_mode < 2:
            self.exp_mode = ["myopic"] * self.N
            self.exp_x = 0
        else:
            self.exp_mode = ["adaptive"] * self.N
            self.exp_x = exp_x_adaptive[0] + (exp_x_adaptive[1]-exp_x_adaptive[0]) * (exp_mode-2) 

        # Toggle model dynamics
        self.calibrate = True
        self.abatement = True

        # Referencing shortcuts
        self.sec, self.reg, self.ex = [0]*3

        # Error log
        self.error = False
        self.log = []

    def generate_random_par(self):
        a = [self.delta * (1 + self.delDelta * (np.random.random() - 0.5)) for i in range(self.N)]
        b = [self.A0 * (1 + self.delAB * (np.random.random() - 0.5)) for i in range(self.N)]
        c = [self.B0 * (1 + self.delAB * (np.random.random() - 0.5)) for i in range(self.N)]
        d = [self.alpha * (1 + self.delAlpha * (np.random.random() - 0.5)) for i in range(self.N)]
        e = [self.eta * (1 + self.delEta * (np.random.random() - 0.5)) for i in range(self.N)]
       
        return [a, b, c, d, e]

    def load_random_par(self, random_par):
        self.delta, self.A0, self.B0, self.alpha, self.eta = random_par
        self.lamb = []  # abatement list
        for i in range(self.N):
            self.lamb.append(self.generate_lamb(self.alpha[i], self.A0[i]))

    # Abatement cost curve
    def generate_lamb(self, alpha, A0):
        lamb = []
        for i in range(self.lamb_n):
            a = (A0*self.lamb_max)/self.lamb_n
            MAC = a*alpha*(i+1)
            b = a*MAC
            lamb.append([a, b])
        return lamb

    # Manage errors

    def report_error(self,statement):
        self.error = True
        if statement not in self.log:
            self.log.append(statement)   

class RL_regulator:

    def __init__(self, p):
        self.pe, self.R = np.zeros((2, p.T+2))
        self.tax = 0.0                  #initial_tax
        self.tax_step = 0.1             #tax_step
        self.x = 0

    def update_policy(self, sec, p, t, rl_agent = None):
        if t >= p.t_start:

            self.x = min((t - p.t_start)/p.t_period + 1, p.t_impl)
            state = [sec.E[t-1], p.reg.pe[t-1]]
            tax_action = rl_agent.act(state)
            self.set_tax(sec, p, t, tax_action)

    def set_tax(self, sec, p, t, tax_value = None):

        if tax_value is not None:
            self.pe[t:t+p.t_period] = tax_value  # RL-controlled tax
        else:
            self.pe[t:t+p.t_period] = p.tax * self.x / p.t_impl
        for j in sec:
            j.pe[t:t+p.t_period] = j.c_e[t:t +
                                         p.t_period] = self.pe[t:t+p.t_period]
    
    def act(self, state):

        last_emissions, last_tax, E, HHI, PL, CC0, CC = state

        if last_emissions > 0.5:
            self.tax += self.tax_step
        else:
            self.tax = max(self.tax - self.tax_step / 2, 0)
        
        return self.tax

class c_sector(list):

    def __init__(self, p):

        super().__init__()
        for j in range(p.N):
            self.append(c_firm(p, j, self))

        # Variable parameters
        self.D, self.E, self.Q, self.u_t = np.zeros((4, p.T+2))

        # Initial values
        for j in self:
            j.s[0] = 1 / p.N
        self.D[0] = p.D0 * np.exp(- sum([j.s[0] * j.pg[0]
                                  for j in self]) * p.gamma)
        for j in self:
            j.D[0] = j.qg_d[0] = j.qg[0] = j.qg_s[0] = j.s[0] * self.D[0]

    def production(self, p, t):
        self.apply("production", p, t)

        # Document total emissions (for grandfathering)
        self.E[t] = sum([j.e[t] for j in self])

        # Document quantity share (for abatement decomposition)
        self.Q[t] = sum([j.qg[t] for j in self])
        for j in self:
            if self.Q[t] > 0:
                j.sq[t] = j.qg[t]/self.Q[t]
            else:
                j.sq[t] = 1/p.N

    def apply(self, method, p, t):
        for obj in self:
            getattr(obj, method)(p, t)

class c_firm:

    def __init__(self, p, j, sec):

        # Params
        self.j = j  # firm index
        self.sec = sec  # sector
        self.alpha = p.alpha[j]  # abatement cost factor
        self.delta = p.delta[j]  # permit price adaption rate
        self.eta = p.eta[j]  # profitability target for investments
        self.exp_mode = p.exp_mode[j]  # expectation rule
        # list of abatement options [[a,b],[a,b],...]
        self.lamb = copy.deepcopy(p.lamb[j])
        self.lamb0 = copy.deepcopy(p.lamb[j])  # copy of list for documentation
        self.x = p.exp_x  # expectation factor

        # dynamic variables
        self.s, self.sq, self.f, self.e, self.qg, self.qg_s, self.qg_d, self.qg_I, self.D, self.Dl, self.pg, self.m, self.A, self.B, self.pe, self.pe2, self.qp_d, self.u_i, self.u_t, self.cu_t, self.c_e, self.c_pr = np.zeros(
            (22, p.T+2))

        # initial values
        self.m[0] = p.m0[j]  # mark-up rate
        self.pg[0] = p.B0[j] * (1+p.m0[j])  # sales price
        self.A[0] = self.A[1] = p.A0[j]  # emissions intensity
        self.B[0] = self.B[1] = p.B0[j]  # production costs

    def set_expectations(self, p, t):

        # set desired output
        if self.exp_mode == "trend" and t <= 1:
            self.qg_d[t] = self.D[t-1] + self.x * (self.D[t-1] - self.D[t-2])
        elif self.exp_mode == "adaptive":
            self.qg_d[t] = self.x * self.D[t-1] + (1 - self.x) * self.qg_d[t-1]
        else:
            self.qg_d[t] = self.D[t-1]

        self.qg_d[t] = max(0, self.qg_d[t] * (1 + p.I_d) - self.qg_I[t-1])

        # set desired mark-up
        if t != 1 and self.s[t-2] > 0.01:
            self.m[t] = self.m[t-1] * \
                (1 + p.theta * (self.s[t-1] - self.s[t-2]) / self.s[t-2])
        else:
            self.m[t] = self.m[t-1]

    def production(self, p, t):

        # Production
        self.qg[t] = self.qg_d[t]

        # Emissions
        self.e[t] = self.qg[t] * self.A[t]

        # Inventory update
        self.qg_I[t] = self.qg_I[t-1] + self.qg[t]

        # Sales price
        self.pg[t] = max(0, (self.A[t] * self.pe[t] +
                         self.B[t]) * (1 + self.m[t]))

        # Tax collection and permit submission
        p.reg.R[t] += self.e[t] * self.pe[t]

    def abatement(self, p, t):

        o, a, b = [0] * 3

        if len(self.lamb) > 0:  # check if abatement options available
            a, b = self.lamb[0]  # extract best abatement option
            MAC = b / a  # marginal costs of abatement

            if p.abatement == True and MAC * (1 + self.eta) <= self.pe[t] and self.s[t-1] > 0.01:

                o = 1  # activate abatement
                self.lamb.pop(0)  # remove used option from list

        self.A[t+1] = self.A[t] - o * a
        self.B[t+1] = self.B[t] + 0 * a

def trade_commodities(sec, p, t):

    for j in sec:
        # fitness
        j.f[t] = -p.omg[0] * j.pg[t] - p.omg[1] * j.Dl[t-1]
    f_mean = sum([j.f[t] * j.s[t-1] for j in sec])

    for j in sec:
        # market-share evolution
        j.s[t] = max(0, j.s[t-1] * (1 - p.chi * (j.f[t] - f_mean) / f_mean))
    # avg. price
    p_mean = sum([j.s[t] * j.pg[t] for j in sec])
    # total demand
    sec.D[t] = D = p.D0 * np.exp(- p_mean * p.gamma)

    for j in sec:
        # demand allocation
        j.D[t] = j.s[t] * D
        # sold goods
        j.qg_s[t] = min(j.D[t], j.qg_I[t])
        # inventory update
        j.qg_I[t] -= j.qg_s[t]
        # unfilled demand
        j.Dl[t] = j.D[t] - j.qg_s[t]

    # correct for numerical errors
    x = 1 - sum([j.s[t] for j in sec])
    if x != 0:
        for i in sec:
            i.s[t] = i.s[t] * 1 / (1-x)

# other
# round left until end of period

def tl(p, t):
    return p.t_period - (t-1) % p.t_period

# rounds passed within period

def tp(p, t):
    return (t-1) % p.t_period

class ClimatePolicyEnv:
    def __init__(self):

        """
        Initializes the ClimatePolicyEnv.

        Parameters:
            param_values (list): A list of parameters. These are used to instantiate c_parameters.
            
        Key parameters (from c_parameters):
            TP         : Total number of rounds (simulation steps)
            t_start    : Round at which the policy starts to be applied
            t_period   : Length of a regulation period (e.g., 10 rounds)
            t_impl     : Number of implementation periods (used to scale the policy gradually)
            D0         : Maximum demand in the goods market
            A0         : Baseline emission intensity for firms
            B0         : Baseline production costs for firms
            lamb_n     : Number of technological steps (abatement options)
            I_d        : Desired inventory share (safety stock level)
            tax        : Initial upper bound for the emission tax
            Variable parameters:
                N         : Number of firms in the sector
                gamma     : Price sensitivity of demand
                delAB     : Heterogeneity in production factors (affecting A0 and B0)
                E_max     : Emission target
                m0        : Initial mark-up rate for firms
                theta     : Mark-up adaptation rate
                chi       : Market share adaptation rate
                dOmg      : Market share weight difference (affects pricing)
                delta     : Permit price adaptation rate
                delDelta  : Heterogeneity of permit price adaptation
                lamb_max  : Maximum abatement potential factor
                alpha     : Abatement cost factor
                delAlpha  : Heterogeneity in abatement cost factor
                eta       : Profitability target for abatement investments
                delEta    : Heterogeneity in the profitability target
                ex_mode   : Expectation mode (uniform/discriminate)
                exp_mode  : Expectation type (trend, myopic, adaptive)
        """

        param_values = []
        for par in param_range["bounds"]:
            param_values.append(par[0] + (par[1] - par[0]) * np.random.random())

        # Init. parameter object (randomised)
        self.params = c_parameters(param_values)
        random_par = self.params.generate_random_par()
        self.params.load_random_par(random_par)

        # Create Sector (collection of firms) and Regulator
        self.sector = c_sector(self.params)
        self.params.reg = self.regulator = RL_regulator(self.params)

        # Set sim. clock to t = 1
        self.t = 1
        self.done = False

        self.results = []

        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (6,), dtype = np.float32)
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape =  (1,), dtype = np.float32)
        self.last_action = 0.0
        self.step(self.last_action)
        
        print(f"Environment Initialised w. {self.params.N} firms, and Emissions Target {self.params.E_max}...")
    
    def reset(self):
        """
        Resets the environment for a new episode.
        Randomizes parameters and resets the simulation clock.
        Returns the initial observation.
        """

        param_values = []
        for par in param_range["bounds"]:
            param_values.append(par[0] + (par[1] - par[0]) * np.random.random())

        self.params = c_parameters(param_values)    
        random_par = self.params.generate_random_par()
        self.params.load_random_par(random_par)

        # Create Sector (collection of firms) and Regulator
        self.sector = c_sector(self.params)
        self.params.reg = self.regulator = RL_regulator(self.params)

        # Set sim. clock to t = 1
        self.t = 1
        self.done = False

        self.last_action = 0.0
        self.step(self.last_action)

        obs = self.observe()
        self.last_obs = obs
        #print(f"Environment Reset w. {self.params.N} firms, and Emissions Target {self.params.E_max}...")

        return obs
    
    def step(self, action):
        """
        Applies an action (e.g., a new tax level) and advances the simulation for one regulation period.
        
        Parameters:
            action (float): The tax level to be applied for the current regulation period.
        
        Returns:
            next_obs (np.array): The next state (observation) after the period.
            reward (float): The reward obtained during the period.
            done (bool): Whether the simulation has ended.
            info (dict): Additional information (e.g., diagnostic data).
        """

        #action = np.clip(action, self.action_space.low, self.action_space.high)
        new_action = np.clip(self.last_action + action, 0, 3)
        self.regulator.set_tax(self.sector, self.params, self.t, tax_value=new_action)

        t_current = self.t

        period = self.params.t_period
        breakloop = False

        if self.t <= self.params.T:
            for _ in range(period):

                if self.t > self.params.T:
                    print(f"t at break: {self.t}")
                    breakloop = True
                    break

                # Each round: update expectations, abatement, production, and trading.
                self.sector.apply("set_expectations", self.params, self.t)
                self.sector.apply("abatement", self.params, self.t)
                self.sector.production(self.params, self.t)
                trade_commodities(self.sector, self.params, self.t)
            
                self.t += 1
        else:
            self.done = True

        next_obs = self.observe()
    
        reward = self.calculate_reward(next_obs, new_action, self.last_action)
        info = {"time" : self.t}
        self.last_obs = next_obs
        self.last_action = action

        return next_obs, reward, self.done, info
    
    def observe(self):

        # 1. Last Emissions, LE
        # 2. Last Tax, LT
        # 3. Total Emissions in Regulation Period, E
        # 4. HHI (Market Concentration), HHI
        # 5. Avg. Profit Rate, PL
        # 6. Avg. Sales Price, CC0

        sec, p, t = self.sector, self.params, self.t
        
        LE = sec.E[t-1]
        LT = p.reg.pe[t-1]

        E = sum([sec.E[ti] for ti in range(t-p.t_period, t)])

        HHI = sum([sum([j.s[ti]**2 for j in sec]) for ti in range(t-p.t_period, t)])

        # Consumer Impact
    
        PL = sum([sum([j.qg_s[ti] * (j.pg[ti] - (j.c_e[ti] * j.A[ti] + j.B[ti])) for j in sec]) for ti in range(t-p.t_period, t)]
                    ) / sum([sum([j.qg[ti] * (j.c_e[ti] * j.A[ti] + j.B[ti]) for j in sec]) for ti in range(t-p.t_period, t)])
        CC0 = sum([sum([j.s[ti] * j.pg[ti] for j in sec])
                    for ti in range(t-p.t_period, t)])

        raw_obs = [LE, LT, E, HHI, PL, CC0]
        max_vals = [1.0, 1.0, 10.0, 1.0, 1.0, 25.0]
        norm_obs = [x / m for x, m in zip(raw_obs, max_vals)]

        #print(f"Observations at t={t}: {obs}")
        #print(f"Normalised at: {normalised_obs}")

        return np.array(norm_obs, dtype=np.float32)

    @staticmethod
    def calculate_reward(obs, action, last_action):
        emissions = obs[2]
        target = 0.2

        deviation = emissions - target

        # emissions_reward = -np.log(1 + 50 * deviation ** 2)
        # smoothness_penalty = -0.5 * (action - last_action) ** 2

        emissions_reward = -np.exp(10 * abs(deviation)) + 1
        smoothness_penalty = -0.2 * (action - last_action)**2

        return emissions_reward + smoothness_penalty

    def close(self):
        pass
    
    def render(self):
        plot_emissions_over_time([self.sector, self.params])
