#///////////////////////////////////////IMPORTS/////////////////////////////////////////#

from select import select
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import copy
# import csv

from operator import itemgetter
# from tabulate import tabulate
# from datetime import date

from SALib.sample import saltelli

from ABM_plots import plot_abatement_analysis


#///////////////////////////////////////SETTINGS////////////////////////////////////////#

# scenarios to compare
scenarios = ["Tax"]

analyse_single_run = True  # show main dyn. for a single run
single_run_details = False  # show time-series for all variables
fixed_seed = True  # fixed-seed for random variables?
# how to select from param_range: mid-point, upper-bound, lower-bound, random, id
single_run_mode = "mid-point"
glob_id = 1  # ID for single_run_mode


print_errors = True
fs = 13  # font size for plots


#////////////////////////////////////////PARAMETERS/////////////////////////////////////#

if fixed_seed == True:
    np.random.seed(42)

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

class c_parameters:
    def __init__(self, variable_parameters):
        # Fixed Params
        self.TP = 30  # no. periods
        self.t_start = 11  # delay until policy starts
        self.t_period = 10  # length of regulation period
        self.t_impl = 10  # no. of implementation periods
        self.D0 = 1  # max. demand
        self.A0 = 1  # emission intensity
        self.B0 = 1  # prod. costs
        self.lamb_n = 20  # no. of technological steps
        self.pe0 = 0.01  # initial permit price
        self.I_d = 0.1  # desired inventory share

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
    
    # Scenarios 
    def load_sc(self,scenario):
        # Load Scenario
        getattr(self,scenario)() 

    def Tax(self):
        self.mode = "Tax"
        self.emission_tax = True
        self.permit_market = False


class c_regulator:

    def __init__(self, p):
        self.qp, self.pe, self.R = np.zeros((3, p.T+2))
        self.emission_tax = False
        self.x = 0

    def update_policy(self, sec, p, t):
        if t >= p.t_start:
            self.x = min((t - p.t_start)/p.t_period + 1, p.t_impl)
            if p.emission_tax == True:
                self.emission_tax = True
                self.set_tax(sec, p, t)


    def set_tax(self, sec, p, t):

        self.pe[t:t+p.t_period] = p.tax * self.x / p.t_impl
        for j in sec:
            j.pe[t:t+p.t_period] = j.c_e[t:t +
                                         p.t_period] = self.pe[t:t+p.t_period]


#////////////////////////////////////////SECTOR/////////////////////////////////////////#

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


#////////////////////////////////////////FIRM///////////////////////////////////////////#

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
        if p.reg.emission_tax == True:
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


#///////////////////////////////////////MARKETS/////////////////////////////////////////#

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



#///////////////////////////////////////DYNAMICS/////////////////////////////////////////#

def run_model(scenarios, param_values=0, calibrating=False, p_cal=0):

    results = []

    # set up random parameters
    if calibrating == False:
        pr = c_parameters(param_values)
        random_par = pr.generate_random_par()

    # iterate through all scenarios
    for scenario in scenarios:

        # load parameters
        if calibrating == False:
            p = c_parameters(param_values)
            p.load_sc(scenario)
            p.load_random_par(random_par)
        else:
            p = p_cal

        # calibrate tax
        if scenario == "Tax" and p.calibrate == True and calibrating == False:
            calibrate_tax(p)

        # initialise agents
        # 1, sector and firms
        p.sec = sec = c_sector(p)
        # 2, regulator
        p.reg = reg = c_regulator(p)

        # run simulation
        t = 1

        while t <= p.T:

            if (t-1) % p.t_period == 0:
                reg.update_policy(sec, p, t)

            sec.apply("set_expectations", p, t)

            sec.apply("abatement", p, t)
            sec.production(p, t)
            trade_commodities(sec, p, t)

            # move to next round
            t += 1

    results.append([sec, p])

    # check for errors
    if print_errors == True and p.error == True and calibrating == False:
        print("Errors found in Scenario", scenario)
        print(p.log)

    return results


def calibrate_tax(p_cal):

    c = 0
    mintax = 0
    maxtax = p_cal.tax

    p_cal.tax = (mintax + maxtax) / 2

    results = run_model(["Tax"], p_cal=p_cal, calibrating=True)
    sec, p = results[0]

    while abs(sec.E[p.T-1] - p.E_max) > p.calibration_threshold:
        if sec.E[p.T-1] >= p.E_max:
            mintax = p_cal.tax
        else:
            maxtax = p_cal.tax

        p_cal.tax = (mintax + maxtax) / 2

        results = run_model(["Tax"], p_cal=p_cal, calibrating=True)
        sec, p = results[0]
        c += 1

        if c > p.calibration_max_runs:
            p_cal.report_error("Error in calibrate_tax: c_max reached")
            break

    return


def calc_abatement_analysis(sc):

    sec, p = sc

    t0 = 1
    T = p.T+1

    delE = [sum([j.e[t]-j.e[t0] for j in sec]) for t in range(t0, T)]
    ab_tot = [-x for x in delE]

    delQ = [sum([j.qg[t]-j.qg[t0] for j in sec]) for t in range(t0, T)]

    def x1(sec, j, t):  # Technology change
        return (j.qg[t0] + j.qg[t]) / 2 * (j.A[t] - j.A[t0])

    def x2(sec, j, t):  # Production change
        return (j.A[t0] + j.A[t]) / 2 * (j.sq[t] * sec.Q[t] - j.sq[t0] * sec.Q[t0])

    ab_1 = [- sum([x1(sec, j, t) for j in sec]) for t in range(t0, T)]
    ab_2 = [- sum([x2(sec, j, t) for j in sec]) for t in range(t0, T)]

    # Further decomposition of production level

    def x21(sec, j, t):  # Compositional change
        return (sec.Q[t0] + sec.Q[t]) / 2 * (j.sq[t] - j.sq[t0])

    def x22(sec, j, t):  # Overall production level change
        return (j.sq[t0] + j.sq[t]) / 2 * (sec.Q[t] - sec.Q[t0])

    def A_mean(sec, j, t):
        return (j.A[t0] + j.A[t]) / 2

    ab_21 = [- sum([x21(sec, j, t) * A_mean(sec, j, t) for j in sec])
             for t in range(t0, T)]  # Compositional change
    ab_22 = [- sum([x22(sec, j, t) * A_mean(sec, j, t) for j in sec])
             for t in range(t0, T)]  # Overall production level change

    return [ab_21, ab_1, ab_22, ab_tot]


measure_names = ["Run", "Scenario", "Emissions", "Abatement \n Costs", "Emissions \n Costs", "Technology \n Adoption", "Compositional \n Change",
                 "Product \n Sales", "Profit \n rate", "Market \n Concentration", "Sales \n Price", "Consumer \n Impact", "Emissions \n Price", "Trading \n Volume"]


def evaluation_measures(results, i):

    measures = []

    for sc in results:
        sec0, p0 = results[0]
        sec, p = sc
        t = p.T+1

        # Effectiveness & Economic Impact
        E = sum([sec.E[ti] for ti in range(t-p.t_period, t)])
        CA = sum([sum([(j.B[ti]-j.B[1])*j.qg[ti] for j in sec])
                 for ti in range(t-p.t_period, t)]) / E
        CE = sum([sum([j.e[ti]*j.c_e[ti] for j in sec])
                 for ti in range(t-p.t_period, t)])
        HHI = sum([sum([j.s[ti]**2 for j in sec])
                  for ti in range(t-p.t_period, t)])

        # Efficiency / Abatement Decomposition
        # "Compositional change","Technology adoption","Reduction of total production"
        ac, at, ar, ab_tot = calc_abatement_analysis(sc)
        AT = sum(at[t-p.t_period:t])
        AC = sum(ac[t-p.t_period:t])

        # Consumer Impact
        S = sum([sum([j.qg_s[ti] for j in sec])
                for ti in range(t-p.t_period, t)])
        PL = sum([sum([j.qg_s[ti] * (j.pg[ti] - (j.c_e[ti] * j.A[ti] + j.B[ti])) for j in sec]) for ti in range(t-p.t_period, t)]
                 ) / sum([sum([j.qg[ti] * (j.c_e[ti] * j.A[ti] + j.B[ti]) for j in sec]) for ti in range(t-p.t_period, t)])
        CC0 = sum([sum([j.s[ti] * j.pg[ti] for j in sec])
                  for ti in range(t-p.t_period, t)]) / 10
        CC = sum([sum([j.s[ti] * j.pg[ti] for j in sec]) for ti in range(t-p.t_period, t)]
                 ) / 10 - sum([p.reg.R[ti] for ti in range(t-p.t_period, t)]) / S
        R = sum([p.reg.R[ti] for ti in range(t-p.t_period, t)])

        # Others
        PE = sum([p.reg.pe[ti] for ti in range(t-p.t_period, t)])
        TV = sum([p.ex.u_t[ti] for ti in range(t)])

        # Corresponds with measure_names
        measures.append([i, p.mode, E, CA, CE, AT, AC,
                        S, PL, HHI, CC0, CC, PE, TV])

    return measures


if analyse_single_run == True:

    print("Starting Single Run")

    param = []
    for par in param_range['bounds']:
        if single_run_mode == "mid-point":
            param.append((par[0]+par[1])/2)
        if single_run_mode == "upper-bound":
            param.append(par[1])
        if single_run_mode == "lower-bound":
            param.append(par[0])
        if single_run_mode == "random":
            param.append(par[0]+(par[1]-par[0])*np.random.random())

    results = run_model(scenarios, param)

print(results)
#print(evaluation_measures(results,1))
