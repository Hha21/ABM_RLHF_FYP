import torch
import torch.nn as nn
import numpy as np 
import sys
import matplotlib.pyplot as plt 

from StaticPDMC import SPDMCAgent

sys.path.insert(1, "./build")
import cpp_env

state_dim = 206    
action_dim = 1


def deploy_SPDMC(scenario = "PESSIMSTIC", seed = None):

    if seed:
        newenv = cpp_env.Environment(scenario, seed)
    else:
        newenv = cpp_env.Environment(scenario)
    
    agent = SPDMCAgent(state_dim, action_dim)
    agent.policy.load_state_dict(torch.load("SPDMC_policy_weights.pt"))
    agent.policy.eval()

    init_state = newenv.reset()

    chi_test = agent.select_chi(init_state, explore_std = 0.0)
    print(chi_test)

deploy_SPDMC()