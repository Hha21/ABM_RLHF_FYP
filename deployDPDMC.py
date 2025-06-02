import torch
import torch.nn as nn
import numpy as np
import random
import sys
from torch.distributions import Normal
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt

from SAC_ABM import SACAgent
sys.path.insert(1, "./build")
import cpp_env

from DynPDMC import DynamicPDMCAgent
from SAC_ABM import SACAgent

ENV_SEED = 42

state_dim = 206    
action_dim = 1

CCSACagent = SACAgent(state_dim, action_dim)
CCSACagent.actor.load_state_dict(torch.load("actor_policy_weights.pt"))
CCSACagent.actor.eval()

agent = DynamicPDMCAgent(state_dim)
agent.policy.load_state_dict(torch.load("Dynamic_PDMC_weights.pt"))
agent.policy.eval()

def deploy_DPDMC(scenario = "AVERAGE", seed = None):

    if seed:
        env = cpp_env.Environment(tech_mode = scenario, seed = seed)
    else:
        env = cpp_env.Environment(tech_mode = scenario)
    state = env.reset()
    prev_state = np.zeros(state_dim)
    hidden = None
    done = False

    states, chis, log_probs, rewards = [], [], [], []
    prev_chi = 0.05

    agent = DynamicPDMCAgent(state_dim)
    agent.policy.load_state_dict(torch.load("Dynamic_PDMC_weights.pt"))
    agent.policy.eval()

    while not done:
        state_tensor = torch.FloatTensor(state)
        mean, std, hidden = agent.policy(state_tensor, prev_chi, hidden)
        chi = mean

        action = CCSACagent.actor.get_action(state, prev_state, chi.item(), deterministic=True)
        next_state, re, ra, done = env.step(action)

        r = (1 - chi.item()) * re + chi.item() * ra - 0.1 * abs(chi.item() - prev_chi)

        chis.append(chi.item())
        rewards.append(r)

        prev_state = state
        state = next_state
        prev_chi = chi.item()
    
    print(chis)
    print(sum(rewards))
    env.outputTxt()

def plot_chi_trajectories(CCSACagent, agent, scenarios=["PESSIMISTIC", "AVERAGE", "OPTIMISTIC"], n_episodes=250):
    
    state_dim = 206
    max_steps = 0
    all_results = {}
    
    for scenario in scenarios:
        all_chis = []
        for ep in range(n_episodes):
            env = cpp_env.Environment(tech_mode=scenario)
            state = env.reset()
            prev_state = np.zeros(state_dim)
            hidden = None
            done = False
            prev_chi = 0.05
            chis = []
            while not done:
                state_tensor = torch.FloatTensor(state)
                mean, std, hidden = agent.policy(state_tensor, prev_chi, hidden)
                chi = mean
                action = CCSACagent.actor.get_action(state, prev_state, chi.item(), deterministic=True)
                next_state, re, ra, done = env.step(action)
                chis.append(chi.item())
                prev_state = state
                state = next_state
                prev_chi = chi.item()
            all_chis.append(chis)
            max_steps = max(max_steps, len(chis))
    
        for i in range(len(all_chis)):
            if len(all_chis[i]) < max_steps:
                all_chis[i] += [np.nan] * (max_steps - len(all_chis[i]))
        all_results[scenario] = np.array(all_chis)
    
    # Plot
    plt.figure(figsize=(10,6))
    colors = {"PESSIMISTIC": "red", "AVERAGE": "green", "OPTIMISTIC": "blue"}
    for scenario in scenarios:
        chis_arr = all_results[scenario]
        mean_chi = np.nanmean(chis_arr, axis=0)
        std_chi = np.nanstd(chis_arr, axis=0)
        steps = np.arange(max_steps)
        plt.plot(steps, mean_chi, label=scenario, color=colors[scenario])
        plt.fill_between(steps, mean_chi - std_chi, mean_chi + std_chi, color=colors[scenario], alpha=0.2)
    plt.xlabel("Policy Timestep", fontsize = 20)
    plt.ylabel(r"$\chi$", fontsize = 20)
    plt.grid(visible = True, which = "major")
    plt.xlim([0,30])
    plt.ylim([0,1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize = 20)
    plt.tight_layout()
    plt.show()

plot_chi_trajectories(CCSACagent, agent)

# deploy_DPDMC(scenario = "PESSIMISTIC", seed = ENV_SEED)
# deploy_DPDMC(scenario = "AVERAGE", seed = ENV_SEED)
# deploy_DPDMC(scenario = "OPTIMISTIC", seed = ENV_SEED)