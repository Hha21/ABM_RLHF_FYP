import torch
import torch.nn as nn
import numpy as np 
import sys
import matplotlib.pyplot as plt 

from SAC_ABM import SACAgent

# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

# Fixed Seed
ENV_SEED = 42

# Parameters
state_dim = 206     # (50 firms * 4) + 3 sector features
action_dim = 1     # discrete action space

#LOAD CRITIC, RANKER, ACTOR
agent = SACAgent(state_dim, action_dim)

agent.critic1.load_state_dict(torch.load("critic_weights.pt"))
agent.critic1.eval()

agent.actor.load_state_dict(torch.load("actor_policy_weights.pt"))
agent.actor.eval()

agent.ranker.load_state_dict(torch.load("SAC_ranker.pt"))
agent.ranker.eval()

def deploy_agent(agent, chi_ = 0.5, scenario = "AVERAGE", output = True):
    newenv = cpp_env.Environment(scenario, ENV_SEED, target = 0.2, chi = chi_)
    state = newenv.reset()
    prev_state = np.zeros(state_dim)
    done = False

    # REWARDS
    re = 0
    ra = 0

    while (not done):

        action = agent.actor.get_action(state, prev_state, chi_, deterministic = True)
        next_state, re_ep, ra_ep, done = newenv.step(action)

        prev_state = state
        state = next_state

        re += re_ep
        ra += ra_ep
    
    if output:
        newenv.outputTxt()

    return re, ra

#_, _ = deploy_agent(agent, chi_ = 0.5, scenario = "PESSIMISTIC")

def evaluate_returns(agent, scenario, chi_values, n_episodes=10):
    returns = []
    for chi_ in chi_values:
        episode_returns = []
        for _ in range(n_episodes):
            re, ra = deploy_agent(agent, chi_, scenario, output = False)
            return_ep = (1 - chi_) * re + (chi_) * ra
            episode_returns.append(return_ep)
        returns.append(np.mean(episode_returns))
    return np.array(returns)

# chi_values = np.linspace(0, 1, 35)

# returns_opt = evaluate_returns(agent, "OPTIMISTIC", chi_values)
# returns_avg = evaluate_returns(agent, "AVERAGE", chi_values)
# returns_pes = evaluate_returns(agent, "PESSIMISTIC", chi_values)

# # Plotting
# plt.figure(figsize=(8,5))
# plt.plot(chi_values, returns_opt, '-', color = "red", label="Optimistic Scenario", linewidth=2)
# plt.plot(chi_values, returns_pes, '-', color = "blue", label="Pessimistic Scenario",linewidth=2)
# plt.plot(chi_values, returns_avg, '-', color = "green", label="Average Scenario", linewidth=2)
# plt.xlabel(r'$\chi$', fontsize=20)
# plt.ylabel("Average Episode Return", fontsize=20)
# #plt.title("Average Episode Return vs $\chi$")
# plt.legend(fontsize=20)
# # plt.grid(True, alpha=0.3)
# plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
# plt.tight_layout()
# plt.xlim([0.0,1.0])
# plt.ylim([14,30])
# plt.show()

def compare_q_vs_return_across_chi(agent, scenario="AVERAGE", chi_values=None, n_episodes=10):
    if chi_values is None:
        chi_values = np.linspace(0, 1, 11)
    
    # Store results: [chi, mean, std]
    pred_qs_agree = []
    actual_rets_agree = []
    pred_qs_emiss = []
    actual_rets_emiss = []

    for chi_ in chi_values:
        chi_agree_preds = []
        chi_agree_rets = []
        chi_emiss_preds = []
        chi_emiss_rets = []

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False

            agree_return = 0
            emiss_return = 0

            agree_preds = 0
            emiss_preds = 0

            first_ep = True
            gamma = 1

            while not done:
                action = agent.actor.get_action(state, prev_state, chi_, deterministic = True)
                next_state, re_ep, ra_ep, done = newenv.step(action)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                chi_tensor = torch.FloatTensor(np.array(chi_)).unsqueeze(0)
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)

                q1e, q1a = agent.critic1(state_tensor, prev_tensor, chi_tensor, action_tensor)

                if first_ep:
                    emiss_preds += q1e.item()
                    agree_preds += q1a.item()

                prev_state = state
                state = next_state

                emiss_return += re_ep * gamma
                agree_return += ra_ep * gamma

                gamma *= 0.99#agent.gamma
                first_ep = False
            
            chi_agree_rets.append(agree_return)
            chi_emiss_rets.append(emiss_return)

            chi_emiss_preds.append(emiss_preds)
            chi_agree_preds.append(agree_preds) 

        pred_qs_agree.append([np.mean(chi_agree_preds), np.std(chi_agree_preds)])
        actual_rets_agree.append([np.mean(chi_agree_rets), np.std(chi_agree_rets)])
        pred_qs_emiss.append([np.mean(chi_emiss_preds), np.std(chi_emiss_preds)])
        actual_rets_emiss.append([np.mean(chi_emiss_rets), np.std(chi_emiss_rets)])

    pred_qs_agree = np.array(pred_qs_agree)
    actual_rets_agree = np.array(actual_rets_agree)
    pred_qs_emiss = np.array(pred_qs_emiss)
    actual_rets_emiss = np.array(actual_rets_emiss)

    # Set up side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    # Plot for Agreeableness
    ax1.errorbar(chi_values, pred_qs_agree[:,0], yerr=pred_qs_agree[:,1], fmt='-', capsize=4, color = "green",
                 label='Predicted Q (Agreeableness)')
    ax1.errorbar(chi_values, actual_rets_agree[:,0], yerr=actual_rets_agree[:,1], fmt='-', capsize=4, color = "blue",
                label='Actual Return (Agreeableness)')
    ax1.set_xlabel(r'$\chi$', fontsize=16)
    ax1.set_ylabel("Episode Returns", fontsize=16)
    ax1.set_title("Agreeableness Task", fontsize=15, fontweight='bold')
    ax1.set_xlim([0,1])
    ax1.legend(fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot for Emissions
    ax2.errorbar(chi_values, pred_qs_emiss[:,0], yerr=pred_qs_emiss[:,1], fmt='-', capsize=4, color = "red",
                label='Predicted Q (Emissions)')
    ax2.errorbar(chi_values, actual_rets_emiss[:,0], yerr=actual_rets_emiss[:,1], fmt='-', capsize=4, color = "blue",
                label='Actual Return (Emissions)')
    ax2.set_xlabel(r'$\chi$', fontsize=16)
    ax2.set_title("Emissions Task", fontsize=15, fontweight='bold')
    ax2.set_xlim([0,1])
    ax2.legend(fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

chi_grid = np.linspace(0, 1, 11)
compare_q_vs_return_across_chi(agent, scenario="PESSIMISTIC", chi_values=chi_grid, n_episodes=50)