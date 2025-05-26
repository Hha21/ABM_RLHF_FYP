import torch
import torch.nn as nn
import numpy as np 
import sys
import matplotlib.pyplot as plt 

from MultiTaskRL import MultiTaskAgent

# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

# Fixed Seed
ENV_SEED = 42

# Parameters
state_dim = 206     # (50 firms * 4) + 3 sector features
action_dim = 10     # discrete action space

# LOAD AGENT (Ranker currently)

agent = MultiTaskAgent(state_dim, action_dim)
agent.net.load_state_dict(torch.load("ranker_agent_weights.pt"))
agent.net.eval()

def deploy_agent(agent, chi_ = 0.9, scenario = "AVERAGE"):
    newenv = cpp_env.Environment(scenario, ENV_SEED, target = 0.2, chi = chi_)
    state = newenv.reset()
    prev_state = np.zeros(state_dim)
    done = False
    return_ep = 0

    while (not done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)

        emissions_q, agree_q = agent.net(state_tensor, prev_state_tensor, chi=torch.FloatTensor([chi_]))

        combined_q = (1 - chi_) * emissions_q + chi_ * agree_q

        # action_probs = torch.softmax(combined_q / temperature, dim = 1)
        # action = torch.multinomial(action_probs, num_samples=1).item()
        action = np.argmax(combined_q.detach())
        print(f"ACTION TAKEN: {action}")
        print(f"EMISSIONS: {emissions_q}, MAX = {np.argmax(emissions_q.detach())}")
        print(f"AGREE: {agree_q}, MAX = {np.argmax(agree_q.detach())}")


        prev_state = state
        state, re, ra, done = newenv.step(action)

        return_ep += (1 - chi_)*re + chi_*ra

    print(f"EPISODE RETURN {chi_} = {return_ep}")
    #newenv.outputTxt()

deploy_agent(agent, chi_ = 0.5, scenario = "PESSIMISTIC")

def evaluate_returns(agent, scenario, chi_values, n_episodes=50):
    means = []
    stds = []
    for chi_ in chi_values:
        episode_returns = []
        for _ in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False
            return_ep = 0
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                emissions_q, agree_q = agent.net(state_tensor, prev_state_tensor, chi=torch.FloatTensor([chi_]))
                combined_q = (1 - chi_) * emissions_q + chi_ * agree_q
                action = np.argmax(combined_q.detach())
                prev_state = state
                state, re, ra, done = newenv.step(action)
                return_ep += (1 - chi_) * re + chi_ * ra
            episode_returns.append(return_ep)
        means.append(np.mean(episode_returns))
        stds.append(np.std(episode_returns))
    return np.array(means), np.array(stds)

chi_values = np.linspace(0, 1, 35)

returns_opt, stds_opt = evaluate_returns(agent, "OPTIMISTIC", chi_values)
returns_avg, stds_avg = evaluate_returns(agent, "AVERAGE", chi_values)
returns_pes, stds_pes = evaluate_returns(agent, "PESSIMISTIC", chi_values)

plt.figure(figsize=(8,5))
plt.plot(chi_values, returns_opt, color="red", label="Optimistic Scenario", linewidth=2)
plt.plot(chi_values, returns_pes, color="blue", label="Pessimistic Scenario", linewidth=2)
plt.plot(chi_values, returns_avg, color="green", label="Average Scenario", linewidth=2)

# plt.fill_between(chi_values, returns_opt - stds_opt, returns_opt + stds_opt, alpha=0.2)
# plt.fill_between(chi_values, returns_pes - stds_pes, returns_pes + stds_pes, alpha=0.2)
plt.fill_between(chi_values, returns_avg - stds_avg, returns_avg + stds_avg, alpha=0.2, color = "green")

plt.xlabel(r'$\chi$', fontsize=20)
plt.ylabel("Average Episode Return", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.xlim([0.0,1.0])
plt.ylim([14,30])
plt.show()

def compare_q_vs_return_across_chi(agent, scenario="AVERAGE", chi_values=None, n_episodes=100):
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

            # Predicted Q at episode start
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
            emissions_q, agree_q = agent.net(state_tensor, prev_state_tensor, chi=torch.FloatTensor([chi_]))
            chi_agree_preds.append(torch.max(agree_q).item())
            chi_emiss_preds.append(torch.max(emissions_q).item())

            gamma = 1

            while not done:
                emissions_q_step, agree_q_step = agent.net(
                    torch.FloatTensor(state).unsqueeze(0),
                    torch.FloatTensor(prev_state).unsqueeze(0),
                    chi=torch.FloatTensor([chi_])
                )
                combined_q = (1 - chi_) * emissions_q_step + chi_ * agree_q_step
                action = np.argmax(combined_q.detach())
                prev_state = state
                state, re, ra, done = newenv.step(action)
                agree_return += ra * gamma
                emiss_return += re * gamma

                gamma *= agent.gamma

            chi_agree_rets.append(agree_return)
            chi_emiss_rets.append(emiss_return)
        
        pred_qs_agree.append([np.mean(chi_agree_preds), np.std(chi_agree_preds)])
        actual_rets_agree.append([np.mean(chi_agree_rets), np.std(chi_agree_rets)])
        pred_qs_emiss.append([np.mean(chi_emiss_preds), np.std(chi_emiss_preds)])
        actual_rets_emiss.append([np.mean(chi_emiss_rets), np.std(chi_emiss_rets)])

    # Convert to arrays for easier indexing
    pred_qs_agree = np.array(pred_qs_agree)
    actual_rets_agree = np.array(actual_rets_agree)
    pred_qs_emiss = np.array(pred_qs_emiss)
    actual_rets_emiss = np.array(actual_rets_emiss)

    # Set up side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    # Plot for Agreeableness
    ax1.plot(chi_values, pred_qs_agree[:,0], color = "green", label='Predicted Q (Agreeableness)')
    ax1.fill_between(chi_values, pred_qs_agree[:,0] - pred_qs_agree[:,1], pred_qs_agree[:,0] + pred_qs_agree[:,1], alpha = 0.2, color = "green")
    ax1.plot(chi_values, actual_rets_agree[:,0], color = "blue", label='Actual Return (Agreeableness)')
    ax1.fill_between(chi_values, actual_rets_agree[:,0] - actual_rets_agree[:,1], actual_rets_agree[:,0] + actual_rets_agree[:,1], alpha = 0.2, color = "blue")
    ax1.set_xlabel(r'$\chi$', fontsize=20)
    ax1.set_ylabel("Returns from " r'$s_{0}$', fontsize=20)
    ax1.set_title("Agreeableness Task", fontsize=15, fontweight='bold')
    ax1.set_xlim([0,1])
    ax1.set_ylim([15,40])
    ax1.legend(fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot for Emissions
    ax2.plot(chi_values, pred_qs_emiss[:,0], color = "red", label='Predicted Q (Emissions)')
    ax2.fill_between(chi_values, pred_qs_emiss[:,0] - pred_qs_emiss[:,1], pred_qs_emiss[:,0] + pred_qs_emiss[:,1], alpha = 0.2, color = "red")
    ax2.plot(chi_values, actual_rets_emiss[:,0], color = "blue", label='Actual Return (Emissions)')
    ax2.fill_between(chi_values, actual_rets_emiss[:,0] - actual_rets_emiss[:,1], actual_rets_emiss[:,0] + actual_rets_emiss[:,1], alpha = 0.2, color = "blue")
    ax2.set_xlabel(r'$\chi$', fontsize=20)
    ax2.set_title("Emissions Task", fontsize=15, fontweight='bold')
    ax2.set_xlim([0,1])
    ax2.set_ylim([10,35])
    ax2.legend(fontsize=18)
    ax2.grid(True, linestyle='--', alpha=0.5)

    #plt.suptitle(f"Predicted Q vs Actual Return Across $\chi$ ({scenario} scenario)", fontsize=17, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

chi_grid = np.linspace(0, 1, 100)
#compare_q_vs_return_across_chi(agent, scenario="PESSIMISTIC", chi_values=chi_grid, n_episodes=100)
