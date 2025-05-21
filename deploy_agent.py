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

deploy_agent(agent, chi_ = 0.9, scenario = "PESSIMISTIC")

def evaluate_returns(agent, scenario, chi_values, n_episodes=100):
    returns = []
    for chi_ in chi_values:
        episode_returns = []
        for _ in range(n_episodes):
            # Slightly modified deploy_agent to return final return_ep
            newenv = cpp_env.Environment(scenario, target = 0.2, chi = chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False
            return_ep = 0
            while (not done):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                emissions_q, agree_q = agent.net(state_tensor, prev_state_tensor, chi=torch.FloatTensor([chi_]))
                combined_q = (1 - chi_) * emissions_q + chi_ * agree_q
                action = np.argmax(combined_q.detach())
                prev_state = state
                state, re, ra, done = newenv.step(action)
                return_ep += (1 - chi_)*re + chi_*ra
            episode_returns.append(return_ep)
        returns.append(np.mean(episode_returns))
    return np.array(returns)

chi_values = np.linspace(0, 1, 35)

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