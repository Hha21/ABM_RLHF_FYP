import torch
import torch.nn as nn
import numpy as np 
import sys

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

def deploy_agent(agent, chi_ = 0.5, scenario = "AVERAGE"):
    newenv = cpp_env.Environment(scenario, ENV_SEED, target = 0.2, chi = chi_)
    state = newenv.reset()
    prev_state = np.zeros(state_dim)
    done = False

    while (not done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)

        emissions_q, agree_q = agent.net(state_tensor, prev_state_tensor, chi=torch.FloatTensor([chi_]))

        combined_q = (1 - chi_) * emissions_q + chi_ * agree_q

        # action_probs = torch.softmax(combined_q / temperature, dim = 1)
        # action = torch.multinomial(action_probs, num_samples=1).item()
        action = np.argmax(combined_q.detach())
        print(action)
        print(combined_q)

        prev_state = state
        state, _, _, done = newenv.step(action)

    newenv.outputTxt()

# deploy_agent(agent, chi_ = 0.1, temperature = 0.01, scenario = "OPTIMISTIC")
# deploy_agent(agent, chi_ = 0.3, temperature = 0.01, scenario = "OPTIMISTIC")
# deploy_agent(agent, chi_ = 0.5, temperature = 0.01, scenario = "OPTIMISTIC")
# deploy_agent(agent, chi_ = 0.7, temperature = 0.01, scenario = "OPTIMISTIC")
# deploy_agent(agent, chi_ = 0.9, temperature = 0.01, scenario = "OPTIMISTIC")

deploy_agent(agent, chi_ = 0.1, scenario = "PESSIMISTIC")
# deploy_agent(agent, chi_ = 0.3, temperature = 0.01, scenario = "PESSIMISTIC")
# deploy_agent(agent, chi_ = 0.5, temperature = 0.01, scenario = "PESSIMISTIC")
# deploy_agent(agent, chi_ = 0.7, temperature = 0.01, scenario = "PESSIMISTIC")
deploy_agent(agent, chi_ = 0.9, scenario = "PESSIMISTIC")