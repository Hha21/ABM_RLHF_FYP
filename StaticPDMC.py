import torch
import torch.nn as nn
import numpy as np 
import random
from collections import deque
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
from torch.distributions import Normal  

from SAC_ABM import SACAgent

sys.path.insert(1, "./build")

# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

# Parameters
state_dim = 206     # (50 firms * 4) + 6 sector features
action_dim = 1

# Load CCSAC Policy
CCSACagent = SACAgent(state_dim, action_dim)
CCSACagent.actor.load_state_dict(torch.load("actor_policy_weights.pt"))
CCSACagent.actor.eval()

class PDMCNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, 1)
        self.logstd_head = nn.Linear(64, 1)
    
    def forward(self, init_state):
        x = self.net(init_state)
        mean = torch.sigmoid(self.mean_head(x))
        log_std = torch.clamp(self.logstd_head(x), -2, 2)
        std = torch.exp(log_std)

        return mean, std

class SPDMCAgent:
    def __init__(self, state_dim):

        self.policy = PDMCNet(state_dim)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr = 1e-4)

        self.baseline = {}

    def select_chi(self, init_state, deterministic = False):
        init_state = torch.FloatTensor(init_state).unsqueeze(0)
        mean, std = self.policy(init_state)
        dist = Normal(mean, std)

        if deterministic:
            chi = mean
        else:
            chi = dist.rsample()
        
        chi = torch.clamp(chi, 0.0, 1.0)
        
        return chi.item(), dist.log_prob(chi)

    def update_policy(self, log_prob, ep_return, scenario):
        # Update baseline
        if scenario not in self.baseline:
            self.baseline[scenario] = ep_return
        else:
            self.baseline[scenario] = 0.95 * self.baseline[scenario] + 0.05 * ep_return
        
        advantage = ep_return - self.baseline[scenario]
        loss = -(log_prob * advantage)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()

def deploy_agent(agent, chi_=0.5, scenario="AVERAGE", output=False, seed=False):
    if seed:
        env = cpp_env.Environment(scenario, seed, target=0.2, chi=chi_)
    else:
        env = cpp_env.Environment(scenario, target=0.2, chi=chi_)
    state = env.reset()
    prev_state = np.zeros(state_dim)
    done = False
    re, ra = 0, 0
    while not done:
        action = agent.actor.get_action(state, prev_state, chi_, deterministic=True)
        next_state, re_ep, ra_ep, done = env.step(action)
        prev_state = state
        state = next_state
        re += re_ep
        ra += ra_ep
    if output:
        env.outputTxt()
    return re, ra

def train_spdmc(agent, CCSACagent, num_episodes = 30000, scenario = "PESSIMISTIC"):
    rewards = []
    losses = []

    choices = ["PESSIMISTIC", "AVERAGE", "OPTIMISTIC"]

    for ep in range(num_episodes):
        
        if ep < 10000:
            choice = "AVERAGE"
        elif 10000 < ep < 20000:
            choice = "PESSIMISTIC"
        elif 20000 < ep < 30000:
            choice = "OPTIMISTIC"
        else:
            choice = random.choice(choices)

        newenv = cpp_env.Environment(tech_mode = choice)
        state = newenv.reset()
        prev_state = np.zeros(state_dim)
        done = False

        chi, log_prob = agent.select_chi(state)

        re = 0
        ra = 0

        while (not done):
            action = CCSACagent.actor.get_action(state, prev_state, chi, deterministic = True)
            next_state, re_ep, ra_ep, done = newenv.step(action)

            prev_state = state
            state = next_state

            re += re_ep
            ra += ra_ep

        ep_return = (1 - chi) * re + chi * ra
        rewards.append(ep_return)

        loss = agent.update_policy(log_prob, ep_return, choice)
        losses.append(loss)

        if ep % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_loss = np.mean(losses[-20:])
            print(f"Episode {ep} | Avg Return (last 20): {avg_reward:.2f} | Avg Loss: {avg_loss:.4f}")
    
    torch.save(agent.policy.state_dict(), "SPDMC_policy_weights.pt")
    
    return rewards, losses

def deploy_SPDMC(scenario = "PESSIMSTIC", seed = None):

    if seed:
        newenv = cpp_env.Environment(scenario, seed)
    else:
        newenv = cpp_env.Environment(scenario)
    
    agent = SPDMCAgent(state_dim)
    agent.policy.load_state_dict(torch.load("SPDMC_policy_weights.pt"))
    agent.policy.eval()

    init_state = newenv.reset()

    chi_test, _ = agent.select_chi(init_state, deterministic = True)
    print(f"Chosen chi for deployment: {chi_test}")

if __name__ == "__main__":
    agent = SPDMCAgent(state_dim)
    rewards, losses = train_spdmc(agent, CCSACagent, num_episodes= 35000)

    deploy_SPDMC()
    deploy_SPDMC(scenario = "AVERAGE")
    deploy_SPDMC(scenario = "OPTIMISTIC")
