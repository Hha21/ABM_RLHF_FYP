import torch
import torch.nn as nn
import numpy as np
import random
import sys
from torch.distributions import Normal
from collections import deque
from copy import deepcopy

from SAC_ABM import SACAgent
sys.path.insert(1, "./build")
import cpp_env

# Parameters
state_dim = 206
action_dim = 1

#LOAD CCSAC
CCSACagent = SACAgent(state_dim, action_dim)
CCSACagent.actor.load_state_dict(torch.load("actor_policy_weights.pt"))
CCSACagent.actor.eval()

class DynamicPDMC(nn.Module):
    def __init__(self, state_dim, dense_dim = 128, lstm_dim = 256):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim + 1, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dense_dim),
            nn.ReLU(),
            )

        self.lstm = nn.LSTM(input_size = dense_dim, hidden_size = lstm_dim, num_layers = 3, batch_first = True)
        self.mean_head = nn.Linear(lstm_dim, 1)
        self.logstd_head = nn.Linear(lstm_dim, 1)
    
    def forward(self, state, prev_chi, hidden):
        x = torch.cat([state, torch.FloatTensor([prev_chi])], dim=0)
        x = self.dense(x)
        lstm_input = x.unsqueeze(0).unsqueeze(0)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        out = lstm_out.squeeze(0).squeeze(0)
        mean = torch.sigmoid(self.mean_head(out))
        log_std = torch.clamp(self.logstd_head(out), -2, 2)
        std = torch.exp(log_std)
        return mean, std, hidden

class DynamicPDMCAgent:
    def __init__(self, state_dim):
        self.policy = DynamicPDMC(state_dim)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr = 5e-5)
        self.chi_penalty = 0.1
        self.max_steps = 30
        self.baseline = np.zeros(self.max_steps)
        self.counts = np.zeros(self.max_steps)
    
    def train_episode(self, CCSACagent, scenario = "AVERAGE"):
        env = cpp_env.Environment(tech_mode = scenario)
        state = env.reset()
        prev_state = np.zeros(state_dim)
        hidden = None
        done = False

        states, chis, log_probs, rewards = [], [], [], []
        prev_chi = 0.05

        while not done:
            state_tensor = torch.FloatTensor(state)
            mean, std, hidden = self.policy(state_tensor, prev_chi, hidden)
            dist = Normal(mean, std)
            chi = dist.rsample().clamp(0,1)

            action = CCSACagent.actor.get_action(state, prev_state, chi.item(), deterministic=True)
            next_state, re, ra, done = env.step(action)

            r = (1 - chi.item()) * re + chi.item() * ra - self.chi_penalty * abs(chi.item() - prev_chi)

            states.append(state_tensor)
            chis.append(chi)
            log_probs.append(dist.log_prob(chi))
            rewards.append(r)

            prev_state = state
            state = next_state
            prev_chi = chi.item()

        returns = np.cumsum(rewards[::-1])[::-1].copy()
        for t in range(len(returns)):
            self.counts[t] += 1
            self.baseline[t] = 0.95 * self.baseline[t] + 0.05 * returns[t]

        baseline = torch.FloatTensor(self.baseline[:len(returns)])
        advantages = torch.FloatTensor(returns) - baseline

        log_probs = torch.stack(log_probs).squeeze()

        loss = -(log_probs * advantages).mean()

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return returns[0].item(), loss.item()

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


if __name__ == "__main__":
    agent = DynamicPDMCAgent(state_dim)
    episodes = 5000

    for ep in range(episodes):
        total_return, loss = agent.train_episode(CCSACagent, scenario = "AVERAGE")

        if ep % 10 == 0:
            print(f"Episode {ep} | Return: {total_return:.2f} | Loss: {loss:.4f} | Baseline: {agent.baseline[0]:.2f}")

    torch.save(agent.policy.state_dict(), "Dynamic_PDMC_weights.pt")

    deploy_DPDMC(scenario = "PESSIMISTIC")


