import torch
import torch.nn as nn
import numpy as np 
import random
from collections import deque
import sys
import matplotlib.pyplot as plt
# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

env = cpp_env.Environment()

# Parameters
state_dim = 203     # (50 firms * 4) + 3 sector features
action_dim = 10     # discrete action space

# Multi-Task Q Net.
class MultiTaskQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # FIRM FEATURES
        self.firm_branch = nn.Sequential(
            nn.Linear(200, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        # SECTOR FEATURES
        self.sector_branch = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )

        # SHARED LAYERS
        self.shared_layers = nn.Sequential(
            nn.Linear(128 + 32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # EMISSIONS TASK HEAD
        self.emissions_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # AGREEABLENESS TASK HEAD
        self.agreeableness_task_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):

        firm_features = state[:, :200]
        sector_features = state[:, 200:]

        firm_out = self.firm_branch(firm_features)
        sector_out = self.sector_branch(sector_features)

        combined = torch.cat([firm_out, sector_out], dim = 1)
        shared = self.shared_layers(combined)

        emissions_q = self.emissions_head(shared)
        agreeableness_q = self.agreeableness_task_head(shared)

        return emissions_q, agreeableness_q

# AGENT
class MultiTaskAgent:
    def __init__(self, state_dim, action_dim, decay_rate, chi=0.5, temperature=0.6):
        self.net = MultiTaskQNet(state_dim, action_dim)

        # SEPARATE OPTIMISERS FOR TWO HEADS, AND SHARED LAYERS
        self.optimiser_emissions = torch.optim.Adam(self.net.emissions_head.parameters(), lr = 1e-4)
        self.optimiser_agreeableness = torch.optim.Adam(self.net.agreeableness_task_head.parameters(), lr = 1e-4)
        self.optimiser_shared = torch.optim.Adam(self.net.shared_layers.parameters(), lr = 5e-4)

        self.gamma = 0.99
        self.memory = deque(maxlen = 5000)
        self.batch_size = 64                        #Init
        self.epsilon = 1.0
        self.decay_rate = decay_rate
        self.chi = chi
        self.temp = temperature

        self.loss_e_rec = 0.0
        self.loss_a_rec = 0.0

    def get_action(self, state):
    
        if (random.random() < self.epsilon):
            return random.randint(0, action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        emissions_q, agree_q = self.net(state)
        combined_q = (1 - self.chi) * emissions_q + (self.chi) * agree_q

        action_probs = torch.softmax(combined_q / self.temp, dim=1)
        action = torch.multinomial(action_probs, num_samples = 1).item()

        # DECAY TEMPERATURE
        self.temp = max(0.1, self.temp * self.decay_rate)
        return action

    def remember(self, state, action, rew_emissions, rew_agree, next_state, done):
        self.memory.append([state, action, rew_emissions, rew_agree, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rew_e, rew_a, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rew_e = torch.FloatTensor(rew_e)
        rew_a = torch.FloatTensor(rew_a)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # CURRENT Q-VALUES
        emissions_q, agree_q = self.net(states)
        emissions_q = emissions_q.gather(1, actions).squeeze(1)
        agree_q = agree_q.gather(1, actions).squeeze(1)

        # NEXT Q-VALUES
        next_emissions_q, next_agree_q = self.net(next_states)
        next_emissions_q = next_emissions_q.max(1)[0]
        next_agree_q = next_agree_q.max(1)[0]

        # REGRESSION TARGET
        target_e = rew_e
        target_a = rew_a

        # LOSS
        loss_e = nn.MSELoss()(emissions_q, target_e.detach())
        loss_a = nn.MSELoss()(agree_q, target_a.detach())

        self.loss_e_rec  = loss_e
        self.loss_a_rec  = loss_a

        # ZERO ALL
        self.optimiser_shared.zero_grad()
        self.optimiser_emissions.zero_grad()
        self.optimiser_agreeableness.zero_grad()

        # BACKWARD EMISSIONS HEAD, AGREEABLENESS HEAD
        loss_e.backward(retain_graph=True)          # SHARED LAYERS USED TWICE
        loss_a.backward()

        # UPDATE ALL
        self.optimiser_shared.step()
        self.optimiser_emissions.step()
        self.optimiser_agreeableness.step()

        # DECAY EPSILON
        self.epsilon = max(0.05, self.epsilon * self.decay_rate)

# TRAINING
def train(env, agent, episodes):
    total_rewards = []
    total_MSE_losses_e = []
    total_MSE_losses_a = []

    for ep in range (episodes):
        state = env.reset()
        done = False
        total_reward_e, total_reward_a = 0,0

        while (not done):
            action = agent.get_action(state)
            next_state, rew_emissions, rew_agree, done = env.step(action)
            agent.remember(state, action, rew_emissions, rew_agree, next_state, done)
            state = next_state
            total_reward_e += rew_emissions
            total_reward_a += rew_agree

            agent.replay()
        
        print(f"Episode {ep+1}/{episodes} | Reward (e): {total_reward_e:.3f}, Reward (a): {total_reward_a:.3f}, Epsilon: {agent.epsilon:.3f}, Temp: {agent.temp:.3f}, Losses (e,a): ({agent.loss_e_rec:.5f}, {agent.loss_a_rec:.5f})")
        total_rewards.append((total_reward_e + total_reward_a))
        total_MSE_losses_e.append(agent.loss_e_rec)
        total_MSE_losses_a.append(agent.loss_a_rec)

# DEPLOY AGENT 
def deploy_agent(env, agent, chi = 0.5, temperature = 1.0):
    state = env.reset()
    done = False

    while (not done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        emissions_q, agree_q = agent.net(state_tensor)

        combined_q = (1 - chi) * emissions_q + chi * agree_q

        action_probs = torch.softmax(combined_q / temperature, dim = 1)
        action = torch.multinomial(action_probs, num_samples=1).item()
        
        state, _, _, done = env.step(action)

    env.outputTxt()

agent = MultiTaskAgent(state_dim, action_dim, 0.99999)
episodes = 3000
train(env, agent, episodes) 

deploy_agent(env, agent, chi = 0.05, temperature = 0.1)
