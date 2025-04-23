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

# Fixed Seed
ENV_SEED = 42

# Parameters
state_dim = 203     # (50 firms * 4) + 3 sector features
action_dim = 10     # discrete action space

# Multi-Task RANKER Q Net.
class MultiTaskQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # FIRM FEATURES
        self.firm_branch = nn.Sequential(
            nn.Linear(200, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        # SECTOR FEATURES
        self.sector_branch = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )

        # SHARED LAYERS
        self.shared_layers = nn.Sequential(
            nn.Linear(128 + 32, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        # EMISSIONS TASK HEAD
        self.emissions_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # AGREEABLENESS TASK HEAD
        self.agreeableness_task_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
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
    def __init__(self, state_dim, action_dim, decay_rate = 0.995, chi=0.5, temperature = 1.0):
        self.net = MultiTaskQNet(state_dim, action_dim)

        # SEPARATE OPTIMISERS FOR TWO HEADS, AND SHARED LAYERS
        self.optimiser_emissions = torch.optim.Adam(self.net.emissions_head.parameters(), lr = 1e-4)
        self.optimiser_agreeableness = torch.optim.Adam(self.net.agreeableness_task_head.parameters(), lr = 1e-4)
        self.optimiser_shared = torch.optim.Adam(self.net.shared_layers.parameters(), lr = 5e-4)

        self.gamma = 0.99                           #Discount Factor
        self.memory = deque(maxlen = 15000)
        self.batch_size = 256                       #Init
        self.chi = chi                              #Chi for Ranker

        self.decay_rate = decay_rate
        self.temp = temperature
        self.temp_min = 0.1

        self.loss_e_rec = 0.0
        self.loss_a_rec = 0.0

        self.warmup_steps = 5000

    def get_action(self, state, chi=None):
        
        if chi is None:
            # BALANCE REWARDS
            chi = chi_train = np.random.beta(a=2, b=3)

        state = torch.FloatTensor(state).unsqueeze(0)
        emissions_q, agree_q = self.net(state)
        combined_q = (1 - chi) * emissions_q + (chi) * agree_q

        action_probs = torch.softmax(combined_q / self.temp, dim=1)
        action = torch.multinomial(action_probs, num_samples = 1).item()

        # DECAY TEMPERATURE
        self.temp = max(self.temp_min, self.temp * self.decay_rate)

        return action

    def remember(self, state, action, rew_emissions, rew_agree, next_state, done):
        self.memory.append([state, action, rew_emissions, rew_agree, next_state, done])

    def replay(self):
        if len(self.memory) < self.warmup_steps:
            return None, None, None, None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rew_e, rew_a, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rew_e = torch.FloatTensor(rew_e)
        rew_a = torch.FloatTensor(rew_a)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # CURRENT Q-VALUES
        emissions_q, agree_q = self.net(states)                     #shape: [batch_size, action_dim]

        emissions_q = emissions_q.gather(1, actions).squeeze(1)     #taken action
        agree_q = agree_q.gather(1, actions).squeeze(1)

        # NEXT Q-VALUES
        next_emissions_q, next_agree_q = self.net(next_states)
        next_emissions_q = next_emissions_q.max(1)[0]
        next_agree_q = next_agree_q.max(1)[0]

        # BELLMAN TARGETS   
        target_e = rew_e + self.gamma * next_emissions_q * (1 - dones)
        target_a = rew_a + self.gamma * next_agree_q * (1 - dones)

        # target_e = target_e.clamp(-1.0, 1.0)
        # target_a = target_a.clamp(-1.0, 1.0)

        # LOSS
        loss_e = nn.MSELoss()(emissions_q, target_e.detach())
        loss_a = nn.MSELoss()(agree_q, target_a.detach())

        self.loss_e_rec  = loss_e.detach().item()
        self.loss_a_rec  = loss_a.detach().item()

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

        return emissions_q.mean().item(), agree_q.mean().item(), target_e.mean().item(), target_a.mean().item()

# TRAINING
def train(env, agent, episodes):
    total_rewards_e, total_rewards_a = [], []
    total_losses_e, total_losses_a = [], []
    avg_pred_q_e, avg_pred_q_a = [], []
    avg_targets_e, avg_targets_a = [], []

    for ep in range (episodes):
        state = env.reset()
        done = False

        episode_rewards_e, episode_rewards_a = 0, 0

        episode_pred_q_e, episode_pred_q_a = [], []
        episode_targets_e, episode_targets_a = [], []

        while (not done):

            # SELECT RANDOM CHI IN TRAINING
            chi_train = np.random.uniform(0, 1)
            action = agent.get_action(state, chi=chi_train)
            next_state, rew_emissions, rew_agree, done = env.step(action)
            agent.remember(state, action, rew_emissions, rew_agree, next_state, done)

            state = next_state
            episode_rewards_e += rew_emissions
            episode_rewards_a += rew_agree

            pred_q_e, pred_q_a, targ_q_e, targ_q_a = agent.replay()

            # TRACK PREDICTIONS AND TARGETS:
            if pred_q_e is not None:
                episode_pred_q_e.append(pred_q_e)
                episode_pred_q_a.append(pred_q_a)
                episode_targets_e.append(targ_q_e)
                episode_targets_a.append(targ_q_a)
            
        #LOG PER EPISODE:
        total_rewards_e.append(episode_rewards_e)
        total_rewards_a.append(episode_rewards_a)
        total_losses_e.append(agent.loss_e_rec)
        total_losses_a.append(agent.loss_a_rec)
        avg_pred_q_e.append(np.mean(episode_pred_q_e) if episode_pred_q_e else 0)
        avg_pred_q_a.append(np.mean(episode_pred_q_a) if episode_pred_q_a else 0)
        avg_targets_e.append(np.mean(episode_targets_e) if episode_targets_e else 0)
        avg_targets_a.append(np.mean(episode_targets_a) if episode_targets_a else 0)

        print(f"Episode {ep+1}/{episodes} | Temp: {agent.temp:.3f} | Rew(e,a): ({episode_rewards_e:.2f},{episode_rewards_a:.2f}) | Loss(e,a): ({agent.loss_e_rec:.4f},{agent.loss_a_rec:.4f})")
    
    # VISUALISE

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(total_losses_e, label='Emissions Loss')
    axs[0, 0].plot(total_losses_a, label='Agreeableness Loss')
    axs[0, 0].set_title('Losses')
    axs[0, 0].legend()

    axs[0, 1].plot(total_rewards_e, label='Emissions Reward')
    axs[0, 1].plot(total_rewards_a, label='Agreeableness Reward')
    axs[0, 1].set_title('Rewards per Episode')
    axs[0, 1].legend()

    axs[1, 0].plot(avg_pred_q_e, label='Avg Pred Q Emissions')
    axs[1, 0].plot(avg_targets_e, label='Avg Target Emissions')
    axs[1, 0].set_title('Emissions Q-Values & Targets')
    axs[1, 0].legend()

    axs[1, 1].plot(avg_pred_q_a, label='Avg Pred Q Agree')
    axs[1, 1].plot(avg_targets_a, label='Avg Target Agree')
    axs[1, 1].set_title('Agreeableness Q-Values & Targets')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return total_losses_e, total_losses_a

# DEPLOY AGENT 
def deploy_agent(agent, chi_ = 0.5, temperature = 1.0):
    newenv = cpp_env.Environment("AVERAGE", ENV_SEED, target = 0.2, chi = chi_)
    state = newenv.reset()

    done = False

    while (not done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        emissions_q, agree_q = agent.net(state_tensor)

        combined_q = (1 - chi_) * emissions_q + chi_ * agree_q

        action_probs = torch.softmax(combined_q / temperature, dim = 1)
        action = torch.multinomial(action_probs, num_samples=1).item()
        
        state, _, _, done = newenv.step(action)

    newenv.outputTxt()

agent = MultiTaskAgent(state_dim, action_dim, 0.99999)
episodes = 3000
losses_e, losses_a = train(env, agent, episodes) 

deploy_agent(agent, chi_ = 0.1, temperature = 0.01)
deploy_agent(agent, chi_ = 0.7, temperature = 0.01)