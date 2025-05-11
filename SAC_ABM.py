import torch
import torch.nn as nn
import numpy as np 
import random
from collections import deque
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
from torch.distributions import Normal  

sys.path.insert(1, "./build")

# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

# Fixed Seed
ENV_SEED = 42

# Parameters
state_dim = 203     # (50 firms * 4) + 3 sector features
action_dim = 1

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # FIRM FEATURES (2 * 200 [t, t-1])
        self.firm_branch = nn.Sequential(
            nn.Linear(400, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        # SECTOR FEATURES (2 * 3 [t, t-1] + chi + action)
        self.sector_branch = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )

        # SHARED LAYERS
        self.shared_layers = nn.Sequential(
            nn.Linear(128 + 16, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        # EMISSIONS TASK HEAD
        self.emissions_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # AGREEABLENESS TASK HEAD
        self.agreeableness_task_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, prev_state, chi, action):
        
        # [200]
        firm_features = state[:, :200]

        # [3]
        sector_features = state[:, 200:]

        prev_firm_features = prev_state[:, :200]
        prev_sector_features = prev_state[:, 200:]

        firm_input = torch.cat([firm_features, prev_firm_features], dim=1)

        if chi.dim() == 1:
            chi = chi.unsqueeze(1)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        sector_input = torch.cat([sector_features, prev_sector_features, chi, action], dim=1)

        firm_out = self.firm_branch(firm_input)
        sector_out = self.sector_branch(sector_input)

        combined = torch.cat([firm_out, sector_out], dim = 1)
        shared = self.shared_layers(combined)

        emissions_q = self.emissions_head(shared)
        agreeableness_q = self.agreeableness_task_head(shared)

        return emissions_q, agreeableness_q

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit=0.3):
        super().__init__()
        # SAME STATE ENCODING (without action)

        self.firm_branch = nn.Sequential(
            nn.Linear(400, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        # -1 for no action input
        self.sector_branch = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )

        # shared: (128 + 16) → 256
        self.shared = nn.Sequential(
            nn.Linear(128 + 16, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

        # output Gaussian params: 2*action_dim
        self.mu_logstd = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2*action_dim), nn.Tanh()
        )

        self.action_limit = action_limit

    def forward(self, state, prev_state, chi):
        # state, prev_state: [B,203]; chi: [B,1]

         # [200]
        firm_features = state[:, :200]

        # [3]
        sector_features = state[:, 200:]

        prev_firm_features = prev_state[:, :200]
        prev_sector_features = prev_state[:, 200:]

        firm_input = torch.cat([firm_features, prev_firm_features], dim=1)

        if chi.dim() == 1:
            chi = chi.unsqueeze(1)
        sector_input = torch.cat([sector_features, prev_sector_features, chi], dim=1)

        firm_out = self.firm_branch(firm_input)
        sector_out = self.sector_branch(sector_input)

        combined = torch.cat([firm_out, sector_out], dim = 1)
        shared = self.shared(combined)

        mu, logstd = self.mu_logstd(shared).chunk(2, dim=-1)
        logstd = torch.clamp(logstd, -5, 2)
        std = torch.exp(logstd)
        dist = Normal(mu, std)

        return dist

    def get_action(self, state, prev_state, chi, deterministic=False):
        
        state = torch.FloatTensor(state).unsqueeze(0)
        prev_state = torch.FloatTensor(prev_state).unsqueeze(0)

        dist = self.forward(state, prev_state, chi)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        # scale to action_limit
        return torch.tanh(action).item() * self.action_limit, dist.log_prob(action).sum(-1, keepdim=True)

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, alpha=0.2, tau=0.005, buffer_size=20000, batch_size=256):

        # NETWORKS
        self.actor  = SACActor(state_dim, action_dim)
        self.critic1 = SACCritic(state_dim, action_dim)
        self.critic2 = SACCritic(state_dim, action_dim)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)

        # OPTIMISER
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # SEPARATE OPTIMISERS FOR TWO HEADS, AND SHARED LAYERS
        self.optimiser_emissions1 = torch.optim.Adam(self.critic1.emissions_head.parameters(), lr=5e-5)
        self.optimiser_agreeableness1 = torch.optim.Adam(self.critic1.agreeableness_task_head.parameters(), lr=5e-5)
        self.optimiser_shared1 = torch.optim.Adam(self.critic1.shared_layers.parameters(), lr=2.5e-4)

        self.optimiser_emissions2 = torch.optim.Adam(self.critic2.emissions_head.parameters(), lr=5e-5)
        self.optimiser_agreeableness2 = torch.optim.Adam(self.critic2.agreeableness_task_head.parameters(), lr=5e-5)
        self.optimiser_shared2 = torch.optim.Adam(self.critic2.shared_layers.parameters(), lr=2.5e-4)


        # HYPERPARAMETERS
        self.gamma = gamma
        self.alpha = alpha
        self.tau   = tau
        self.batch_size = batch_size

        # REPLAY BUFFER
        self.buffer = deque(maxlen=buffer_size)

        self.warmup_steps = 10000

    def remember(self, state, prev_state, chi, action, rew_e, rew_a, next_state, done):
        self.buffer.append((state, prev_state, chi, action, rew_e, rew_a, next_state, done))

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)

    def replay(self): 
        
        if len(self.buffer) < self.warmup_steps:
            return None
        
        # 1 - GET PAST EXPERIENCES
        batch = random.sample(self.buffer, self.batch_size)
        states, prev_states, chis, actions, rew_es, rew_as, next_states, dones = zip(*batch)

        # CONVERT TO TENSOR
        states = torch.FloatTensor(states)
        prev_states = torch.FloatTensor(prev_states)
        actions = torch.LongTensor(actions)
        rew_es = torch.FloatTensor(rew_es)
        rew_as = torch.FloatTensor(rew_as)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        chis = torch.FloatTensor(chis)
        
        # 2 - CRITIC TARGETS
        with torch.no_grad():
            next_actions, logp_next_actions = self.actor(next_states, states, chis)
            q1e_next, q1a_next = self.critic1_target(next_states, states, chis, next_actions)
            q2e_next, q2a_next = self.critic2_target(next_states, states, chis, next_actions)

            qe_next = torch.min(q1e_next, q2e_next)
            qa_next = torch.min(q1a_next, q2a_next)

            target_e = rew_es + self.gamma * (qe_next - self.alpha * logp_next_actions) * (1 - dones)
            target_a = rew_as + self.gamma * (qa_next - self.alpha * logp_next_actions) * (1 - dones)

        
        # 3 - CURRENT Q
        q1e, q1a = self.critic1(states, prev_states, chis, actions)
        q2e, q2a = self.critic2(states, prev_states, chis, actions)

        # 4 - CRITIC LOSSES
        loss_q1_e = nn.MSELoss()(q1e, target_e.detach())
        loss_q1_a = nn.MSELoss()(q1a, target_a.detach())

        loss_q2_e = nn.MSELoss()(q2e, target_e.detach())
        loss_q2_a = nn.MSELoss()(q2a, target_a.detach())

        self.loss1_e_rec  = loss_q1_e.detach().item()
        self.loss1_a_rec  = loss_q1_a.detach().item()

        self.loss2_e_rec  = loss_q2_e.detach().item()
        self.loss2_a_rec  = loss_q2_a.detach().item()
        
        # ZERO ALL
        self.optimiser_shared1.zero_grad()
        self.optimiser_emissions1.zero_grad()
        self.optimiser_agreeableness1.zero_grad()

        self.optimiser_shared2.zero_grad()
        self.optimiser_emissions2.zero_grad()
        self.optimiser_agreeableness2.zero_grad()

        loss_q1_e.backward(retain_graph=True)       # SHARED LAYERS USED TWICE
        loss_q1_a.backward()

        loss_q2_e.backward(retain_graph=True)
        loss_q2_a.backward()

        # UPDATE CRITICS
        self.optimiser_shared1.step()
        self.optimiser_emissions1.step()
        self.optimiser_agreeableness1.step()

        self.optimiser_shared2.step()
        self.optimiser_emissions2.step()
        self.optimiser_agreeableness2.step()

        # 5 - ACTOR LOSS
        actions_pi, logp_actions_pi = self.actor.get_action(states, prev_states, chis)
        qe_pi1, qa_pi1 = self.critic1.forward_critic(states, prev_states, chis, actions_pi)
        qe_pi2, qa_pi2 = self.critic2.forward_critic(states, prev_states, chis, actions_pi)

        qe_pi = torch.min(qe_pi1, qe_pi2)
        qa_pi = torch.min(qa_pi1, qa_pi2)

        q_comb_pi = (1 - chis) * qe_pi + chis * qa_pi
        loss_actor = (self.alpha * logp_actions_pi - q_comb_pi).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        # 6 - SOFT UPDATE CRITICS
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return {
        "loss_q1_e": self.loss1_e_rec,
        "loss_q1_a": self.loss1_a_rec,
        "loss_q2_e": self.loss2_e_rec,
        "loss_q2_a": self.loss2_a_rec,
        "loss_actor": loss_actor.item(),
        "alpha": self.alpha,
    }



def train(env, agent, episodes):
    total_rewards_e, total_rewards_a = [], []
    total_losses_e, total_losses_a = [], []
    avg_pred_q_e, avg_pred_q_a = [], []
    avg_targets_e, avg_targets_a = [], []
    actor_losses, alpha_values = [], []

    for ep in range(episodes):
        state = env.reset()
        prev_state = np.zeros(203)
        done = False

        episode_rewards_e, episode_rewards_a = 0, 0
        episode_pred_q_e, episode_pred_q_a = [], []
        episode_targets_e, episode_targets_a = [], []

        while not done:
            chi_train = torch.FloatTensor([np.random.uniform(0.05, 0.95)])

            action, _ = agent.actor.get_action(state, prev_state, chi_train)

            next_state, rew_emissions, rew_agree, done = env.step(action)

            agent.remember(state, prev_state, chi_train, action, rew_emissions, rew_agree, next_state, done)

            prev_state = state
            state = next_state

            episode_rewards_e += rew_emissions
            episode_rewards_a += rew_agree

            replay_out = agent.replay()

            if replay_out is not None:
                episode_pred_q_e.append(replay_out["loss_q1_e"])
                episode_pred_q_a.append(replay_out["loss_q1_a"])
                episode_targets_e.append(replay_out["loss_q2_e"])
                episode_targets_a.append(replay_out["loss_q2_a"])
                actor_losses.append(replay_out["loss_actor"])
                alpha_values.append(replay_out["alpha"])

        # Episode-level logging
        total_rewards_e.append(episode_rewards_e)
        total_rewards_a.append(episode_rewards_a)
        total_losses_e.append(np.mean(episode_pred_q_e) if episode_pred_q_e else 0)
        total_losses_a.append(np.mean(episode_pred_q_a) if episode_pred_q_a else 0)
        avg_pred_q_e.append(np.mean(episode_pred_q_e) if episode_pred_q_e else 0)
        avg_pred_q_a.append(np.mean(episode_pred_q_a) if episode_pred_q_a else 0)
        avg_targets_e.append(np.mean(episode_targets_e) if episode_targets_e else 0)
        avg_targets_a.append(np.mean(episode_targets_a) if episode_targets_a else 0)

        print(f"Episode {ep+1}/{episodes} | Rew(e,a): ({episode_rewards_e:.2f},{episode_rewards_a:.2f}) "
              f"| Loss(e,a): ({total_losses_e[-1]:.4f},{total_losses_a[-1]:.4f}) | Alpha: {agent.alpha:.4f}")

    # PLOTS
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    axs[0, 0].plot(total_losses_e, label='Emissions Loss')
    axs[0, 0].plot(total_losses_a, label='Agreeableness Loss')
    axs[0, 0].set_title('Critic Losses')
    axs[0, 0].legend()

    axs[0, 1].plot(total_rewards_e, label='Emissions Reward')
    axs[0, 1].plot(total_rewards_a, label='Agreeableness Reward')
    axs[0, 1].set_title('Episode Rewards')
    axs[0, 1].legend()

    axs[1, 0].plot(avg_pred_q_e, label='Q_pred Emissions')
    axs[1, 0].plot(avg_targets_e, label='Q_target Emissions')
    axs[1, 0].set_title('Emissions Q-Values')
    axs[1, 0].legend()

    axs[1, 1].plot(avg_pred_q_a, label='Q_pred Agreeableness')
    axs[1, 1].plot(avg_targets_a, label='Q_target Agreeableness')
    axs[1, 1].set_title('Agreeableness Q-Values')
    axs[1, 1].legend()

    axs[2, 0].plot(actor_losses, label='Actor Loss')
    axs[2, 0].set_title('Actor Loss')
    axs[2, 0].legend()

    axs[2, 1].plot(alpha_values, label='Alpha Value')
    axs[2, 1].set_title('Alpha (Entropy Weight)')
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    env = cpp_env.Environment()
    agent = SACAgent(state_dim, action_dim)
    episodes = 400
    train(env, agent, episodes) 