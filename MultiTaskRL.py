import torch
import torch.nn as nn
import numpy as np 
import random
from collections import deque
from copy import deepcopy
import sys
import matplotlib.pyplot as plt

# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

# Fixed Seed
ENV_SEED = 42       # for testing deployed agent
NP_SEED = 42

# Parameters
state_dim = 206     # (50 firms * 4) + 6 sector features
action_dim = 10     # discrete action space

# GENERATE RANDOM ENV SEEDS
np.random.seed(NP_SEED)
# SEED_LIST = np.random.randint(0, 5000, size = 3).tolist()
# print("Training Seed:", SEED_LIST)

class ChiEmbedding(nn.Module):
    def __init__(self, embed_dim = 8):
        super().__init__()
        self.chi_embedding = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.embed_dim = embed_dim

    def forward(self, chi):

        chi_out = self.chi_embedding(chi)
    
        return chi_out


# Multi-Task RANKER Q Net.
class MultiTaskQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # self.chi_embed_emissions = ChiEmbedding(embed_dim = 16)
        # self.chi_embed_agree = ChiEmbedding(embed_dim = 16)

        # FIRM FEATURES (2 * 200 [t, t-1])
        self.firm_branch = nn.Sequential(
            nn.Linear(400, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        # SECTOR FEATURES (2 * 6 [t, t-1])
        self.sector_branch = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )

        # SHARED LAYERS
        self.shared_layers = nn.Sequential(
            nn.Linear(128 + 16, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        # EMISSIONS TASK HEAD
        self.emissions_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # AGREEABLENESS TASK HEAD
        self.agreeableness_task_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state, prev_state, chi):
        
        # [200]
        firm_features = state[:, :200]

        # [4]
        sector_features = state[:, 200:]

        prev_firm_features = prev_state[:, :200]
        prev_sector_features = prev_state[:, 200:]

        firm_input = torch.cat([firm_features, prev_firm_features], dim=1)

        sector_input = torch.cat([sector_features, prev_sector_features], dim=1)

        firm_out = self.firm_branch(firm_input)
        sector_out = self.sector_branch(sector_input)

        combined = torch.cat([firm_out, sector_out], dim = 1)
        shared = self.shared_layers(combined)

        # chi_embed_e = self.chi_embed_emissions(chi.unsqueeze(1))
        # chi_embed_a = self.chi_embed_agree(chi.unsqueeze(1))

        # x_emissions = torch.cat([shared, chi_embed_e], dim = 1)
        # x_agree = torch.cat([shared, chi_embed_a], dim = 1)

        emissions_q = self.emissions_head(shared)
        agreeableness_q = self.agreeableness_task_head(shared)

        return emissions_q, agreeableness_q

# AGENT
class MultiTaskAgent:
    def __init__(self, state_dim, action_dim, decay_rate = 0.9985, chi=0.5, temperature = 30.0):
        self.net = MultiTaskQNet(state_dim, action_dim)

        # SEPARATE OPTIMISERS FOR TWO HEADS, AND SHARED LAYERS

        # self.optimiser_emissions = torch.optim.Adam(
        #     list(self.net.emissions_head.parameters()) + 
        #     list(self.net.chi_embed_emissions.parameters()), lr = 5e-5)

        # self.optimiser_agreeableness = torch.optim.Adam(
        #     list(self.net.agreeableness_task_head.parameters()) +
        #     list(self.net.chi_embed_agree.parameters()), lr = 5e-5)

        self.optimiser_emissions = torch.optim.Adam(self.net.emissions_head.parameters(), lr = 5e-5)
        self.optimiser_agreeableness = torch.optim.Adam(self.net.agreeableness_task_head.parameters(), lr = 5e-5)

        self.optimiser_shared = torch.optim.Adam(
            list(self.net.shared_layers.parameters()) + 
            list(self.net.firm_branch.parameters()) + 
            list(self.net.sector_branch.parameters()), lr = 1e-4)

        self.gamma = 0.99                           #Discount Factor
        self.memory = deque(maxlen = 20000)
        self.batch_size = 256                       #Init
        self.chi = chi                              #Chi for Ranker

        self.decay_rate = decay_rate
        self.temp = temperature
        self.temp_min = 0.01

        self.loss_e_rec = 0.0
        self.loss_a_rec = 0.0

        self.warmup_steps = 10000

        # TARGET NETWORK
        self.target_net = deepcopy(self.net)
        self.target_update_freq = 20
        self.update_counter = 0

    def get_action(self, state, prev_state, chi):

        state = torch.FloatTensor(state).unsqueeze(0)
        prev_state = torch.FloatTensor(prev_state).unsqueeze(0)

        emissions_q, agree_q = self.net(state, prev_state, chi)
        combined_q = (1.0 - chi) * emissions_q + (chi) * agree_q

        action_probs = torch.softmax(combined_q / self.temp, dim=1)
        action = torch.multinomial(action_probs, num_samples = 1).item()

        return action

    def remember(self, state, action, rew_emissions, rew_agree, next_state, done, chi, prev_state):
        self.memory.append([state, action, rew_emissions, rew_agree, next_state, done, chi, prev_state])

    def replay(self):
        if len(self.memory) < self.warmup_steps:
            return None, None, None, None

        # GET PAST EXPERIENCES
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rew_e, rew_a, next_states, dones, chis, prev_states = zip(*batch)

        # CONVERT TO TENSOR
        states = torch.FloatTensor(states)
        prev_states = torch.FloatTensor(prev_states)
        actions = torch.LongTensor(actions).unsqueeze(1)            #taken actions
        rew_e = torch.FloatTensor(rew_e)
        rew_a = torch.FloatTensor(rew_a)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        chis = torch.FloatTensor(chis)

        # CURRENT Q-VALUES
        emissions_q, agree_q = self.net(states, prev_states, chi=chis)           #shape: [batch_size, action_dim]

        emissions_q = emissions_q.gather(1, actions).squeeze(1)     #Q-value pred. for taken actions
        agree_q = agree_q.gather(1, actions).squeeze(1)

        # NEXT Q-VALUES (TARGET)
        with torch.no_grad():
            next_emissions_q, next_agree_q = self.target_net(next_states, prev_states, chi=chis)
        next_emissions_q = next_emissions_q.max(1)[0]               # ASSUME OPTIMAL POLICY THEREAFTER
        next_agree_q = next_agree_q.max(1)[0]

        # BELLMAN TARGETS   
        target_e = rew_e + self.gamma * next_emissions_q * (1 - dones)
        target_a = rew_a + self.gamma * next_agree_q * (1 - dones)

        target_e = target_e.clamp(-5.0, 40.0)
        target_a = target_a.clamp(-5.0, 40.0)

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
def train(agent, episodes):
    total_rewards_e, total_rewards_a = [], []
    total_losses_e, total_losses_a = [], []
    avg_pred_q_e, avg_pred_q_a = [], []
    avg_targets_e, avg_targets_a = [], []

    for ep in range (episodes):
        
        env = cpp_env.Environment()
        state = env.reset()
        
        prev_state = np.zeros(state_dim)  #Init Prev State to Zeros
        done = False

        episode_rewards_e, episode_rewards_a = 0, 0

        episode_pred_q_e, episode_pred_q_a = [], []
        episode_targets_e, episode_targets_a = [], []

        # FOR TARGET UPDATE
        agent.update_counter += 1
        if agent.update_counter % agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.net.state_dict())
            print(f"TARGET NET UPDATED @ EPISODE {ep}")

        if ep < 700:
            chi_train = torch.FloatTensor([np.random.uniform(0.075, 0.125)])
        else:
            chi_train = torch.FloatTensor([np.random.uniform(0.1, 0.95)])

        while (not done):

            action = agent.get_action(state, prev_state, chi = chi_train)
            next_state, rew_emissions, rew_agree, done = env.step(action)
            agent.remember(state, action, rew_emissions, rew_agree, next_state, done, chi_train, prev_state)

            # UPDATE STATES
            prev_state = state
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

        if episode_pred_q_e:
            # DECAY TEMPERATURE
            agent.temp = max(agent.temp_min, agent.temp * agent.decay_rate)

        print(f"Episode {ep+1}/{episodes} | Temp: {agent.temp:.3f} | Rew(e,a): ({episode_rewards_e:.2f},{episode_rewards_a:.2f}) | Loss(e,a): ({agent.loss_e_rec:.4f},{agent.loss_a_rec:.4f}), Chi: {chi_train.item():.3f}")
    
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

    torch.save(agent.net.state_dict(), "ranker_agent_weights.pt")

    return total_losses_e, total_losses_a

# DEPLOY AGENT 
def deploy_agent(agent, chi_ = 0.5, temperature = 0.01, scenario = "AVERAGE"):
    newenv = cpp_env.Environment(scenario, ENV_SEED, target = 0.2, chi = chi_)
    state = newenv.reset()
    prev_state = np.zeros(state_dim)
    done = False

    while (not done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0)

        emissions_q, agree_q = agent.net(state_tensor, prev_state_tensor, chi=torch.FloatTensor([chi_]))

        combined_q = (1 - chi_) * emissions_q + chi_ * agree_q

        action_probs = torch.softmax(combined_q / temperature, dim = 1)
        action = torch.multinomial(action_probs, num_samples=1).item()

        prev_state = state
        state, _, _, done = newenv.step(action)

    newenv.outputTxt()

if __name__ == "__main__":

    agent = MultiTaskAgent(state_dim, action_dim)
    episodes = 2500
    losses_e, losses_a = train(agent, episodes) 

    deploy_agent(agent, chi_ = 0.1, temperature = 0.01, scenario = "OPTIMISTIC")
    deploy_agent(agent, chi_ = 0.3, temperature = 0.01, scenario = "OPTIMISTIC")
    deploy_agent(agent, chi_ = 0.5, temperature = 0.01, scenario = "OPTIMISTIC")
    deploy_agent(agent, chi_ = 0.7, temperature = 0.01, scenario = "OPTIMISTIC")
    deploy_agent(agent, chi_ = 0.9, temperature = 0.01, scenario = "OPTIMISTIC")

    deploy_agent(agent, chi_ = 0.1, temperature = 0.01, scenario = "PESSIMISTIC")
    deploy_agent(agent, chi_ = 0.3, temperature = 0.01, scenario = "PESSIMISTIC")
    deploy_agent(agent, chi_ = 0.5, temperature = 0.01, scenario = "PESSIMISTIC")
    deploy_agent(agent, chi_ = 0.7, temperature = 0.01, scenario = "PESSIMISTIC")
    deploy_agent(agent, chi_ = 0.9, temperature = 0.01, scenario = "PESSIMISTIC")

