import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from torch.distributions import Normal  

sys.path.insert(1, "./build")

#from ABM_env import ClimatePolicyEnv

import cpp_env
print(dir(cpp_env))

env = cpp_env.Environment()
cumulative_reward = 0
emissions_total = 0

done = False

while (not done):
    [obs, reward, done] = env.step(0.1)
    print(f"MEAN EMISSIONS: {obs[0]} , PRICE OF GOODS: {obs[1]}, t = {env.getTime()}")

env.outputTxt()



def set_seed(env, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class SAC(nn.Module):
    def __init__ (self, state_dim, action_dim, gamma=0.99, alpha=1e-3, tau=1e-2,
                    batch_size=64, pi_lr=1e-3, q_lr=1e-3):
        
        super().__init__()

        #Initialise Pi Network Model
        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(),
                                nn.Linear(128, 2*action_dim), nn.Tanh())
        
        #Initialise q1 Network Model
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(),
                                nn.Linear(128, 1))

        #Initialise q2 Network Model
        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(),
                                nn.Linear(128, 1))


        #Set Hyperparameters : discount, entropy coeff., 
        # smooth training param., batch size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        #Init. Memory
        self.memory = []

        #set gradient descent algorithm
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), pi_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1_model.parameters(), q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_model.parameters(), q_lr)

        #Init. Target Networks
        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)


    def predict_actions(self, states):
        #pi model predicts action and log of its probabilities by state

        means, log_stds = self.pi_model(states).chunk(2, dim=-1)
        log_stds = torch.clamp(log_stds, -5, 0)
        dists = Normal(means, torch.exp(log_stds))
        actions = dists.rsample()
        log_probs = dists.log_prob(actions)

        return actions, log_probs

    def get_action(self, state, deterministic = False):
        #predict action by state

        state = torch.FloatTensor(state).unsqueeze(0)

        if deterministic:
            means, _ = self.pi_model(state).chunk(2, dim=-1)
            action = means.squeeze(1).detach().numpy() 
        else:
            action, _ = self.predict_actions(state)
            action = action.squeeze(1).detach().numpy()

        max_step = 0.5
        return np.clip(action, -1, 1) * max_step


    def update_model(self, loss, optimizer, model = None, target_model = None):
        #update given network with given loss
        
        #gradient descent
        loss.backward()

        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
        
        optimizer.step()
        optimizer.zero_grad()

        if model != None and target_model != None:
            for param, target_param in zip(model.parameters(), target_model.parameters()):
                new_target_param = (1 - self.tau) * target_param + self.tau * param
                target_param.data.copy_(new_target_param)

    def fit(self, state, action, reward, done, next_state):
        #one training step for the network models

        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size:

            #sample batch from memory and convert to torch.tensor
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
            
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            dones = torch.FloatTensor(dones).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)

            rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

            #determine RHS of Bellman 
            next_actions, next_log_probs = self.predict_actions(next_states)
            next_states_and_actions = torch.concatenate((next_states, next_actions), dim = 1)
            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q2_target_model(next_states_and_actions)
            next_min_q_values = torch.min(next_q1_values, next_q2_values)
            targets = rewards + self.gamma * (1 - dones) * (next_min_q_values - self.alpha * next_log_probs)

            #update q1 and q2 networks to predict RHS
            states_and_actions = torch.concatenate((states, actions), dim = 1)
            q1_loss = torch.mean((self.q1_model(states_and_actions) - targets.detach()) ** 2)
            q2_loss = torch.mean((self.q2_model(states_and_actions) - targets.detach()) ** 2)
            self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
            self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

            #update pi network so that it minimises q1 and q2 network values
            pred_actions, log_probs = self.predict_actions(states)
            states_and_pred_actions = torch.concatenate((states, pred_actions), dim=1)
            q1_values = self.q1_model(states_and_pred_actions)
            q2_values = self.q2_model(states_and_pred_actions)
            min_q_values = torch.min(q1_values, q2_values)
            pi_loss = - torch.mean(min_q_values - self.alpha * log_probs)
            self.update_model(pi_loss, self.pi_optimizer)

def train(env, agent, episode_n, total_rewards = [], rollout_len = 200):
    #train agent given no. episodes

    for episode in range(episode_n):

        total_reward = 0
        state = env.reset()

        for t in range(rollout_len):

            action = agent.get_action(state, deterministic = True)

            next_state, reward, done, _ = env.step(action)

            agent.fit(state, action, reward, done, next_state)

            total_reward += reward

            state = next_state
        
        print(f"episode: {len(total_rewards)}, total_reward: {total_reward}, Final Emissions: {state[2]}, Action: {action}")
        total_rewards.append(total_reward)
    
    plt.figure()
    plt.plot(total_rewards)
    plt.title('total_rewards')
    plt.grid()
    plt.show()

def rollout_deterministic(env, agent, rollout_len=200, render=True):
    state = env.reset()
    emissions = []
    prices = []
    total_reward = 0

    for t in range(rollout_len):

        action = agent.get_action(state, deterministic=True)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
    
    print(f"Total Reward (det.): {total_reward}")
    if render:
        env.render()

# agent = SAC(state_dim, action_dim, alpha = 0.4, batch_size = 128, gamma = 0.95, pi_lr=1e-4, q_lr=2e-4)
# train(env, agent, episode_n=1000) 

# rollout_deterministic(env, agent)
# env.render()
