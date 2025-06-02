import torch
import torch.nn as nn
import numpy as np 
import sys
import matplotlib.pyplot as plt 

from SAC_ABM import SACAgent

# GET .so ENV
sys.path.insert(1, "./build")
import cpp_env

# Fixed Seed
ENV_SEED = 42

# Parameters
state_dim = 206     # (50 firms * 4) + 6 sector features
action_dim = 1

#LOAD CRITIC, RANKER, ACTOR
agent = SACAgent(state_dim, action_dim)

agent.critic1.load_state_dict(torch.load("critic1_weights.pt"))
agent.critic1.eval()

agent.critic2.load_state_dict(torch.load("critic2_weights.pt"))
agent.critic2.eval()

agent.actor.load_state_dict(torch.load("actor_policy_weights.pt"))
agent.actor.eval()

agent.ranker.load_state_dict(torch.load("SAC_ranker.pt"))
agent.ranker.eval()

def deploy_agent(agent, chi_ = 0.5, scenario = "AVERAGE", output = True, seed = False):

    if seed:
        newenv = cpp_env.Environment(scenario, ENV_SEED, target = 0.2, chi = chi_)
    else:
        newenv = cpp_env.Environment(scenario, target = 0.2, chi = chi_)

    state = newenv.reset()
    prev_state = np.zeros(state_dim)
    done = False

    # REWARDS
    re = 0
    ra = 0

    while (not done):

        action = agent.actor.get_action(state, prev_state, chi_, deterministic = True)
        next_state, re_ep, ra_ep, done = newenv.step(action)

        prev_state = state
        state = next_state

        re += re_ep
        ra += ra_ep
    
    if output:
        newenv.outputTxt()

    return re, ra

# _, _ = deploy_agent(agent, chi_ = 0.1, scenario = "PESSIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.3, scenario = "PESSIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.5, scenario = "PESSIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.7, scenario = "PESSIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.9, scenario = "PESSIMISTIC", output = True)

# _, _ = deploy_agent(agent, chi_ = 0.1, scenario = "OPTIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.3, scenario = "OPTIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.5, scenario = "OPTIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.7, scenario = "OPTIMISTIC", output = True)
# _, _ = deploy_agent(agent, chi_ = 0.9, scenario = "OPTIMISTIC", output = True)


def evaluate_returns(agent, scenario, chi_values, n_episodes=20):
    means = []
    stds = []
    for chi_ in chi_values:
        episode_returns = []
        for _ in range(n_episodes):
            re, ra = deploy_agent(agent, chi_, scenario, output=False)
            return_ep = (1 - chi_) * re + (chi_) * ra
            episode_returns.append(return_ep)
        means.append(np.mean(episode_returns))
        stds.append(np.std(episode_returns))

    return np.array(means), np.array(stds)

chi_values = np.linspace(0, 1, 50)

returns_opt, stds_opt = evaluate_returns(agent, "OPTIMISTIC", chi_values, n_episodes = 50)
returns_avg, stds_avg = evaluate_returns(agent, "AVERAGE", chi_values, n_episodes = 150)
returns_pes, stds_pes = evaluate_returns(agent, "PESSIMISTIC", chi_values, n_episodes = 50)

plt.figure(figsize=(8,5))
plt.plot(chi_values, returns_opt, '-', color="red", label="Optimistic Scenario", linewidth=2)
#plt.fill_between(chi_values, returns_opt - stds_opt, returns_opt + stds_opt, color="red", alpha=0.18)
plt.plot(chi_values, returns_pes, '-', color="blue", label="Pessimistic Scenario", linewidth=2)
# plt.fill_between(chi_values, returns_pes - stds_pes, returns_pes + stds_pes, color="blue", alpha=0.18)
plt.plot(chi_values, returns_avg, '-', color="green", label="Average Scenario", linewidth=2)
plt.fill_between(chi_values, returns_avg - stds_avg, returns_avg + stds_avg, color="green", alpha=0.18)
plt.xlabel(r'$\chi$', fontsize=25)
plt.ylabel("Average Episode Return", fontsize=25)
plt.legend(fontsize=20)
plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.xlim([0.0,1.0])
plt.ylim([14,30])
plt.show()

def compare_q_vs_return_across_chi(agent, scenario="AVERAGE", chi_values=None, n_episodes=10):
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

            agree_preds = 0
            emiss_preds = 0

            first_ep = True
            gamma = 1

            while not done:
                action = agent.actor.get_action(state, prev_state, chi_, deterministic = True)
                next_state, re_ep, ra_ep, done = newenv.step(action)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                chi_tensor = torch.FloatTensor(np.array(chi_)).unsqueeze(0)
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)

                q1e, q1a = agent.critic1(state_tensor, prev_tensor, chi_tensor, action_tensor)
                q2e, q2a = agent.critic2(state_tensor, prev_tensor, chi_tensor, action_tensor)

                if first_ep:
                    emiss_preds += torch.min(q1e, q2e).item()
                    agree_preds += torch.min(q1a, q2a).item()

                prev_state = state
                state = next_state

                emiss_return += re_ep * gamma
                agree_return += ra_ep * gamma

                gamma *= agent.gamma
                first_ep = False
            
            chi_agree_rets.append(agree_return)
            chi_emiss_rets.append(emiss_return)

            chi_emiss_preds.append(emiss_preds)
            chi_agree_preds.append(agree_preds) 

        pred_qs_agree.append([np.mean(chi_agree_preds), np.std(chi_agree_preds)])
        actual_rets_agree.append([np.mean(chi_agree_rets), np.std(chi_agree_rets)])
        pred_qs_emiss.append([np.mean(chi_emiss_preds), np.std(chi_emiss_preds)])
        actual_rets_emiss.append([np.mean(chi_emiss_rets), np.std(chi_emiss_rets)])

    pred_qs_agree = np.array(pred_qs_agree)
    actual_rets_agree = np.array(actual_rets_agree)
    pred_qs_emiss = np.array(pred_qs_emiss)
    actual_rets_emiss = np.array(actual_rets_emiss)

    # Set up side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    # Plot for Agreeableness
    ax1.plot(chi_values, pred_qs_agree[:,0], color = "green", label='Predicted Q (Agreeableness)')
    ax1.fill_between(chi_values, pred_qs_agree[:,0] - pred_qs_agree[:,1], pred_qs_agree[:,0] + pred_qs_agree[:,1], color = "green", alpha = 0.2)
    ax1.plot(chi_values, actual_rets_agree[:,0],color = "blue", label='Actual Return (Agreeableness)')
    ax1.fill_between(chi_values, actual_rets_agree[:,0] - actual_rets_agree[:,1], actual_rets_agree[:,0] + actual_rets_agree[:,1], color = "blue", alpha = 0.2)
    ax1.set_xlabel(r'$\chi$', fontsize=22)
    ax1.set_ylabel("Episode Returns (Discounted)", fontsize=22)
    ax1.set_title("Agreeableness Task", fontsize=24, fontweight='bold')
    ax1.set_xlim([0,1])
    ax1.set_ylim([5,35])
    ax1.tick_params(labelsize=18)
    ax1.legend(fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot for Emissions
    ax2.plot(chi_values, pred_qs_emiss[:,0], color = "red", label='Predicted Q (Emissions)')
    ax2.fill_between(chi_values, pred_qs_emiss[:,0] - pred_qs_emiss[:,1], pred_qs_emiss[:,0] + pred_qs_emiss[:,1], alpha = 0.2, color = "red")
    ax2.plot(chi_values, actual_rets_emiss[:,0], color = "blue", label='Actual Return (Emissions)')
    ax2.fill_between(chi_values, actual_rets_emiss[:,0] - actual_rets_emiss[:,1], actual_rets_emiss[:,0] + actual_rets_emiss[:,1], color = "blue", alpha = 0.2)
    ax2.set_xlabel(r'$\chi$', fontsize=22)
    ax2.set_title("Emissions Task", fontsize=24, fontweight='bold')
    ax2.set_xlim([0,1])
    ax2.set_ylim([5,35])
    ax2.tick_params(labelsize=18)
    ax2.legend(fontsize=18)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

chi_grid = np.linspace(0, 1, 30)
#compare_q_vs_return_across_chi(agent, scenario="PESSIMISTIC", chi_values=chi_grid, n_episodes=50)

def compare_ranker_vs_actual_return(agent, scenario = "PESSIMISTIC", chi_values = None, n_episodes = 10):
    if chi_values is None:
        chi_values = np.linspace(0, 1, 11)
    
    ranker_preds = []
    actual_returns = []
    linear_preds = []

    for chi_ in chi_values:
        chi_ranker_preds = []
        chi_actual_returns = []
        chi_linear_preds = []

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False

            total_return = 0
            gamma = 1.0
            first_step = True

            while not done:
                action = agent.actor.get_action(state, prev_state, chi_, deterministic=True)
                next_state, re_ep, ra_ep, done = newenv.step(action)

                if first_step:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                    chi_tensor = torch.FloatTensor([chi_]).unsqueeze(0)
                    action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)

                    with torch.no_grad():
                        q1e, q1a = agent.critic1(state_tensor, prev_tensor, chi_tensor, action_tensor)
                        q2e, q2a = agent.critic2(state_tensor, prev_tensor, chi_tensor, action_tensor)
                        RE = torch.min(q1e, q2e)
                        RA = torch.min(q1a, q2a)
                        ranker_pred = agent.ranker(RE, RA, chi_tensor).item()
                        linear_pred = (1 - chi_) * RE.item() + chi_ * RA.item()
                    chi_ranker_preds.append(ranker_pred)
                    chi_linear_preds.append(linear_pred)
                    first_step = False

                total_return += ((1 - chi_) * re_ep + chi_ * ra_ep) * gamma

                prev_state = state
                state = next_state
                gamma *= agent.gamma

            chi_actual_returns.append(total_return)

        ranker_preds.append([np.mean(chi_ranker_preds), np.std(chi_ranker_preds)])
        linear_preds.append([np.mean(chi_linear_preds), np.std(chi_linear_preds)])
        actual_returns.append([np.mean(chi_actual_returns), np.std(chi_actual_returns)])

    ranker_preds = np.array(ranker_preds)
    linear_preds = np.array(linear_preds)
    actual_returns = np.array(actual_returns)

    plt.figure(figsize=(8, 6))
    plt.plot(chi_values, ranker_preds[:,0], label='Ranker Predicted Return', color='purple')
    plt.fill_between(chi_values, ranker_preds[:,0]-ranker_preds[:,1], ranker_preds[:,0]+ranker_preds[:,1], color='purple', alpha=0.2)
    plt.plot(chi_values, linear_preds[:,0], 'r--', label='Linear (RVD) Return')
    plt.fill_between(chi_values, linear_preds[:,0]-linear_preds[:,1], linear_preds[:,0]+linear_preds[:,1], color='red', alpha=0.12)
    plt.plot(chi_values, actual_returns[:,0], label='Actual Return', color='blue')
    plt.fill_between(chi_values, actual_returns[:,0]-actual_returns[:,1], actual_returns[:,0]+actual_returns[:,1], color='blue', alpha=0.2)
    plt.xlabel(r'$\chi$', fontsize=22)
    plt.ylabel("Episode Return (Discounted)", fontsize=22)
    plt.xlim([0,1])
    plt.ylim([5,35])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#chi_grid = np.linspace(0, 1, 30)
#compare_ranker_vs_actual_return(agent, scenario="PESSIMISTIC", chi_values=chi_grid, n_episodes=50)


def interpret_ranker(agent, scenario = "PESSIMISTIC", chi_values = None, n_episodes = 10):
    if chi_values is None:
        chi_values = np.linspace(0, 1, 11)

    for chi_ in chi_values:

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False

            while not done:
                action = agent.actor.get_action(state, prev_state, chi_, deterministic = True)
                next_state, re_ep, ra_ep, done = newenv.step(action)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                chi_tensor = torch.FloatTensor(np.array(chi_)).unsqueeze(0)
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)

                q1e, q1a = agent.critic1(state_tensor, prev_tensor, chi_tensor, action_tensor)
                q2e, q2a = agent.critic2(state_tensor, prev_tensor, chi_tensor, action_tensor)

                RE = torch.min(q1e, q2e)
                RA = torch.min(q1a, q2a)

                prev_state = state
                state = next_state

                return_pred = agent.ranker(RE, RA, chi_tensor)
                chi_ranker = (return_pred.item() - RE.item()) / (RA.item() - RE.item())

def plot_ranker_chi_over_time(agent, scenario="PESSIMISTIC", chi_set=[0.1, 0.3, 0.5, 0.7, 0.9], n_episodes=10):
    import matplotlib.pyplot as plt
    import numpy as np

    for chi_ in chi_set:
        # Collect all episode ranker_chi lists for this chi
        episode_ranker_chis = []

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False
            ranker_chis = []
            steps = 0

            while not done:
                action = agent.actor.get_action(state, prev_state, chi_, deterministic=True)
                next_state, re_ep, ra_ep, done = newenv.step(action)

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                chi_tensor = torch.FloatTensor([chi_]).unsqueeze(0)
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)

                with torch.no_grad():
                    q1e, q1a = agent.critic1(state_tensor, prev_tensor, chi_tensor, action_tensor)
                    q2e, q2a = agent.critic2(state_tensor, prev_tensor, chi_tensor, action_tensor)
                    RE = torch.min(q1e, q2e).item()
                    RA = torch.min(q1a, q2a).item()
                    return_pred = agent.ranker(
                        torch.tensor([[RE]], dtype=torch.float32),
                        torch.tensor([[RA]], dtype=torch.float32),
                        chi_tensor
                    ).item()
                    # / 0:
                    if abs(RA - RE) > 1e-6:
                        ranker_chi = (return_pred - RE) / (RA - RE)
                    else:
                        ranker_chi = float('nan')
                ranker_chis.append(ranker_chi)
                prev_state = state
                state = next_state

            episode_ranker_chis.append(ranker_chis)

        max_len = max(len(lst) for lst in episode_ranker_chis)
        padded = np.full((n_episodes, max_len), np.nan)
        for i, lst in enumerate(episode_ranker_chis):
            padded[i, :len(lst)] = lst

        mean_ranker = np.nanmean(padded, axis=0)
        std_ranker = np.nanstd(padded, axis=0)

        timesteps = np.arange(len(mean_ranker))
        plt.plot(timesteps, mean_ranker, label=f"$\\chi={chi_}$")
        plt.fill_between(timesteps, mean_ranker - std_ranker, mean_ranker + std_ranker, alpha=0.2)

    plt.xlabel("Timestep", fontsize=15)
    plt.ylabel("Ranker Implied $\\chi$", fontsize=15)
    plt.title("Mean Ranker Implied $\\chi$ (±std) over Episode", fontsize=16)
    plt.ylim([-0.2, 1.2])
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_ranker_contours(agent, chi_values=[0.1, 0.3, 0.5, 0.7, 0.9], re_range=(0, 30), ra_range=(0, 30), num=60):
 
    RE_grid, RA_grid = np.meshgrid(np.linspace(*re_range, num), np.linspace(*ra_range, num))

    for chi in chi_values:
        Ranker_out = np.zeros_like(RE_grid)
        for i in range(RE_grid.shape[0]):
            for j in range(RE_grid.shape[1]):
                with torch.no_grad():
                    Ranker_out[i, j] = agent.ranker(
                        torch.tensor([[RE_grid[i, j]]], dtype=torch.float32),
                        torch.tensor([[RA_grid[i, j]]], dtype=torch.float32),
                        torch.tensor([[chi]], dtype=torch.float32)
                    ).item()
        plt.figure(figsize=(7,6))
        cp = plt.contourf(RE_grid, RA_grid, Ranker_out, levels=5)
        plt.xlabel("RE (Q_E)", fontsize=14)
        plt.ylabel("RA (Q_A)", fontsize=14)
        plt.title(f"Ranker Output Contour (chi={chi})", fontsize=16)
        cbar = plt.colorbar(cp)
        cbar.set_label("Predicted Return", fontsize=13)
        plt.tight_layout()
        plt.show()
    
def plot_ranker_chi_sensitivity(agent, chi_values=[0.1, 0.3, 0.5, 0.9], re_range=(0, 30), ra_range=(0, 30), num=60, epsilon=1e-3):

    RE_grid, RA_grid = np.meshgrid(np.linspace(*re_range, num), np.linspace(*ra_range, num))

    for chi in chi_values:
        sensitivity = np.zeros_like(RE_grid)
        for i in range(RE_grid.shape[0]):
            for j in range(RE_grid.shape[1]):
                RE = RE_grid[i, j]
                RA = RA_grid[i, j]
                with torch.no_grad():
                    f_plus = agent.ranker(
                        torch.tensor([[RE]], dtype=torch.float32),
                        torch.tensor([[RA]], dtype=torch.float32),
                        torch.tensor([[chi + epsilon]], dtype=torch.float32)
                    ).item()
                    f_minus = agent.ranker(
                        torch.tensor([[RE]], dtype=torch.float32),
                        torch.tensor([[RA]], dtype=torch.float32),
                        torch.tensor([[chi - epsilon]], dtype=torch.float32)
                    ).item()
                dG_dchi = (f_plus - f_minus) / (2 * epsilon)
                sensitivity[i, j] = dG_dchi

        plt.figure(figsize=(7,6))
        cp = plt.contourf(RE_grid, RA_grid, sensitivity, cmap='coolwarm', levels=30)
        plt.xlabel("RE (Q_E)", fontsize=14)
        plt.ylabel("RA (Q_A)", fontsize=14)
        plt.title(f"Ranker $\\partial G / \\partial \\chi$ (chi={chi})", fontsize=16)
        cbar = plt.colorbar(cp)
        cbar.set_label("dG/dchi", fontsize=13)
        plt.tight_layout()
        plt.show()

def plot_ranker_surface_3d(agent, chi_values=[0.1, 0.3, 0.5, 0.7, 0.9], re_range=(0, 30), ra_range=(0, 30), num=50):
 
    from mpl_toolkits.mplot3d import Axes3D

    RE = np.linspace(*re_range, num)
    RA = np.linspace(*ra_range, num)
    X, Y = np.meshgrid(RE, RA)

    for chi in chi_values:
        Z = np.zeros_like(X)
        Z_linear = (1 - chi) * X + chi * Y
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                with torch.no_grad():
                    Z[i, j] = agent.ranker(
                        torch.tensor([[X[i, j]]], dtype=torch.float32),
                        torch.tensor([[Y[i, j]]], dtype=torch.float32),
                        torch.tensor([[chi]], dtype=torch.float32)
                    ).item()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='black', alpha=0.9)
        ax.plot_wireframe(X, Y, Z_linear, color='red', linewidth=1.2, alpha=0.85, label='Linear Blend')
        
        ax.set_xlabel('Predicted Emissions Return ' + r'$Q_{E}$' , fontsize=16, labelpad=12)
        ax.set_ylabel('Predicted Agree. Return ' + r'$Q_{A}$' , fontsize=16, labelpad=12)
        ax.set_zlabel(f'Predicted Return (chi = {chi})', fontsize=16, labelpad=12)
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.10, label='Predicted Return')

        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

def plot_ranker_policy_relative_sensitivity(
    agent,
    scenario="PESSIMISTIC",
    chi_values=None,
    n_episodes=20,
    steps_per_ep=1,
    epsilon=1e-4
):
    if chi_values is None:
        chi_values = np.linspace(0.005, 0.995, 50)  

    # For statistics
    chi_bins = []
    rel_grads_QE = []
    rel_grads_QA = []
    rel_grads_chi = []

    for chi_ in chi_values:
        chi_bin_QE, chi_bin_QA, chi_bin_chi = [], [], []

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False

            for step in range(steps_per_ep):  # Only first step if steps_per_ep=1
                action = agent.actor.get_action(state, prev_state, chi_, deterministic=True)
                next_state, re_ep, ra_ep, done = newenv.step(action)

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0)
                chi_tensor = torch.FloatTensor(np.array([chi_])).unsqueeze(0)
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)

                with torch.no_grad():
                    q1e, q1a = agent.critic1(state_tensor, prev_tensor, chi_tensor, action_tensor)
                    q2e, q2a = agent.critic2(state_tensor, prev_tensor, chi_tensor, action_tensor)
                RE = torch.min(q1e, q2e).item()
                RA = torch.min(q1a, q2a).item()

                # Compute finite differences for relative gradients
                def fd(x, delta, idx):
                    inp = [torch.tensor([[RE]], dtype=torch.float32), torch.tensor([[RA]], dtype=torch.float32), torch.tensor([[chi_]], dtype=torch.float32)]
                    inp[idx] = inp[idx] + delta
                    return agent.ranker(inp[0], inp[1], inp[2]).item()

                # Ranker value
                with torch.no_grad():
                    G = agent.ranker(
                        torch.tensor([[RE]], dtype=torch.float32),
                        torch.tensor([[RA]], dtype=torch.float32),
                        torch.tensor([[chi_]], dtype=torch.float32)
                    ).item()

                # Avoid div by zero for relative grad
                if abs(G) < 1e-6:
                    continue

                # Q_E
                dG_dQE = (fd(RE, epsilon, 0) - fd(RE, -epsilon, 0)) / (2 * epsilon)
                rel_grad_QE = dG_dQE * RE / G if abs(RE) > 1e-6 else 0.0

                # Q_A
                dG_dQA = (fd(RA, epsilon, 1) - fd(RA, -epsilon, 1)) / (2 * epsilon)
                rel_grad_QA = dG_dQA * RA / G if abs(RA) > 1e-6 else 0.0

                # chi
                dG_dchi = (fd(chi_, epsilon, 2) - fd(chi_, -epsilon, 2)) / (2 * epsilon)
                rel_grad_chi = dG_dchi * chi_ / G if abs(chi_) > 1e-6 else 0.0

                if (step == 29):
                    chi_bin_QE.append(rel_grad_QE)
                    chi_bin_QA.append(rel_grad_QA)
                    chi_bin_chi.append(rel_grad_chi)

                prev_state = state
                state = next_state
                if steps_per_ep == 1:
                    break

        # Store
        rel_grads_QE.append([np.mean(chi_bin_QE), np.std(chi_bin_QE)])
        rel_grads_QA.append([np.mean(chi_bin_QA), np.std(chi_bin_QA)])
        rel_grads_chi.append([np.mean(chi_bin_chi), np.std(chi_bin_chi)])
        chi_bins.append(chi_)

    rel_grads_QE = np.array(rel_grads_QE)
    rel_grads_QA = np.array(rel_grads_QA)
    rel_grads_chi = np.array(rel_grads_chi)
    chi_bins = np.array(chi_bins)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(chi_bins, rel_grads_QE[:,0], label=r"$\eta_{Q_{E}}$", color='red')
    plt.fill_between(chi_bins, rel_grads_QE[:,0]-rel_grads_QE[:,1], rel_grads_QE[:,0]+rel_grads_QE[:,1], color='red', alpha=0.2)
    plt.plot(chi_bins, rel_grads_QA[:,0], label=r"$\eta_{Q_{A}}$", color='green')
    plt.fill_between(chi_bins, rel_grads_QA[:,0]-rel_grads_QA[:,1], rel_grads_QA[:,0]+rel_grads_QA[:,1], color='green', alpha=0.2)
    plt.plot(chi_bins, rel_grads_chi[:,0], label=r"$\eta_{\chi}$", color='black')
    plt.fill_between(chi_bins, rel_grads_chi[:,0]-rel_grads_chi[:,1], rel_grads_chi[:,0]+rel_grads_chi[:,1], color='black', alpha=0.15)
    plt.xlabel(r"$\chi$", fontsize=18)
    plt.ylabel("Mean Relative Gradient", fontsize=18)
    plt.legend(loc = "upper left", fontsize=24)
    plt.tight_layout()
    plt.xlim([0,1])
    plt.ylim([-0.8,1.4])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(linestyle = "--")
    plt.show()

#plot_ranker_policy_relative_sensitivity(agent, scenario="OPTIMISTIC", n_episodes=10, steps_per_ep=30)

def plot_policy_action_trajectories(
    agent,
    scenario="PESSIMISTIC",
    chi_values=None,
    n_episodes=30,
    max_steps=30,
    action_dim=1
):
    if chi_values is None:
        chi_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    plt.figure(figsize=(10, 6))

    for chi_ in chi_values:
        all_actions = []

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            episode_actions = []
            done = False

            while not done:
                action = agent.actor.get_action(state, prev_state, chi_, deterministic=True)
                episode_actions.append(action)

                next_state, _, _, done = newenv.step(action)
                prev_state = state
                state = next_state

            all_actions.append(np.stack(episode_actions))

        all_actions = np.stack(all_actions)

        if action_dim == 1 or (len(all_actions.shape) == 2):
            mean_actions = np.nanmean(all_actions, axis=0)
            std_actions = np.nanstd(all_actions, axis=0)
            plt.plot(mean_actions, label=f"$\chi$={chi_:.2f}")
            plt.fill_between(np.arange(max_steps), mean_actions - std_actions, mean_actions + std_actions, alpha=0.15)
        else:
            for adim in range(action_dim):
                mean_actions = np.nanmean(all_actions[:,:,adim], axis=0)
                std_actions = np.nanstd(all_actions[:,:,adim], axis=0)
                plt.plot(mean_actions, label=f"$\chi$={chi_:.2f}, dim {adim}")
                plt.fill_between(np.arange(max_steps), mean_actions - std_actions, mean_actions + std_actions, alpha=0.1)

    # plt.xlabel("Policy Timestep", fontsize=20)
    # plt.ylabel(r'$\mu_a$', fontsize=20)
    # plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.grid(linestyle="--", alpha=0.4)
    plt.xlim([0,30])
    plt.ylim([-0.2, 0.3])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

#plot_policy_action_trajectories(agent, scenario="OPTIMISTIC", chi_values=[0.1, 0.3, 0.5, 0.7, 0.9], n_episodes=100, max_steps=30, action_dim=1)
#plot_policy_action_trajectories(agent, scenario="PESSIMISTIC", chi_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], n_episodes=100, max_steps=30, action_dim=1)

def plot_policy_chi_sensitivity_by_chi(
    agent,
    scenario="PESSIMISTIC",
    chi_values=None,
    n_episodes=30,
    max_steps=30,
    action_dim=1,
    epsilon=1e-3,
    relative=True
):
    if chi_values is None:
        chi_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    plt.figure(figsize=(10,6))

    for chi_ in chi_values:
        chi_sensitivities = []

        for ep in range(n_episodes):
            newenv = cpp_env.Environment(scenario, target=0.2, chi=chi_)
            state = newenv.reset()
            prev_state = np.zeros(state_dim)
            done = False

            sensitivities = []
            steps = 0

            while not done and steps < max_steps:
                # Compute action at chi
                action0 = agent.actor.get_action(state, prev_state, chi_, deterministic=True)
                action_plus = agent.actor.get_action(state, prev_state, chi_ + epsilon, deterministic=True)
                action_minus = agent.actor.get_action(state, prev_state, chi_ - epsilon, deterministic=True)

                dact_dchi = (np.array(action_plus) - np.array(action_minus)) / (2 * epsilon)

                if action_dim == 1 or np.isscalar(action0):
                    dact_dchi_val = dact_dchi
                    act0 = action0
                else:
                    dact_dchi_val = np.linalg.norm(dact_dchi)
                    act0 = np.linalg.norm(action0)

                if relative:
                    rel_sens = dact_dchi_val * chi_ / (act0 + 1e-8) if abs(act0) > 1e-8 else 0.0
                else:
                    rel_sens = dact_dchi_val

                sensitivities.append(rel_sens)

                next_state, _, _, done = newenv.step(action0)
                prev_state = state
                state = next_state
                steps += 1

            while len(sensitivities) < max_steps:
                sensitivities.append(np.nan)

            chi_sensitivities.append(sensitivities)

        chi_sensitivities = np.array(chi_sensitivities)
        mean_sens = np.nanmean(chi_sensitivities, axis=0)
        std_sens = np.nanstd(chi_sensitivities, axis=0)

        plt.plot(mean_sens, label=rf"$\chi$={chi_:.2f}")
        plt.fill_between(np.arange(max_steps), mean_sens - std_sens, mean_sens + std_sens, alpha=0.15)

    plt.xlabel("Policy Timestep", fontsize=20)
    plt.ylabel(r'$\frac{\partial \mu_a}{\partial \chi}$', fontsize=27)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0,30])
    plt.ylim([-3.5, 1.0])
    plt.show()


# plot_policy_chi_sensitivity_by_chi(
#     agent,
#     scenario="AVERAGE",
#     chi_values=[0.1, 0.3, 0.5, 0.7, 0.9],
#     n_episodes=150,
#     max_steps=30,
#     action_dim=1,
#     epsilon=1e-4,
#     relative=False
# )