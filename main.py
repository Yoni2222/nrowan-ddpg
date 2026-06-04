import os
import grid2op
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from env_setup.state_extractor import extract_state
from agent.ddpg_agent import DDPGAgent
from agent.memory import ReplayBuffer

def get_save_paths():
    colab_drive_path = '/content/drive/MyDrive/'
    if os.path.exists(colab_drive_path):
        print("Google Drive detection: SUCCESS. Training data will sync to cloud.")
        base_dir = os.path.join(colab_drive_path, 'NROWAN_DDPG_Project')
        models_dir = os.path.join(base_dir, 'saved_models')
        results_dir = os.path.join(base_dir, 'results')
    else:
        print("Google Drive detection: NOT FOUND. Saving to local project directory...")
        models_dir = "saved_models"
        results_dir = "results"
        
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return models_dir, results_dir

def plot_and_save_metrics(rewards, safety_violations, results_dir):
    episodes = range(1, len(rewards) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, color='blue', linewidth=1.5)
    plt.title('DDPG Convergence (Reward per Episode)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, safety_violations, color='red', linewidth=1.5)
    plt.title('Safety Violations (rho >= 1.0)')
    plt.xlabel('Episode')
    plt.ylabel('Number of Violations')
    plt.grid(True)
    
    plt.tight_layout()
    graph_path = os.path.join(results_dir, 'training_metrics.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"=> Graphs successfully saved to: {graph_path}")

def main():
    models_dir, results_dir = get_save_paths()
    
    print("Initializing Grid2Op environment...")
    env = grid2op.make("rte_case14_realistic")
    
    env.chronics_handler.shuffle()
    
    dummy_obs = env.reset()
    state_dim = extract_state(dummy_obs).shape[0]  
    action_dim = env.n_gen                         
    max_action = 10.0                              

    agent = DDPGAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    MAX_EPISODES = 2000      
    MAX_STEPS = 100         
    BATCH_SIZE = 64

    episode_rewards = []
    episode_safety_violations = []
    actor_losses = []
    critic_losses = []

    print(f"\nStarting Training Loop for {MAX_EPISODES} episodes on {agent.device}...")
    
    for episode in tqdm(range(MAX_EPISODES), desc="Training Progress"):
        
        env.set_id(episode % 1000)
        
        obs = env.reset()
        state = extract_state(obs)
        
        ep_reward = 0
        ep_violations = 0
        
        for step in range(MAX_STEPS):
            flat_action = agent.select_action(state, explore=True)
            g2op_action = env.action_space({"redispatch": flat_action})
            
            next_obs, reward, done, info = env.step(g2op_action)
            next_state = extract_state(next_obs)

            # --- SAFETY TRACKER & REWARD SHAPING --- #
            safe_rho = np.nan_to_num(next_obs.rho, nan=0.0)
            max_rho = np.max(safe_rho)

            if max_rho >= 1.0:
                ep_violations += 1

            # --- PROTECTION B: Reward Scaling & Hard Clipping ---
            # Standardize rewards to a safe interval to prevent mathematical explosion
            if max_rho > 0.8:
                penalty = (max_rho - 0.8) * 10.0
                reward -= penalty
            
            # Clip the modified single-step reward strictly between -10.0 and 2.0
            reward = np.clip(reward, -10.0, 2.0)
            # ----------------------------------------------------- #

            replay_buffer.add(state, flat_action, reward, next_state, done)
            
            if replay_buffer.size > BATCH_SIZE:
                c_loss, a_loss = agent.train(replay_buffer, BATCH_SIZE)
                critic_losses.append(c_loss)
                actor_losses.append(a_loss)
                
            state = next_state
            ep_reward += reward
            
            if done:
                break
                
        episode_rewards.append(ep_reward)
        episode_safety_violations.append(ep_violations)
        agent.noise_model.decay_noise()

    print("\nTraining Completed! Exporting data and weights...")
    torch.save(agent.actor.state_dict(), os.path.join(models_dir, 'actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, 'critic.pth'))
    
    np.savetxt(os.path.join(results_dir, "raw_rewards.txt"), episode_rewards)
    np.savetxt(os.path.join(results_dir, "raw_violations.txt"), episode_safety_violations)
    np.savetxt(os.path.join(results_dir, "critic_losses.txt"), critic_losses)
    np.savetxt(os.path.join(results_dir, "actor_losses.txt"), actor_losses)
    
    plot_and_save_metrics(episode_rewards, episode_safety_violations, results_dir)

if __name__ == "__main__":
    main()