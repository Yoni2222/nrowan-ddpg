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
    """Detects if Google Drive is mounted and returns the appropriate paths."""
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
    """Generates and saves convergence and safety graphs."""
    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Convergence Graph (Rewards)
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, color='blue', linewidth=1.5)
    plt.title('DDPG Convergence (Reward per Episode)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    
    # 2. Safety Tracking Graph (Constraint Violations)
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
    # 1. Setup Save Paths (Local or Cloud)
    models_dir, results_dir = get_save_paths()
    
    print("Initializing Grid2Op environment...")
    env = grid2op.make("rte_case14_realistic")
    
    # Shuffle the underlying dataset folders once at the start to break deterministic repetition
    env.chronics_handler.shuffle()
    
    # Determine dimensions
    dummy_obs = env.reset()
    state_dim = extract_state(dummy_obs).shape[0]  # Should be 36
    action_dim = env.n_gen                         # We control the generators
    max_action = 10.0                              # Max MW change per step

    # 2. Initialize Agent and Memory
    agent = DDPGAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Production Hyperparameters for Google Colab GPU
    MAX_EPISODES = 2000      # Set strictly to 2000 episodes as requested
    MAX_STEPS = 100         # Max steps per episode
    BATCH_SIZE = 64

    # Trackers for our plots
    episode_rewards = []
    episode_safety_violations = []

    print(f"\nStarting Training Loop for {MAX_EPISODES} episodes on {agent.device}...")
    
    # 3. The Main Training Loop
    for episode in tqdm(range(MAX_EPISODES), desc="Training Progress"):
        
        # --- THE ABSOLUTE FIX: Cycle to the next shuffled chronic seamlessly --- #
        # First episode relies on initial reset, subsequent episodes advance automatically
        if episode > 0:
            env.step_to_next_chronic()
        # ------------------------------------------------------------------------ #
        
        obs = env.reset()
        state = extract_state(obs)
        
        ep_reward = 0
        ep_violations = 0
        
        for step in range(MAX_STEPS):
            # Select action with NROWAN parameter noise exploration
            flat_action = agent.select_action(state, explore=True)
            g2op_action = env.action_space({"redispatch": flat_action})
            
            # Take a step in the environment
            next_obs, reward, done, info = env.step(g2op_action)
            next_state = extract_state(next_obs)

            # --- SAFETY TRACKER & REWARD SHAPING --- #
            safe_rho = np.nan_to_num(next_obs.rho, nan=0.0)
            max_rho = np.max(safe_rho)

            # Track hard constraints violations for our analysis plots
            if max_rho >= 1.0:
                ep_violations += 1

            # Penalize the agent proportionally if any line exceeds 80% capacity
            # This guides the gradient smoothly before a total collapse occurs
            if max_rho > 0.8:
                penalty = (max_rho - 0.8) * 10.0
                reward -= penalty
            # ---------------------------------------- #

            # Store the modified, safety-guided reward into the experience replay
            replay_buffer.add(state, flat_action, reward, next_state, done)
            
            # Train the agent
            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)
                
            state = next_state
            ep_reward += reward
            
            if done:
                break
                
        episode_rewards.append(ep_reward)
        episode_safety_violations.append(ep_violations)
        
        # Decay the NROWAN noise after each episode
        agent.noise_model.decay_noise()

    # 4. Save the Final Results and Models to the designated directory
    print("\nTraining Completed! Exporting data and weights...")
    
    # Save weights
    torch.save(agent.actor.state_dict(), os.path.join(models_dir, 'actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, 'critic.pth'))
    print(f"=> Weights successfully saved to: {models_dir}")
    
    # Save raw arrays
    np.savetxt(os.path.join(results_dir, "raw_rewards.txt"), episode_rewards)
    np.savetxt(os.path.join(results_dir, "raw_violations.txt"), episode_safety_violations)
    
    # Generate plots
    plot_and_save_metrics(episode_rewards, episode_safety_violations, results_dir)

if __name__ == "__main__":
    main()