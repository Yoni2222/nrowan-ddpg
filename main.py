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
    
    dummy_obs = env.reset()
    state_dim = extract_state(dummy_obs).shape[0]

    # --- ACTION SPACE FIX --- #
    # Only generators flagged redispatchable can be controlled. Setting redispatch
    # on a non-redispatchable generator makes the whole action AMBIGUOUS, so
    # Grid2Op discards it and applies do-nothing -> the agent had zero effect.
    redisp_mask = np.asarray(env.gen_redispatchable, dtype=bool)
    ramp_up = np.asarray(env.gen_max_ramp_up, dtype=np.float32)
    action_dim = int(np.sum(redisp_mask))   # control ONLY redispatchable generators
    max_action = 1.0                         # actor outputs [-1, 1]; scaled by ramp below
    print(f"Controllable (redispatchable) generators: {action_dim} / {env.n_gen}")

    # --- NROWAN hyperparameters (tune these to match the paper) ---
    # sigma_init: initial exploration noise level inside the noisy layers.
    # xi_max: upper bound on the noise-reduction weight. If the D term dominates
    #         the actor loss (watch the printed [policy vs xi*D] split), lower it.
    SIGMA_INIT = 0.5
    XI_MAX = 5.0

    agent = DDPGAgent(state_dim, action_dim, max_action,
                      sigma_init=SIGMA_INIT, xi_max=XI_MAX)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    MAX_EPISODES = 2000
    MAX_STEPS = 100
    BATCH_SIZE = 128
    LOG_EVERY = 50           # print a diagnostics summary every N episodes

    episode_rewards = []
    episode_safety_violations = []
    episode_ambiguous = []   # ambiguous-action count per episode (should be ~0)
    xi_history = []          # NROWAN noise-reduction weight per episode
    noise_history = []       # NROWAN noise magnitude D per episode
    actor_losses = []
    critic_losses = []

    print(f"\nStarting Training Loop for {MAX_EPISODES} episodes on {agent.device}...")
    
    for episode in tqdm(range(MAX_EPISODES), desc="Training Progress"):
        
        # --- THE REAL UNBREAKABLE FIX: Force absolute uniform randomness on every episode --- #
        # We sample a completely random folder index from 0 to 999 independently every time.
        # This completely bypasses Grid2Op's internal sequential loops.
        random_chronic_idx = int(np.random.randint(0, 1000))
        env.set_id(random_chronic_idx)
        # ------------------------------------------------------------------------------------ #
        
        obs = env.reset()
        state = extract_state(obs)
        
        ep_reward = 0
        ep_violations = 0
        ep_ambiguous = 0

        for step in range(MAX_STEPS):
            flat_action = agent.select_action(state, explore=True)

            # Scatter the controlled actions into a full redispatch vector and
            # scale each by that generator's ramp limit (MW). Non-redispatchable
            # generators stay at 0 so the action is never ambiguous.
            full_redispatch = np.zeros(env.n_gen, dtype=np.float32)
            full_redispatch[redisp_mask] = flat_action * ramp_up[redisp_mask]
            g2op_action = env.action_space({"redispatch": full_redispatch})
            
            next_obs, reward, done, info = env.step(g2op_action)
            next_state = extract_state(next_obs)

            # DIAGNOSTIC: did Grid2Op reject the action? After the fix this
            # should stay ~0. If it spikes, the redispatch is still illegal.
            if info.get("is_ambiguous", False) or info.get("is_illegal", False):
                ep_ambiguous += 1

            # --- SAFETY TRACKER & REWARD SHAPING --- #
            safe_rho = np.nan_to_num(next_obs.rho, nan=0.0)
            max_rho = np.max(safe_rho)

            if max_rho >= 1.0:
                ep_violations += 1

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
        episode_ambiguous.append(ep_ambiguous)

        # NROWAN Online Weight Adjustment: update xi (the noise-reduction weight)
        # from this episode's performance. As the policy improves, xi grows and
        # the learned exploration noise is driven down.
        xi = agent.update_noise_weight(ep_reward)
        noise_mag = agent.noise_magnitude()
        xi_history.append(xi)
        noise_history.append(noise_mag)

        # --- PERIODIC DIAGNOSTICS --- #
        if (episode + 1) % LOG_EVERY == 0:
            window = episode_rewards[-LOG_EVERY:]
            avg_reward = float(np.mean(window))
            avg_amb = float(np.mean(episode_ambiguous[-LOG_EVERY:]))
            tqdm.write(
                f"[Ep {episode + 1:>4}] "
                f"avg_reward={avg_reward:8.2f} | "
                f"xi={xi:5.3f} | noise_D={noise_mag:6.4f} | "
                f"ambiguous/ep={avg_amb:4.1f}/{MAX_STEPS}"
            )

    print("\nTraining Completed! Exporting data and weights...")
    torch.save(agent.actor.state_dict(), os.path.join(models_dir, 'actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, 'critic.pth'))
    
    np.savetxt(os.path.join(results_dir, "raw_rewards.txt"), episode_rewards)
    np.savetxt(os.path.join(results_dir, "raw_violations.txt"), episode_safety_violations)
    np.savetxt(os.path.join(results_dir, "critic_losses.txt"), critic_losses)
    np.savetxt(os.path.join(results_dir, "actor_losses.txt"), actor_losses)
    np.savetxt(os.path.join(results_dir, "raw_ambiguous.txt"), episode_ambiguous)
    np.savetxt(os.path.join(results_dir, "xi_history.txt"), xi_history)
    np.savetxt(os.path.join(results_dir, "noise_history.txt"), noise_history)
    
    plot_and_save_metrics(episode_rewards, episode_safety_violations, results_dir)

if __name__ == "__main__":
    main()