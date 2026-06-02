import os
import numpy as np
import matplotlib.pyplot as plt

def plot_smoothed_metrics(window_size=50):
    """
    Reads raw training data from Google Drive or local paths,
    applies a moving average filter, and plots the smoothed convergence trends.
    """
    # 1. Define paths (Checks Colab Drive first, falls back to local)
    colab_drive_path = '/content/drive/MyDrive/NROWAN_DDPG_Project/results/'
    local_path = 'results/'
    
    results_dir = colab_drive_path if os.path.exists(colab_drive_path) else local_path
    
    rewards_path = os.path.join(results_dir, "raw_rewards.txt")
    violations_path = os.path.join(results_dir, "raw_violations.txt")
    
    if not os.path.exists(rewards_path) or not os.path.exists(violations_path):
        print(f"Error: Raw data files not found in {results_dir}")
        print("Please ensure your training loop completed and saved raw data files.")
        return

    # 2. Load the raw arrays
    raw_rewards = np.loadtxt(rewards_path)
    raw_violations = np.loadtxt(violations_path)
    
    # 3. Helper function to compute moving average
    def moving_average(data, window):
        # Use convolution to calculate the rolling mean efficiently
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Apply smoothing
    smoothed_rewards = moving_average(raw_rewards, window_size)
    smoothed_violations = moving_average(raw_violations, window_size)
    
    # Adjust x-axis for the rolling window offset
    episodes = range(window_size, len(raw_rewards) + 1)

    # 4. Generate and plot the smoothed charts
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Smoothed Convergence (Rewards)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(raw_rewards) + 1), raw_rewards, color='blue', alpha=0.15, label='Raw Data')
    plt.plot(episodes, smoothed_rewards, color='blue', linewidth=2, label=f'Moving Avg (w={window_size})')
    plt.title('DDPG Smoothed Convergence trend')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Smoothed Safety Violations
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(raw_violations) + 1), raw_violations, color='red', alpha=0.15, label='Raw Data')
    plt.plot(episodes, smoothed_violations, color='red', linewidth=2, label=f'Moving Avg (w={window_size})')
    plt.title('Smoothed Safety Violations Trend (rho >= 1.0)')
    plt.xlabel('Episode')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save the professional version
    save_path = os.path.join(results_dir, 'smoothed_training_metrics.png')
    plt.savefig(save_path, dpi=300) # Save in high resolution for slides
    plt.close()
    
    print(f"Successfully generated and saved professional graphs to: {save_path}")

# Execute the plotter function
if __name__ == "__main__":
    plot_smoothed_metrics(window_size=50)