import os
import configparser
import numpy as np
import PacmanAgent
from matplotlib import pyplot as plt
from dyna_q import DynaQ
from utils import get_level


def plot_rewards(rewards, title="Rewards per Episode"):
    """
    Plots the rewards over episodes.

    Parameters:
        rewards (list or np.ndarray): Array of rewards for each episode.
        title (str): Title of the plot.
    """
    # Ensure rewards are in NumPy array format
    rewards = np.array(rewards)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot rewards
    plt.plot(rewards, label='Reward per Episode', color='blue', alpha=0.7)

    # Compute and plot a trendline (using a moving average)
    window_size = max(1, len(rewards) // 20)  # 5% of total episodes as window size
    moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), moving_average, label='Moving Average', color='orange', linewidth=2)

    # Add labels and title
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)

    # Add legend
    plt.legend()

    # Show grid
    plt.grid(alpha=0.3)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    result = config.read(os.path.join(os.path.dirname(__file__), 'hyperparameters.ini'))
    PacmanAgent.level_name = config['normal']['level']
    PacmanAgent.level = get_level(PacmanAgent.level_name)
    epochs = int(config['normal']['epochs'])
    learning_rate = float(config['normal']['learning_rate'])
    discount_factor = float(config['normal']['discount_factor'])
    exploration_prob = float(config['normal']['exploration_prob'])
    simulated_steps = int(config['normal']['simulated_steps'])
    env = PacmanAgent.PacmanAgent()

    
    model = DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_steps)
    rewards = model.train()
    # plot_rewards(rewards)
    # model.save()
    # current_dir = Path(__file__).resolve().parent
    # filename = os.path.join(current_dir,"DynaQ - RL05_intersecting_tunnels_H_R E=10000 LR=0.2 DF=0.2 EP=0.2 SS=10.pkl")
    # model = DynaQ.load(env, filename)
    model.play()
