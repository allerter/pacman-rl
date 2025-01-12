from q_learning import QLearning
from PacmanAgent import PacmanAgent, level_name
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    env = PacmanAgent()
    learning_rates = [0.2, 0.5, 0.8]
    discount_factors = [0.2, 0.5, 0.8]
    exploration_probs = [0.2, 0.5, 0.8]
    results_matrix = np.zeros((len(learning_rates), len(discount_factors), len(exploration_probs)))
    epochs = 10

    for i, learning_rate in enumerate(learning_rates):
        for j, discount_factor in enumerate(discount_factors):
            for k, exploration_prob in enumerate(exploration_probs):
                results_matrix[i, j, k] = sum(QLearning(env, epochs, learning_rate, discount_factor, exploration_prob).train())

    # save results for each exploration prob
    for exploration_prob in exploration_probs:
        fixed_ep_idx = exploration_probs.index(exploration_prob)
        results_for_fixed_ep = results_matrix[:, :, fixed_ep_idx]
        # Plot results
        plt.figure(figsize=(10, 6))
        for i, lr in enumerate(learning_rates):
            plt.plot(discount_factors, results_for_fixed_ep[i], label=f'LR={lr}', marker='o', linewidth=2)

        # Add labels, title, and legend
        plt.xlabel('Discount Factor', fontsize=14)
        plt.ylabel('Cumulative Reward', fontsize=14)
        plt.text(0.02, 0.95, f"Epochs: {epochs}", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        plt.title(f'Results for Fixed Exploration Probability = {exploration_probs[fixed_ep_idx]}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"data/{level_name}/{epochs} epochs - {exploration_prob} exploration prob.png")