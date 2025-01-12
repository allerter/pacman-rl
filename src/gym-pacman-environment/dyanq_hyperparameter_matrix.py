import os
from dyna_q import DynaQ
from PacmanAgent import PacmanAgent, level_name
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    env = PacmanAgent()
    learning_rates = [0.2]
    discount_factors = [0.5]
    exploration_probs = [0.1, 0.25]
    simulated_steps = [3, 7]
    results_matrix = np.zeros((len(learning_rates), len(discount_factors), len(exploration_probs), len(simulated_steps)))
    epochs = 1000
    model = DynaQ

    for i, learning_rate in enumerate(learning_rates):
        for j, discount_factor in enumerate(discount_factors):
            for k, exploration_prob in enumerate(exploration_probs):
                for l, simulated_step in enumerate(simulated_steps):
                    results_matrix[i, j, k, l] = sum(DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_step).train())

    # Save and plot results for each exploration prob and simulated step
    for exploration_prob in exploration_probs:
        fixed_ep_idx = exploration_probs.index(exploration_prob)
        for simulated_step in simulated_steps:
            fixed_step_idx = simulated_steps.index(simulated_step)
            results_for_fixed_ep_and_step = results_matrix[:, :, fixed_ep_idx, fixed_step_idx]

            # Plot results
            plt.figure(figsize=(10, 6))
            for i, lr in enumerate(learning_rates):
                plt.plot(
                    discount_factors, 
                    results_for_fixed_ep_and_step[i], 
                    label=f'LR={lr}', 
                    marker='o', 
                    linewidth=2
                )

            # Add labels, title, and legend
            plt.xlabel('Discount Factor', fontsize=14)
            plt.ylabel('Cumulative Reward', fontsize=14)
            plt.text(0.02, 0.95, f"Epochs: {epochs}\nSimulated Step: {simulated_step}", 
                     transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            plt.title(
                f'Results for Exploration Probability = {exploration_probs[fixed_ep_idx]}, Simulated Step = {simulated_step}', 
                fontsize=16
            )
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            # Create the directory if it doesn't exist
            save_dir = f"models/{model.__name__}/{level_name}"
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

            # Save plot
            save_path = os.path.join(save_dir, f"{epochs} epochs - EP={exploration_prob}, Steps={simulated_step}.png")
            plt.savefig(save_path)
            plt.close()