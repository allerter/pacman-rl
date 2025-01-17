import os
from pathlib import Path
from dyna_q import DynaQ
import PacmanAgent
from matplotlib import pyplot as plt
from utils import get_level
import numpy as np

if __name__ == "__main__":
    env = PacmanAgent.PacmanAgent()
    # configuration matrices
    PacmanAgent.level_name = "RL05_intersecting_tunnels_H_R"
    PacmanAgent.level = get_level(PacmanAgent.level_name)
    learning_rates = [0.4, 0.8]
    discount_factors = [0.4, 0.7]
    exploration_probs = [0.2, 0.4]
    simulated_steps = [8, 16]
    results_matrix = np.zeros((len(learning_rates), len(discount_factors), len(exploration_probs), len(simulated_steps)))
    evaluation_matrix = np.zeros_like(results_matrix, dtype=object)  # To store [wins, total_rewards]
    epochs = 9000
    model = DynaQ

    # run all combinations
    for i, learning_rate in enumerate(learning_rates):
        for j, discount_factor in enumerate(discount_factors):
            for k, exploration_prob in enumerate(exploration_probs):
                for l, simulated_step in enumerate(simulated_steps):
                    current_model = DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_step)
                    results_matrix[i, j, k, l] = sum(current_model.train())
                    evaluation_results = current_model.play(verbose=False)  # -> [number_of_wins, total_rewards]
                    evaluation_matrix[i, j, k, l] = evaluation_results  # Store evaluation results

    # Save and plot results for each exploration prob and simulated step
    for exploration_prob in exploration_probs:
        fixed_ep_idx = exploration_probs.index(exploration_prob)
        for simulated_step in simulated_steps:
            fixed_step_idx = simulated_steps.index(simulated_step)
            results_for_fixed_ep_and_step = results_matrix[:, :, fixed_ep_idx, fixed_step_idx]
            evaluations_for_fixed_ep_and_step = evaluation_matrix[:, :, fixed_ep_idx, fixed_step_idx]

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

                # Add text annotations for number of wins and evaluation total rewards
                for j, discount_factor in enumerate(discount_factors):
                    num_wins, total_rewards = evaluations_for_fixed_ep_and_step[i, j]
                    plt.text(
                        discount_factor, 
                        results_for_fixed_ep_and_step[i, j], 
                        f"Evaluation Results\nWins:{num_wins}\nRewards:{total_rewards}",  # Include "Evaluation Results" at the top
                        fontsize=10, 
                        ha='center', 
                        va='bottom', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                        multialignment='left'
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
            save_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, f"models/{model.__name__}/{PacmanAgent.level_name}")
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

            # Save plot
            save_path = os.path.join(save_dir, f"{epochs} epochs - EP={exploration_prob}, Steps={simulated_step}.png")
            plt.savefig(save_path)
            plt.close()
