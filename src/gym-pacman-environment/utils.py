from pathlib import Path
import random
from matplotlib import pyplot as plt
import numpy as np

def get_level(level_name) -> list:
    """
    Converts the contents of a file representing a grid to a nested list format.
    
    :param file_path: Path to the input file containing the grid.
    :return: A list of lists representing the grid.
    """
    file_path = Path(__file__).resolve().parent.parent / "Pacman_Level" / f"{level_name}.pml"
    level = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip the newline character and create a list of characters
            row = list(line.strip())
            # Skip empty lines
            if row:
                level.append(row)
    return level

def randomize_level(grid, tileTypes):
    # Step 1: Identify positions of walls and free spaces
    wall_positions = []
    free_positions = []
    pacman_position = None
    ghost_position = None
    
    # Step 2: Scan the grid and classify positions
    for row_idx, row in enumerate(grid):
        for col_idx, tile in enumerate(row):
            if tile == tileTypes["wall"]:
                wall_positions.append((row_idx, col_idx))
            else:
                free_positions.append((row_idx, col_idx))
                if tile == tileTypes["pacman"]:
                    pacman_position = (row_idx, col_idx)
                elif tile == tileTypes["ghost_hunter"]:
                    ghost_position = (row_idx, col_idx)
    
    # Step 3: Separate and shuffle entities (dots, pacman, ghost)
    num_dots = sum(row.count(tileTypes["dot"]) for row in grid)
    dots = [tileTypes["dot"]] * num_dots
    other_entities = [tileTypes["pacman"], tileTypes["ghost_hunter"]]
    
    # Add any additional ghosts or entities if needed
    num_ghosts = sum(row.count(tileTypes["ghost_rnd"]) for row in grid)
    ghosts = [tileTypes["ghost_rnd"]] * num_ghosts
    
    # Combine all entities to randomize
    entities = dots + other_entities + ghosts
    
    # Step 4: Shuffle entities
    random.shuffle(entities)
    
    # Step 5: Build the new grid by placing the entities in free positions
    new_grid = [['#' if (r, c) in wall_positions else None for c in range(len(row))] for r, row in enumerate(grid)]
    
    # Fill free positions with randomized entities
    entity_idx = 0
    for row_idx, row in enumerate(new_grid):
        for col_idx, cell in enumerate(row):
            if cell is None:  # If it's not a wall
                new_grid[row_idx][col_idx] = entities[entity_idx]
                entity_idx += 1
    
    # Return the randomized grid
    return new_grid


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