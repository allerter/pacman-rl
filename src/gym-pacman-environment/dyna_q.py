import configparser
import math
from pathlib import Path
import pickle
from time import sleep
import random
import os
import numpy as np
import PacmanAgent
from utils import randomize_level

class DynaQ:
    def __init__(self, env, epochs, learning_rate, discount_factor, exploration_prob, siumlated_steps):
        self.env = env
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.simulated_steps = siumlated_steps
        self.min_exploration_prob = 0.1
        self.max_exploration_prob = exploration_prob
        self.exploration_decreasing_decay = 0.999
        self.q_table = dict()
        self.p_model = dict()
        self.r_model = dict()
        self._valid_states = []
        self.level_name = PacmanAgent.level_name

    def state_to_index(self, state):
        """Flattens a state and turns it into a string"""
        return ''.join(map(str, state.flatten().tolist()))


    def find_entities(self, state, entity_name) -> list[list[int, int]]:
        """Finds entities within the environment"""
        results = []
        if entity_name == "ghost":
            for entity in ("hunter_ghost", "random_ghost"):
                result = np.where(state == PacmanAgent.ENTITIES_MAP[entity])
                if result[0]:
                    results.append([result[0][0], result[1][0]])
        else:
            result = np.where(state == PacmanAgent.ENTITIES_MAP[entity_name])
            if result[0]:
                results.append([result[0][0], result[1][0]])
        return results


    def calculate_reward(self, state, done):
        """Calculates reward for each state"""
        reward = 0
        if self.env.lastRemainingDots > self.env.remainingDots:
            reward += 1  # Eating a dot
        if done and self.env.remainingDots == 0:
            return reward + 5  # Clear all dots (win condition)
        if done and self.env.remainingDots > 0:
            return reward - 5  # Penalty losing the game
            
        pacmnan_position = self.find_entities(state, "pacman")
        pacmnan_position = pacmnan_position[0] if pacmnan_position else None
        for ghost in self.find_entities(state, "hunter_ghost"):
            # if next move puts us on the tile next to a ghost, make reward negative
            if pacmnan_position[0] == ghost[0] and abs(ghost[1] - pacmnan_position[1]) <= 1:
                reward -= 2
            if pacmnan_position[1] == ghost[1] and abs(ghost[0] - pacmnan_position[0]) <= 1:
                reward -= 2
        return reward


    def get_useful_actions(self, state):
        """Get viable actions for a state (actions that make pacman move)"""
        useful_actions = []
        position = self.find_entities(state, "pacman")[0]
        # north
        try:
            if state[position[0] - 1, position[1]] != 1:
                useful_actions.append(1)
        except IndexError:
            pass
        # west
        try:
            if state[position[0], position[1] - 1] != 1:
                useful_actions.append(0)
        except IndexError:
            pass
        # east
        try:
            if state[position[0], position[1] + 1] != 1:
                useful_actions.append(3)
        except IndexError:
            pass
        # south
        try:
            if state[position[0] + 1, position[1]] != 1:
                useful_actions.append(2)
        except IndexError:
            pass
        return useful_actions

    def get_best_action(self, state):
        """Tries to come up with actions that could best benifit pacman
        for example eating a dot, avoiding ghosts, etc
        """
        position = self.find_entities(state, "pacman")[0]
        ghost_positions = self.find_entities(state, "ghost")
        useful_actions = []
        useful_directions = [] # directions that won't get closer to ghosts
        available_actions = [] # actions that don't go into a wall
        # north
        try:
            if state[position[0] - 1, position[1]] != 1:
                available_actions.append(1)
                for ghost in ghost_positions:
                    if (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1])) > (abs(ghost[0] - position[0] - 1) + abs(ghost[1] - position[1])):
                        useful_directions.append("north")
                        break
        except IndexError:
            pass     
        # west
        try:
            if state[position[0], position[1] - 1] != 1:
                available_actions.append(0)
                for ghost in ghost_positions:
                    if (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1])) > (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1] - 1)):
                        useful_directions.append("west")
                        break
        except IndexError:
            pass                            
        # east
        try:
            if state[position[0], position[1] + 1] != 1:
                available_actions.append(3)
                for ghost in ghost_positions:
                    if (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1])) > (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1] + 1)):
                        useful_directions.append("east")
                        break
        except IndexError:
            pass
        # south
        try:
            if state[position[0] + 1, position[1]] != 1:
                available_actions.append(2)
                for ghost in ghost_positions:
                    if (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1] + 1)) > (abs(ghost[0] - position[0]) + abs(ghost[1] - position[1])):
                        useful_directions.append("south")
                        break
        except IndexError:
            pass
        # consider actions that will result in eating a dot
        # north
        try:
            if "north" in useful_directions and state[position[0] - 1, position[1]] == 2:
                useful_actions.append(1)
        except IndexError:
            pass
        # west
        try:
            if "west" in useful_directions and state[position[0], position[1] - 1] == 2:
                useful_actions.append(0)
        except IndexError:
            pass
        # east
        try:
            if "east" in useful_directions and state[position[0], position[1] + 1] == 2:
                useful_actions.append(3)
        except IndexError:
            pass
        # south
        try:
            if "south" in useful_directions and state[position[0] + 1, position[1]] == 2:
                useful_actions.append(2)
        except IndexError:
            pass

        # if no action with eating dots has been identified, take any that will not get closer to ghosts
        if not useful_actions:
            for direction in useful_directions:
                if direction == "north":
                    useful_actions.append(1)
                elif direction == "west":
                    useful_actions.append(0)
                elif direction == "east":
                    useful_actions.append(3)
                else:
                    useful_actions.append(2)
        
        # if still no action is identified, take any that won't move into a wall
        if not useful_actions:
            useful_actions = available_actions
        return useful_actions

    def save(self) -> str:
        """save model data to file"""
        filename = f"{self.__class__.__name__} - {self.level_name} E={self.epochs} LR={self.learning_rate} DF={self.discount_factor} EP={self.exploration_prob} SS={self.simulated_steps}.pkl"
        data = {
            "level_name": self.level_name, 
            "epochs": self.epochs, 
            "learning_rate": self.learning_rate, 
            "discount_factor": self.discount_factor, 
            "exploration_prob": self.exploration_prob, 
            "siumlated_steps": self.simulated_steps, 
            "q_table": self.q_table,
            "p_model": self.p_model,
            "r_model": self.r_model,
        }
        with open(os.path.join(os.path.join(Path(__file__).resolve().parent.parent.parent, f"models/{self.__class__.__name__}/{self.level_name}"), filename), "wb") as file:
            pickle.dump(data, file)

    @classmethod
    def load(cls, env, filename):
        """load model data from file"""
        with open(filename, "rb") as file:
            data = pickle.load(file)
        model = cls(env, data["epochs"], data["learning_rate"], data["discount_factor"], data["exploration_prob"], data["siumlated_steps"])
        model.q_table = data["q_table"]
        model.p_model = data["p_model"]
        model.r_model = data["r_model"]
        return model

    def switch_grid(self, level):
        def find_element(array, target):
            for row_index, row in enumerate(array):
                for col_index, element in enumerate(row):
                    if element == target:
                        return row_index, col_index  # Return position as (row, column)

        pacman = find_element(level, "P")
        hunter = find_element(level, "H")
        level[pacman[0]][pacman[1]], level[hunter[0]][hunter[1]] = level[hunter[0]][hunter[1]], level[pacman[0]][pacman[1]]
        return level                                                                                                  

    def train(self) -> list[float]:
        """train the model and return rewards"""
        rewards = []
        for epoch in range(self.epochs):
            epoch_rewards = 0
            original_state = self.env.reset() 
            state = self.state_to_index(original_state)
            # initialize tables
            if self.q_table.get(state) is None:
                self.q_table[state] = [0, 0, 0, 0]
            if self.p_model.get(state) is None:
                    self.p_model[state] = ["0", "0", "0", "0"]
                    self.r_model[state] = [0, 0, 0, 0]

            done = False
            steps = 0
            self.env.lastRemainingDots = self.env.remainingDots
            # restrict steps taken to avoid inifnite loops and useless actions
            while not done and steps < 200:
                if np.random.rand() < self.exploration_prob:
                    action = np.random.choice(self.get_useful_actions(original_state))  # Explore
                else:
                    action =  self.q_table[state].index(max(self.q_table[state])) # Exploit
            
                next_state, reward, done, info = self.env.step(action)
                original_state = next_state
                next_state = self.state_to_index(next_state)
                # initialize state cells
                if self.q_table.get(next_state) is None:
                    self.q_table[next_state] = [0, 0, 0, 0]
                if self.p_model.get(state) is None:
                    self.p_model[state] = ["0", "0", "0", "0"]
                    self.r_model[state] = [0, 0, 0, 0]

                # Reward structure
                reward = self.calculate_reward(original_state, done)
                self.env.lastRemainingDots = self.env.remainingDots
            
                # Update Q-Table and model
                self.q_table[state][action] += self.learning_rate * \
                    (reward + self.discount_factor *
                    max(self.q_table[next_state])) - self.q_table[state][action]
                self.p_model[state][action] = next_state
                self.r_model[state][action] = reward
                # keep track of states with values
                # this will help optimize the planning where random elements need to be selected
                self._valid_states.append(state)

                # planning
                for _ in range(self.simulated_steps):
                    # get random state, action and its reward
                    sampled_state = random.choice(self._valid_states)
                    sampled_action = random.choice([i for i, value in enumerate(self.p_model[sampled_state]) if value != "0"])
                    simulated_next_state = self.p_model[sampled_state][sampled_action]
                    simulated_reward = self.r_model[sampled_state][sampled_action]

                    # update q table from random state-action-reward
                    self.q_table[sampled_state][sampled_action] += self.learning_rate * (
                        simulated_reward
                        + self.discount_factor * max(self.q_table[simulated_next_state])
                        - self.q_table[sampled_state][sampled_action]
                    )

                state = next_state
                steps += 1
                epoch_rewards += reward
            # print(f"Episode #{epoch} finished with reward: {epoch_rewards}")
            step = epoch / self.epochs
            # reduce exploration prob after each episode
            self.exploration_prob = self.min_exploration_prob + (self.max_exploration_prob - self.min_exploration_prob) * (1 - math.log(1 + step))
            rewards.append(epoch_rewards)
        return rewards
    
    def play(self, verbose=True) -> list[list[bool, float]]:
        """evaluate the model by playing"""
        episodes_rewards = [0, 0] # wins & total rewards
        for episode in range(5):
            originial_state = self.env.reset()
            state = self.state_to_index(originial_state)
            done = False
            total_rewards = 0
            won = False

            while not done:
                # try to find state in q table
                # if state hasn't been explored during training, try to come up with a best action
                q_values = self.q_table.get(state)
                if q_values:
                    action =  q_values.index(max(q_values))
                else:
                    useful_actions = self.get_best_action(originial_state)
                    action = np.random.choice(useful_actions)
                    if verbose:
                        print(f"Taking random action from {useful_actions}: {action}")
                next_state, reward, done, info = self.env.step(action)
                originial_state = next_state
                next_state = self.state_to_index(next_state)
                total_rewards += reward
                state = next_state
                if verbose:
                    sleep(0.5)
                    self.env.render(action)
            won = True if self.env.remainingDots == 0 else False
            episode_result = "Won :)" if won else "Lost!"
            if verbose:
                print(f"Episode: {episode + 1}, Total Reward: {total_rewards} - {episode_result}")
            # update play results
            episodes_rewards[0] += 1 if won else 0
            episodes_rewards[1] += total_rewards
        episodes_rewards[1] = round(episodes_rewards[1], 1)
        self.env.close()

        return episodes_rewards


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    result = config.read(os.path.join(os.path.dirname(__file__), 'hyperparameters.ini'))
    epochs = int(config['normal']['epochs'])
    learning_rate = float(config['normal']['learning_rate'])
    discount_factor = float(config['normal']['discount_factor'])
    exploration_prob = float(config['normal']['exploration_prob'])
    simulated_steps = int(config['normal']['simulated_steps'])
    env = PacmanAgent()

    # current_dir = Path(__file__).resolve().parent
    # filename = os.path.join(current_dir, "DynaQRL04_square_tunnel_deadends_H E=10 LR=0.2 DF=0.2 EP=0.2 SS=10.pkl")
    model = DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_steps)
    model.train()
    model.save()
    # model = DynaQ.load(env, filename)
    # model.load(filename)
    model.play()


    