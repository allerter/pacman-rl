import configparser
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
        self.q_table = dict()
        self.p_model = dict()
        self.r_model = dict()
        self.level_name = PacmanAgent.level_name

    def state_to_index(self, state):
        return ''.join(map(str, state.flatten().tolist()))


    def find_entity(self, state, entity_name):
        result = np.where(state == PacmanAgent.ENTITIES_MAP[entity_name])
        return (result[0][0], result[1][0]) if result[0] else None


    def calculate_reward(self, state, done):
        reward = 0
        if self.env.lastRemainingDots > self.env.remainingDots:
            reward += 1  # Eating a dot
        if self.env.remainingDots == 0:
            reward += 10  # Clear all dots (win condition)
        if done and self.env.remainingDots > 0:
            reward -= 5  # Penalty losing the game
            return reward
        pacmnan_position = self.find_entity(state, "pacman")
        hunter_position = self.find_entity(state, "hunter_ghost")
        if pacmnan_position is None:
            return reward
        # if next move puts us on the tile next to a ghost, make reward negative
        if pacmnan_position[0] == hunter_position[0] and abs(hunter_position[1] - pacmnan_position[1]) <= 1:
            reward = -1
        if pacmnan_position[1] == hunter_position[1] and  abs(hunter_position[0] - pacmnan_position[0]) <= 1:
            reward = -1
        return reward


    def get_useful_actions(self, state):
        useful_actions = []
        position = self.find_entity(state, "pacman")
        # north
        if state[position[0] -1, position[1]] != 1:
            useful_actions.append(0)
        # west
        if state[position[0], position[1] - 1] != 1:
            useful_actions.append(1)
        # east
        if state[position[0], position[1] + 1] != 1:
            useful_actions.append(2)
        # south
        if state[position[0] + 1, position[1]] != 1:
            useful_actions.append(3)
        return useful_actions

    def save(self) -> str:
        current_dir = Path(__file__).resolve().parent
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
        with open(os.path.join(current_dir, filename), "wb") as file:
            pickle.dump(data, file)

    @classmethod
    def load(cls, env, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
        model = cls(env, data["epochs"], data["learning_rate"], data["discount_factor"], data["exploration_prob"], data["siumlated_steps"])
        model.q_table = data["q_table"]
        model.p_model = data["p_model"]
        model.r_model = data["r_model"]
        return model

    def train(self) -> list[float]:
        rewards = []
        for epoch in range(self.epochs):
            epoch_rewards = 0
            original_state = self.env.reset() 
            state = self.state_to_index(original_state)
            if self.q_table.get(state) is None:
                self.q_table[state] = [0, 0, 0, 0]
            if self.p_model.get(state) is None:
                    self.p_model[state] = ["0", "0", "0", "0"]
                    self.r_model[state] = [0, 0, 0, 0]

            done = False
            steps = 0
            self.env.lastRemainingDots = self.env.remainingDots
            while not done and steps < 200:
                # Choose action with epsilon-greedy strategy
                if np.random.rand() < self.exploration_prob:
                    # action = np.random.choice(get_useful_actions(original_state))  # Explore
                    action = self.env.action_space.sample()
                else:
                    action =  self.q_table[state].index(max(self.q_table[state])) # Exploit
            
                next_state, reward, done, info = self.env.step(action)
                original_state = next_state
                next_state = self.state_to_index(next_state)
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

                # planning
                for _ in range(self.simulated_steps):
                    sampled_state = random.choice(list(self.p_model.keys()))
                    sampled_action = random.randrange(0, 4)

                    simulated_next_state = self.p_model[sampled_state][sampled_action]
                    simulated_reward = self.r_model[sampled_state][sampled_action]

                    self.q_table[sampled_state][sampled_action] += self.learning_rate * (
                        simulated_reward
                        + self.discount_factor * max(self.q_table[next_state])
                        - self.q_table[sampled_state][sampled_action]
                    )

                state = next_state
                steps += 1
                epoch_rewards += reward
            rewards.append(epoch_rewards)
        return rewards
    
    def play(self) -> list[float]:
        episode_rewards = []
        for episode in range(5):
            # new_grid = randomize_level(PacmanAgent.level, PacmanAgent.tileTypes)
            # PacmanAgent.level = new_grid

            state = self.state_to_index(self.env.reset())
            done = False
            total_rewards = 0
            
            while not done:
                q_values = self.q_table.get(state)
                if q_values:
                    action =  q_values.index(max(q_values))
                else:
                    action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                next_state = self.state_to_index(next_state)
                self.env.render(action)
                total_rewards += reward
                state = next_state
                sleep(0.5)
                if done:
                    print(f"Episode: {episode + 1}, Total Reward: {total_rewards}")
                    break
            episode_rewards.append(total_rewards)
        self.env.close()
        return episode_rewards


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


    