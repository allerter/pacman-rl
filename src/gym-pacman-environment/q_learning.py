from time import sleep
import numpy as np
from PacmanAgent import ENTITIES_MAP

class QLearning:
    def __init__(self, env, epochs, learning_rate, discount_factor, exploration_prob):
        self.env = env
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = dict()

    def state_to_index(self, state):
        return ''.join(map(str, state.flatten().tolist()))


    def find_entity(self, state, entity_name):
        result = np.where(state == ENTITIES_MAP[entity_name])
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


    def train(self) -> list[float]:
        rewards = []
        for epoch in range(self.epochs):
            epoch_rewards = 0
            original_state = self.env.reset() 
            state = self.state_to_index(original_state)
            if self.q_table.get(state) is None:
                self.q_table[state] = [0, 0, 0, 0]

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


                # Reward structure
                reward = self.calculate_reward(original_state, done)
                self.env.lastRemainingDots = self.env.remainingDots
            
                self.q_table[state][action] += self.learning_rate * \
                    (reward + self.discount_factor *
                    self.q_table[next_state][action]) - self.q_table[state][action]

                state = next_state
                steps += 1
                epoch_rewards += reward
            rewards.append(epoch_rewards)
        return rewards
    
    def play(self) -> list[float]:
        episode_rewards = []
        for episode in range(5):
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