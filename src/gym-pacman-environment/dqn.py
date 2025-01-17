import math
from pathlib import Path
import random
import os
from collections import deque
from time import sleep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import PacmanAgent

class DQN:
    def _init_(self, env, state_size=None, action_space=None, discount_factor=None,
                exploration_prob=None, learning_rate=None, 
                batch_size=None, max_replay_size=None, epochs=None, target_update_freq=None):
        self.env = env
        self.state_size = state_size
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.min_exploration_prob = 0.1
        self.max_exploration_prob = exploration_prob
        self.exploration_decreasing_decay = 0.999
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_replay_size = max_replay_size
        self.epochs = epochs
        self.target_update_freq = target_update_freq
        self.replay_buffer = deque(maxlen=max_replay_size)
        self.dq_network = self.create_model()
        self.target_network = self.create_model()
        self.target_network.set_weights(self.dq_network.get_weights())
        self.level_name = PacmanAgent.level_name

    def create_model(self):
        model = Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='mse')
        return model

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

    def flatten_state(self, state):
        state = np.array(state).flatten()
        if state.shape[0] != self.state_size:
            raise Exception()
        return state

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


    def train_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)

        targets = rewards + self.discount_factor * max_next_q_values * (1 - dones)
        q_values = self.dq_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            q_values[i, action] = targets[i]

        self.dq_network.fit(states, q_values, verbose=0, batch_size=self.batch_size)

    def train(self):
        rewards = []
        for epoch in range(self.epochs):
            state = self.flatten_state(self.env.reset())
            total_reward = 0
            done = False
            self.env.lastRemainingDots = self.env.remainingDots

            while not done:
                if np.random.rand() < self.exploration_prob:
                    action = np.random.randint(0, self.action_space)
                else:
                    q_values = self.dq_network.predict(state.reshape(1, -1), verbose=0)
                    action = np.argmax(q_values[0])

                next_state, _, done, _ = self.env.step(action)
                reward = self.calculate_reward(next_state, done)
                next_state = self.flatten_state(next_state)
                total_reward += reward

                self.replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                self.env.lastRemainingDots = self.env.remainingDots
                self.train_q_network()

            step = epoch / self.epochs
            # reduce exploration prob after each episode
            self.exploration_prob = self.min_exploration_prob + (self.max_exploration_prob - self.min_exploration_prob) * (1 - math.log(1 + step))

            if (epoch + 1) % self.target_update_freq == 0:
                self.target_network.set_weights(self.dq_network.get_weights())

            rewards.append(total_reward)
            print(f"Episode {epoch} - Total Reward: {total_reward}")
        return rewards

    def save(self) -> str:
        """save model data to file"""
        filename = f"{self.__class__.__name__} - {self.level_name} E={self.epochs} LR={self.learning_rate} DF={self.discount_factor} EP={self.exploration_prob} SS={self.state_size}.pkl"
        self.dq_network.save(os.path.join(Path(__file__).resolve().parent.parent.parent, f"models/{self.__class__.__name__}/{self.level_name}"), filename)


    @classmethod
    def load(cls, env, filename):
        """load model data from file"""
        model_data = tf.keras.models.load_model(filename)
        model = cls(env)
        model.dq_network = model_data
        return model


    def play(self, verbose=True) -> list[list[bool, float]]:
        """evaluate the model by playing"""
        episodes_rewards = [0, 0] # wins & total rewards
        for episode in range(5):
            state = np.array(self.env.reset()).flatten()
            done = False
            total_rewards = 0
            won = False

            while not done:
                # get action from q values
                q_values = self.dq_network.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
                
                next_state, reward, done, info = self.env.step(action)
                total_rewards += reward
                state = np.array(next_state).flatten()
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