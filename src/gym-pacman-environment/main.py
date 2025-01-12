import os
from PacmanAgent import PacmanAgent
from q_learning import QLearning
import configparser


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    result = config.read(os.path.join(os.path.dirname(__file__), 'hyperparameters.ini'))
    epochs = int(config['normal']['epochs'])
    learning_rate = float(config['normal']['learning_rate'])
    discount_factor = float(config['normal']['discount_factor'])
    exploration_prob = float(config['normal']['exploration_prob'])
    env = PacmanAgent()

    model = QLearning(env, epochs, learning_rate, discount_factor, exploration_prob)
    model.train()
    model.play()


    