import os
import configparser
import PacmanAgent
from dyna_q import DynaQ
from utils import get_level


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

    filename = "" # set this to the file path of the pickled model
    model = DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_steps)
    model = DynaQ.load(env, filename)
    model.play()
