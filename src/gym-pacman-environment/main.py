import os
from pathlib import Path
from PacmanAgent import PacmanAgent, level_name
from dyna_q import DynaQ
import configparser


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

    current_dir = Path(__file__).resolve().parent
    # filename = os.path.join(current_dir,"DynaQ - RL05_intersecting_tunnels_H_R E=10000 LR=0.2 DF=0.2 EP=0.2 SS=10.pkl")
    model = DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_steps)
    model.train()
    # model.save()
    # model = DynaQ.load(env, filename)
    # model.load(filename)
    model.play()


    