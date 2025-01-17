import os
import configparser
import PacmanAgent
from dyna_q import DynaQ
from dqn import DQN
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
    model = config['normal']['model']
    env = PacmanAgent.PacmanAgent()
    
    if model == 'dynaq':
        simulated_steps = int(config['normal']['simulated_steps'])
        model = DynaQ(env, epochs, learning_rate, discount_factor, exploration_prob, simulated_steps)
    else:
        state_size = int(config['normal']['state_size'])
        action_space = int(config['normal']['action_space'])
        batch_size = int(config['normal']['batch_size'])
        max_replay_size = int(config['normal']['max_replay_size'])
        target_update_freq = int(config['normal']['target_update_freq'])
        model = DQN(env, env, state_size, action_space, discount_factor,
                exploration_prob, learning_rate, 
                batch_size, max_replay_size, epochs, target_update_freq)
    
    model.train()
    model.play()
