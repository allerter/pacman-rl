import os
import re
import configparser
import PacmanAgent
from dyna_q import DynaQ
from dqn import DQN
from utils import get_level


if __name__ == "__main__":

    filename = "models/DynaQ/RL02_square_tunnel_H/DynaQ - RL02_square_tunnel_H E=3000 LR=0.2 DF=0.2 EP=0.13070194999971538 SS=0.pkl" # set this to the file path of the pickled model

    env = PacmanAgent.PacmanAgent()
    PacmanAgent.level_name = match = re.search(r"(?:DynaQ|DQN)\s+-\s+(RL\S+)", filename).group(1)
    print(PacmanAgent.level_name)
    PacmanAgent.level = get_level(PacmanAgent.level_name)
    if "dynaq" in filename.lower():
        model = DynaQ.load(env, filename)
    else:
        model = DQN.load(env, filename)
    model.play()
