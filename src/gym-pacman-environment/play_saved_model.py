import os
import re
import configparser
import PacmanAgent
from dyna_q import DynaQ
from dqn import DQN
from utils import get_level


if __name__ == "__main__":

    filename = "models/DQN/RL06_intersecting_tunnels_deadends_H_R/DQN - RL06_intersecting_tunnels_deadends_H_R E=2000 LR=0.001 DF=0.99 EP=0.8 SS=49.keras" # set this to the file path of the pickled model

    env = PacmanAgent.PacmanAgent()
    PacmanAgent.level_name = match = re.search(r"(?:DynaQ|DQN)\s+-\s+(RL\S+)", filename).group(1)
    PacmanAgent.level = get_level(PacmanAgent.level_name)
    if "dynaq" in filename.lower():
        model = DynaQ.load(env, filename)
    else:
        model = DQN.load(env, filename)
    model.play()
