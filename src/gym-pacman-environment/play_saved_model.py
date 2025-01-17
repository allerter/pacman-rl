import os
import configparser
import PacmanAgent
from dyna_q import DynaQ
from dqn import DQN
from utils import get_level


if __name__ == "__main__":

    filename = "" # set this to the file path of the pickled model

    env = PacmanAgent.PacmanAgent()
    if filename.lower().startswith("dynaq"):
        model = DynaQ.load(env, filename)
    else:
        model = DQN.load(filename)
    model.play()
