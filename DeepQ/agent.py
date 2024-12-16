from model import DQN
import numpy as np

class Agent:
    def __init__(self, net):
        self.net = net
    
    def play_step(self, state):
        out = self.net(state)
        action = np.argmax(out)
        return action