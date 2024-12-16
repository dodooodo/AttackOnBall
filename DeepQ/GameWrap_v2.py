from pathlib import Path
import sys
aob = str(Path(__file__).parent.parent.joinpath('AttackOnBall'))
sys.path.append(aob)

from attack_on_ball_rl import AttackOnBall
from PIL import Image
import numpy as np
from collections import deque

IMG_SIZE = (600, 325) #(W, H)
# self.img_resize = (84, 84) #(W, H)
CLIP_REWARD = 4


class GameWrapper:
    def __init__(self, *args, **kwargs):
        # super().__init__()
        self.game = AttackOnBall(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.game, name)
    
    def reset(self):
        return self.game.reset()

    def reset__(self):
        self.state.clear()
        obs, _, gameover, game_score = self.game.reset()
        self.state.append(self.preprocess(obs))
        if self.frame_stack == 1:
            return np.expand_dims(self.state, axis=(0,)), 0, gameover, game_score

        # roughly same as self.step, but stack frame_stack - 1 times.
        for _ in range(self.frame_stack - 1):
            obs, _, gameover, game_score = self.game.step(0)
            self.state.append(self.preprocess(obs))
            # ''' clip reward to -1 ~ 1'''
            # reward = game_score - 1
            # clip bonus reward to 1
            # if reward > 0:
            #     reward = 1
            # total_reward += reward
        # clip reward to 0 ~ 1
        # total_reward /= self.frame_stack
        # (4, 84, 84) -> (1, 4, 84, 84)
        return np.expand_dims(self.state, axis=(0,)), 0, gameover, game_score


    def step(self, action):
        total_reward = 0
        gameover = False
        for i in range(1, self.frame_stack + 1):
            if not gameover:
                obs, reward, gameover, game_score = self.game.step(action)
                self.state.append(self.preprocess(obs))
                # normal reward
                # reward = game_score - 1
                # total_reward += round(reward)
                total_reward += reward - 1
            else:
                # keep game state dim = (frame_stack, H, W)
                for _ in range(self.frame_stack - i):
                    # self.state.append(np.zeros((self.img_resize[1], self.img_resize[0]), np.float32))
                    self.state.append(self.preprocess(obs))
                # gameover reward = -1
                total_reward = -1
                break
        # bonus reward = 1
        # if total_reward > 1:
        #     total_reward = 1
        # total_reward /= self.frame_stack
        # (4, 84, 84) -> (1, 4, 84, 84)
        return np.expand_dims(self.state, axis=(0,)), total_reward, gameover, game_score
    
    def hello(self):
        print('ghello')


import os
os.environ['SDL_VIDEODRIVER']='dummy'


class ProcessFrameWrapper(GameWrapper):
    def __init__(self, game, img_resize=(84, 84), *args, **kwargs):
        super().__init__(game, *args, **kwargs)
        self.img_resize = img_resize
    
    @staticmethod
    def preprocess(self, img):
        img = Image.frombytes('RGB', IMG_SIZE, img)
        img = img.convert('L')
        img = img.resize(self.img_resize)
        return np.array(img).astype(np.float32) / 255.
    
    def hello(self):
        print('pheoll')
    


class FrameStackWrapper(GameWrapper):
    def __init__(self, game, frame_stack=4, *args, **kwargs):
        super().__init__(game, *args, **kwargs)
        self.frame_stack = frame_stack
        self.state = deque(maxlen=frame_stack)
        # self.outshape = (1, frame_stack, img_resize[1], img_resize[0])
    
    def hello(self):
        print('frame stack')

# a = AttackOnBall()
def create_game():
    g = AttackOnBall()
    p = ProcessFrameWrapper(g)
    f = FrameStackWrapper(p)
    return f
f = create_game()
f.hello()

print(f.reset())
