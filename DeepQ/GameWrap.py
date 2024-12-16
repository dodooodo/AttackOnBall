from pathlib import Path
import sys
# aob = str(Path(__file__).parent.parent.joinpath('AttackOnBall'))
aob = str(Path(__file__).parent.parent.joinpath('SimpleVersion'))
sys.path.append(aob)

from attack_on_ball_rl import AttackOnBall
from PIL import Image
import numpy as np
from collections import deque
# from TIME import timeit

IMG_SIZE = (500, 300) #(W, H)
# self.img_resize = (84, 84) #(W, H)
CLIP_REWARD = 4


class GameWrapper(AttackOnBall):
    def __init__(self, img_resize=(84, 84), frame_stack=4, *args, **kwargs):
        self.game = AttackOnBall(*args, **kwargs)
        self.img_resize = img_resize
        self.frame_stack = frame_stack
        self.outshape = (1, frame_stack, img_resize[1], img_resize[0])
        self.state = deque(maxlen=frame_stack)


    def __getattr__(self, name):
        return getattr(self.game, name)
    

    def reset(self):
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
                total_reward += reward
            else:
                # keep game state dim = (frame_stack, H, W)
                for _ in range(self.frame_stack - i):
                    self.state.append(np.zeros((self.img_resize[1], self.img_resize[0]), np.float32))
                # gameover reward = -1
                total_reward = -1
                break
        # bonus reward = 1
        if total_reward > 1:
            total_reward = 1
        # total_reward /= self.frame_stack
        # (4, 84, 84) -> (1, 4, 84, 84)
        return np.expand_dims(self.state, axis=(0,)), total_reward, gameover, game_score


    def preprocess(self, img):
        img = Image.frombytes('RGB', IMG_SIZE, img)
        img = img.convert('L')
        img = img.resize(self.img_resize)
        # img.show()
        # print(asdf)
        return np.array(img, dtype=np.float32) / 255.

# import os
# os.environ['SDL_VIDEODRIVER']='dummy'

