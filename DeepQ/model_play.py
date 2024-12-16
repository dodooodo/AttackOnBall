from GameWrap import GameWrapper
import torch
from noisy_duel_net import DQN
import time
import pygame

if __name__ == '__main__':
    game = GameWrapper(hideScreen=True, state_mode='image', frame_stack=4)
    state, reward, gameover, game_score = game.reset()
    # print(reward, game_score)
    # model = DQN((1, 4, 84, 84), 3)
    # model.load_state_dict(torch.load(r'model_out\0415_15-13-52_nlp-ws3.pt', map_location='cpu'))
    clock = pygame.time.Clock()
    t = []
    for _ in range(1000):
        # clock.tick(30) #frame stack = 4
        # clock.tick(120) #frame stack = 1
        clock.tick()
        # print(clock.get_fps())
        t.append(clock.get_fps())
        
        # state = torch.from_numpy(state).to('cpu')
        # with torch.no_grad():
            # x = model(state)
        # print(x)
        # action = torch.argmax(x)
        # state, reward, gameover, game_score = game.step(action)
        # state, reward, gameover, game_score = game.step(1)
        import random
        state, reward, gameover, game_score = game.step(random.choice((0, 1, 2)))
        # time.sleep(0.5)
        if gameover:
            # time.sleep(3)
            state, reward, gameover, game_score = game.reset()

    print(sum(t)/1000)