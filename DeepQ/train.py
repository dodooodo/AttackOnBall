from GameWrap import GameWrapper
from model import DQN
import time
import datetime
import random
from collections import deque
import numpy as np
import torch
import torch.cuda
from torch.nn import MSELoss
from torch.optim import AdamW
import logging
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
os.environ['SDL_VIDEODRIVER']='dummy'

# goal
# 100s * 120f/s = 12000f
MEAN_REWARD_GOAL = 12_000

# preprocess
FRAME_STACK = 4
IMG_SIZE = (600, 325)
IMG_RESIZE = (128, 84)
CLIP_REWARD = 4

# training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

GAMMA = 0.99
EPSILON_START = 1.
EPSILON_END = 0.1
EPSILON_STEP = 100

MEMORY_SIZE = 100
SYNC_TARGET_FRAMES = 10


random.seed(901119)
np.random.seed(901119)
torch.manual_seed(901119)


class replayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
    
    def __len__(self):
        return len(self.memory)

    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        indexes = np.random.choice(len(self.memory), size=batch_size, replace=False)
        states, actions, rewards, gameovers, new_states = zip(*[self.memory[i] for i in indexes])
        return np.array(states), np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), \
            np.array(gameovers, dtype=np.uint8), np.array(new_states)


class Agent:
    def __init__(self, game, replay_memory):
        self.game = game
        self.replay_memory = replay_memory
        self.reset()
    
    def reset(self):
        self.state, _, _ = self.game.reset()
        self.total_reward = 0
    
    def play_step(self, net, epsilon, device='cpu'):
        # 2. Select random action epsilon of time else a = argmax(Qs,a).
        if random.random() < epsilon:
            action = random.choice((0, 1, 2))
        else:
            state = torch.from_numpy(self.state).to(device)
            Qvalues = net(state)
            action = torch.argmax(Qvalues).item()
        # 3. Play and get reward & next state s'.
        new_state, reward, gameover = self.game.step(action)
        # 4. Store transition in replay memory.
        transition = (self.state.squeeze(0), action, reward, gameover, new_state.squeeze(0))
        self.replay_memory.append(transition)
        # get total reward
        self.total_reward += reward
        # return total reward at the end of every episode.
        episode_finish_reward = None
        if gameover:
            episode_finish_reward = self.total_reward
            self.reset()
        return episode_finish_reward


# def init_param(net, device):
#     net = net.to(device).float()
#     net(torch.ones((1, FRAME_STACK, IMG_RESIZE[0], IMG_RESIZE[1])).to(device))
#     return net


def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, gameovers, next_states = batch

    states = torch.from_numpy(states).to(device)
    actions = torch.from_numpy(actions).to(device)
    rewards = torch.from_numpy(rewards).to(device)
    gameover_mask = torch.BoolTensor(gameovers).to(device)
    next_states = torch.from_numpy(next_states).to(device)

    Qnet = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        nextQ = tgt_net(next_states).max(1)[0]
        nextQ[gameover_mask] = 0.
        nextQ = nextQ.detach()
    
    expectedQ = rewards + nextQ * GAMMA
    return MSELoss()(Qnet, expectedQ)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='enable cuda')
    parser.add_argument('-r', '--mean_reward_goal', default=MEAN_REWARD_GOAL, 
                        help=f'Mean reward goal, default = {MEAN_REWARD_GOAL}')
    parser.add_argument('-f', '--frame_stack', default=FRAME_STACK, type=int,
                        help=f'Frame stack, default = {FRAME_STACK}')
    args = parser.parse_args()

    filename = datetime.datetime.now().strftime('%m%d_%H%M_%S')
    logging.basicConfig(level=logging.DEBUG, filename=f'train_out/{filename}.txt', filemode='w', format='%(message)s')
    logging.info(
        'INPUT_SHAPE {}, MEAN_REWARD_GOAL {}, FRAME_STACK {}, BATCH_SIZE {},\nLEARNING_RATE {}, EPSILON_END {}, MEMORY_SIZE {}'
        .format(
            IMG_RESIZE, args.mean_reward_goal, args.frame_stack, BATCH_SIZE, LEARNING_RATE, EPSILON_END, MEMORY_SIZE
        )
    )
    writer = SummaryWriter()
    
    game = GameWrapper(frame_stack=args.frame_stack, hideScreen=True, state_mode='image')
    replay_memory = replayMemory(MEMORY_SIZE)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    net = DQN(game.outshape, game.n_actions).to(device)
    logging.info(net)
    tgt_net = DQN(game.outshape, game.n_actions).to(device)
    agent = Agent(game, replay_memory)
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    mean_reward = 0
    best_mean_reward = 0
    time_start_frame = 0
    time_start = time.perf_counter()
    iframe = 0
    while True:
        iframe += 1
        epsilon = max(EPSILON_END, EPSILON_START - iframe * (EPSILON_START - EPSILON_END) / EPSILON_STEP)
        episode_finish_reward = agent.play_step(net, epsilon, device)
        # Show episode reward in Tensorboard
        if episode_finish_reward is not None:
            total_rewards.append(episode_finish_reward)
            mean_reward = np.mean(total_rewards[-100:]) * CLIP_REWARD
            speed = (iframe - time_start_frame) / (time.perf_counter() - time_start)
            time_start_frame = iframe
            time_start = time.perf_counter()
            logging.info('{}: done {} games, mean reward {:.2f}, epsilon {:.2f}, speed {:.2f} f/s.'.format(
                iframe, len(total_rewards), mean_reward, epsilon, speed
            ))
            writer.add_scalar('epsilon', epsilon, iframe)
            writer.add_scalar('reward', episode_finish_reward * CLIP_REWARD, iframe)
            writer.add_scalar('mean_reward_100', mean_reward, iframe)
            writer.add_scalar('speed', speed, iframe)
            # update best reward and save model
            if mean_reward > best_mean_reward:
                torch.save(net.state_dict(), f'model_out/{filename}.pt')
                best_mean_reward = mean_reward
                if best_mean_reward >= args.mean_reward_goal:
                    print(f'Game solved in {iframe} frames!')
                    print(f'We achieved {best_mean_reward} mean reward for 100 games!')
                    break

        if len(replay_memory) < MEMORY_SIZE:
            continue

        if iframe % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        
        optimizer.zero_grad()
        batch = replay_memory.sample(BATCH_SIZE)
        loss = calc_loss(batch, net, tgt_net, device)
        loss.backward()
        optimizer.step()
    writer.close()
    # s, _, _ = game.reset()
    # input_tensor = preprocess(s)
    # for _ in range(100000):
    #     s, r, g = game.step(random.choice((0, 1, 2)))
    #     # time.sleep(0.033 / 6)
    #     # time.sleep(0.33)
    #     if g:
    #         print(preprocess(s).shape)
    #         time.sleep(1.5)
    #         game.reset()
