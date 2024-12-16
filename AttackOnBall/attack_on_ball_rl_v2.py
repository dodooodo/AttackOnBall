import pygame
import math
import random
from settings import Settings
from player import Player
from ball_v2 import Ball
from gift import Gift


class AttackOnBall(Settings):
    def __init__(self, hideScreen=False, state_mode='tuples'):
        super().__init__()
        self.hideScreen = hideScreen
        # rl
        assert state_mode == 'tuples' or state_mode == 'image', 'state_mode must be "tuples" or "image".'
        self.state_mode = state_mode
        # self.observation_shape = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT) if state_mode == 'image'
        self.n_actions = 3

        # initiate pygame and font module
        pygame.init()
        pygame.font.init()

        # create screen
        # self.screen = pygame.display.set_mode(
        #     (self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
        #     flags=pygame.HIDDEN if hideScreen else 0,
        # )
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # pygame.display.set_caption('Attack On Ball')

        # create background
        self.background = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.BG_HEIGHT)
        self.ground = pygame.Rect(0, self.BG_HEIGHT, self.SCREEN_WIDTH, self.FLOOR_HEIGHT)

        # create needed object
        # self.font = pygame.font.SysFont('jf-openhuninn-1.1', 25)


    def step(self, action: int) -> tuple:
        # start game
        assert self.gameover == False, 'Game Over. Please reset the game'
        self.iframe += 1
        # rl
        self.reward = 0
        ''' --------------Time Control---------------'''
        # clock.tick()

        ''' ------------ watch user event -----------------'''
        # watch event
        # if not self.hideScreen:
        #     for event in pygame.event.get():
        #         # exit game
        #         if event.type == pygame.QUIT:
        #             pygame.quit()

        '''---------------Player Move------------------'''   
        if action == 0:
            player_move = False
        elif action == 1:
            self.player.left()
            player_move = True
        elif action == 2:
            self.player.right()
            player_move = True
        else:
            raise ValueError('Available actions: 0, 1, 2')

        # boundary for player
        if self.player.x < -5:
            self.player.left(-1)

        elif self.player.x >= 565:
            self.player.right(-1)
        
        # where to face when stop moving
        if not player_move:
            if self.player.face == 1:
                self.player.image = self.player.standr
            else:
                self.player.image = self.player.standl
        
        ''' ------------------Ball-------------------------'''
        # add a ball to screen every half second
        if self.iframe % (self.fps * 0.75) == 0 and len(self.balls) < self.MAX_BALLS:
            newball = Ball(self)
            self.balls.append(newball)
            self.ball_dir *= -1
        # balls move
        [ball.move() for ball in self.balls]
        # delete ball if it reaches end.
        for b in range(len(self.balls)-1, -1, -1):
            if self.balls[b].isfinish():
                self.balls.pop(b)
            
        '''--------------------Bonus--------------------------'''
        self.gift_clock += 1
        # get bonus when player touches gift
        if self.gift.appear and self.distance(self.player.cx, self.player.cy, self.gift.x, self.gift.y) < 20:
            self.score += self.gift.score
            self.gift_clock = 0
            self.gift.appear = False
            # rl
            self.reward += self.gift.score

        # reset gift every 4 seconds
        if not self.gift.appear and self.gift_clock == 4 * self.fps:
            self.gift.reset()

        '''---------------------Score---------------------------'''
        self.score += 1 / self.fps
        text = self.font.render(f'{self.score:.2f}', 0, (0, 0, 0))

        ''' ----------------Draw Screen--------------------'''

        # draw background
        self.screen.fill(self.BG_COLOR, self.background)
        self.screen.fill(self.FLOOR_COLOR, self.ground)

        # draw player
        self.player.blitme()
        
        # draw balls
        [ball.blitme() for ball in self.balls if ball.inscreen()] 

        # draw gift
        if self.gift.appear:
            self.gift.blitme()

        # draw score
        # self.screen.blit(text, (5, 5))
        
        # update screen
        # pygame.draw.circle(self.screen, (0, 0, 0), (self.player.cx, self.player.cy), 5, 0)
        # pygame.display.flip()
        # pygame.display.update()

        ''' ----------------Game Over-----------------'''
        # game over if player touch ball
        if self.collide():
            # self.gameover = True
            self.reward = 0
        else:
            self.reward += 1 / self.fps
            # break
        # print(self.score)
        # print(time, self.bonus)       
        '''  ---------------yield observation----------------'''
        return (
            self.get_state(),
            self.reward,
            self.gameover,
            self.score,
        )

        '''  --------------------------------------   '''

    def collide(self):
        for ball in self.balls:
            if self.distance(self.player.cx, self.player.cy, ball.x, ball.y) < 20 + ball.radius*0.85:
            # if ((abs(self.player.cx - ball.x) < self.player.rect.width / 2 + ball.radius) and (abs(self.player.cy - ball.y) < self.player.rect.height / 2 + ball.radius)):
                return True
        return False
    
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    def reset(self):
        self.player = Player(self)
        self.balls = []
        self.ball_dir = random.choice((1, -1))
        self.gift = Gift(self)
        self.gift_clock = 0
        self.score = 0
        self.gameover = False
        self.iframe = -1
        # rl
        self.reward = 0
        # first image is blank, so take 1 step first.
        return self.step(0)
        # return (
        #     self.get_state(),
        #     self.score,
        #     self.gameover, 
        # )

    
    def get_state(self):
        if self.state_mode == 'tuples':
            return (
                (self.player.x, self.player.y), 
                tuple((ball.x, ball.y, ball.radius) for ball in self.balls if -ball.radius < ball.x < self.SCREEN_WIDTH + ball.radius),
                (self.gift.x, self.gift.y),
            )
        elif self.state_mode == 'image':
            return pygame.image.tostring(self.screen, 'RGB')     
            # (W, H, C)        
            # return pygame.surfarray.array3d(self.screen)
        else:
            raise ValueError(self.state_mode)
        # image = pygame.image.tostring(self.screen, 'RGB')
        # from PIL import Image

        # image = Image.frombytes('RGB', (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), image)
        # image.save('foo.png')
        # import numpy
        # numpy.frombuffer()

if __name__ == '__main__':
    game = AttackOnBall()
    game.start()
    
