import pygame
import sys, math
# from decimal import Decimal
from settings import Settings
from player import Player
from ball import Ball
from gift import Gift
import random


class AttackOnBall(Settings):
    def __init__(self, gamemode='human', hideScreen=False):
        super().__init__()
        # initiate pygame and font module
        pygame.init()
        pygame.font.init()

        self.settings = Settings()

        # create screen
        self.screen = pygame.display.set_mode(
            (self.settings.SCREEN_WIDTH, self.settings.SCREEN_HEIGHT),
            flags=pygame.HIDDEN if hideScreen else 0,
        )
        
        pygame.display.set_caption('Attack On Ball')

        # create background
        self.background = pygame.Rect(0, 0, self.settings.SCREEN_WIDTH, self.settings.BG_HEIGHT)
        self.ground = pygame.Rect(0, self.settings.BG_HEIGHT, self.settings.SCREEN_WIDTH, self.settings.FLOOR_HEIGHT)

        # create needed object
        self.player = Player(self)
        self.balls = []
        self.gift = Gift(self)
        self.font = pygame.font.SysFont('jf-openhuninn-1.1', 25)
        self.last_tick = 0
        self.score = 0
        self.bonus = 0
        self.ball_dir = -1

    def start(self):
        i = -1
        clock = pygame.time.Clock()
        # start game
        while True:
            i = i + 1
            player_move = False
            ''' --------------Time Control---------------'''
            clock.tick(self.settings.fps)

            '''---------------Player Move------------------'''
            # watch event
            for event in pygame.event.get():
                # exit game
                if event.type == pygame.QUIT:
                    sys.exit()
                # detect key event
                elif event.type == pygame.KEYDOWN:
                    player_move = True
                    if event.key == pygame.K_LEFT:
                        self.player.left()

                    elif event.key == pygame.K_RIGHT:
                        self.player.right()

            # detect key press
            key_pressed = pygame.key.get_pressed()

            if key_pressed[pygame.K_LEFT]:
                player_move = True
                self.player.left()

            elif key_pressed[pygame.K_RIGHT]:
                player_move = True
                self.player.right()
            
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
            # add a ball to screen
            if i % 25 == 0 and len(self.balls) < self.settings.MAX_BALLS:
                newball = Ball(self.screen, self.settings.BG_HEIGHT, -1)
                self.balls.append(newball)
            # balls move
            [ball.move() for ball in self.balls]
            # delete ball if it reaches end.
            for b in range(len(self.balls)-1, -1, -1):
                if self.balls[b].isfinish():
                    self.balls.pop(b)
                
            '''--------------------Bonus--------------------------'''
            # get bonus when player touches gift
            if self.distance(self.player.cx, self.player.cy, self.gift.x, self.gift.y) < 20:
                self.bonus += self.gift.score
                self.gift.reset()

            '''---------------------Score---------------------------'''
            # score = time + bonus
            time = round(i / self.settings.fps, 2)
            self.score  = f'{time + self.bonus:.2f}'
            text = self.font.render(str(self.score), 0, (0, 0, 0))

            ''' ----------------Draw Screen--------------------'''

            # draw background
            self.screen.fill(self.settings.BG_COLOR, self.background)
            self.screen.fill(self.settings.FLOOR_COLOR, self.ground)

            # draw player
            self.player.blitme()
            
            # draw balls
            [ball.blitme() for ball in self.balls]

            # draw gift
            self.gift.spawn()

            # draw score
            self.screen.blit(text, (5, 5))
            
            # update screen
            # pygame.draw.circle(self.screen, (0, 0, 0), (self.player.cx, self.player.cy), 5, 0)
            pygame.display.flip()

            ''' ----------------Game Over-----------------'''
            
            # game over if player touch ball
            if self.collide():
                self.restart()
                i = -1
            
            ''' --------------Time Control---------------'''

            # pygame.time.delay(1)

            '''  --------------------------------------   '''

    def collide(self):
        for ball in self.balls:
            if self.distance(self.player.cx, self.player.cy, ball.x, ball.y) < (40-ball.radius)*0.4+(ball.radius):
                return True
        return False
    
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    def restart(self):
        self.player.reset()
        self.balls = []
        self.last_tick = pygame.time.get_ticks()
        self.score = 0
        self.bonus = 0
        self.ball_dir = random.choice((-1, 1))

if __name__ == '__main__':
    game = AttackOnBall()
    game.start()
    
