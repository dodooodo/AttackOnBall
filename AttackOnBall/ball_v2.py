import pygame
import random
import numpy as np


class Ball:
    def __init__(self, game):
        self.screen = game.screen
        self.color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (200, 0, 255)])
        self.radius = random.randint(25, 40)
        self.direction = game.ball_dir
        if self.direction == 1:
            self.x = -random.randint(50, 70)
        else:
            self.x = random.randint(650, 670)
        xs = np.linspace(1., 1.25, 50, endpoint=False)
        ys = np.linspace(-1.7, -2.7, 50, endpoint=False)
        self.xspeed, self.yspeed0 = random.choice([*zip(xs, ys)])
        # self.xspeed = 1.15
        # self.yspeed0 = -2.7
        # self.xspeed = random.choice((0.55, 0.8))
        # self.xspeed = 0.75
        # self.xspeed = 0.5
        # self.yspeed0 = random.choice(np.linspace(-2., -3., 50, endpoint=False))
        # self.yspeed0 = -3.
        # self.yspeed0 = -2.1
        self.yspeed = self.yspeed0
        self.y0 = game.BG_HEIGHT - self.radius + 9
        self.y = self.y0
        self.G = 0.02
        # self.G = 0.03

    # update ball position using parabola function  
    def move(self):
        self.x += self.xspeed * self.direction
        self.y += self.yspeed
        self.yspeed += self.G
        if round(self.y) == round(self.y0):
            self.yspeed = self.yspeed0


    # check if ball reach the end point
    def isfinish(self):
        if self.direction == 1 and self.x >= 600 + self.radius:
            return True
        elif self.direction == -1 and self.x <= - self.radius:
            return True
        return False
    
    def blitme(self):
        pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius, 0) # ball 
        pygame.draw.circle(self.screen, (0,0,0), (self.x, self.y), self.radius, 1) # boundary
    
    def inscreen(self):
        if -self.radius < self.x < self.screen.get_width() + self.radius:
            return True
        return False

    # def test(self):
    #     x1 = math.sqrt((self.BG_HEIGHT - self.radius) / self.a) + self.h
    #     x2 = -(math.sqrt((self.BG_HEIGHT - self.radius) / self.a) + self.h)
    #     if abs(x1 - x2) % self.xspeed < 0.001:
    #         print('x1:', x1, 'x2:', x2, 'h:', self.h)
    #         return True