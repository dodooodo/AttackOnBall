import pygame
import random
import math

# 4a(y - k) = (x - h)**2
# y = (x - h)**2 / a + k
# 2.2 <= a <= 8.8
class Ball:
    def __init__(self, game):
        self.testi = 0
        self.prevy = 0
        self.screen = game.screen
        self.BG_HEIGHT = game.BG_HEIGHT + 4
        self.color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (200, 0, 255)])
        self.radius = random.randint(25, 40)
        self.xspeed = game.BALL_SPEED
        self.a = random.randint(10 , 30)
        self.k = random.randint(10, 110)
        self.direction = game.ball_dir
        if self.direction == 1:
            self.x = -50
            self.h = -50
        else:
            self.x = 650
            self.h = 650
        self.updateY()

    # update ball position using parabola function  
    def move(self):
        self.x = self.x + self.xspeed * self.direction
        self.updateY()
        # update parabola if ball reach ground
        if self.y > self.BG_HEIGHT - self.radius:
            self.h += 2 * (self.x - self.h)
            # if self.y - (self.BG_HEIGHT - self.radius) < 1:
            # if self.testi > 0 and abs(self.prevy - self.y) > 0.1:
            # print(self.y)
                # print(
                #     f'y reach ground: x:{self.x}, y:{self.y}, radius: {self.radius}, xspeed: {self.xspeed}, a: {self.a}, k: {self.k}'
                # )
            # self.testi += 1
            # self.prevy = self.y

    # y = a(x - h)**2 + k
    def updateY(self):
        self.y = (self.x - self.h)**2 / self.a + self.k

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
    

    # def test(self):
    #     x1 = math.sqrt((self.BG_HEIGHT - self.radius) / self.a) + self.h
    #     x2 = -(math.sqrt((self.BG_HEIGHT - self.radius) / self.a) + self.h)
    #     if abs(x1 - x2) % self.xspeed < 0.001:
    #         print('x1:', x1, 'x2:', x2, 'h:', self.h)
    #         return True