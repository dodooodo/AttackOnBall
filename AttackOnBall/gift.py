import pygame
import random

class Gift:
    def __init__(self, game):
        self.screen = game.screen
        self.FONT = pygame.font.SysFont('jf-openhuninn-1.1', 25)
        self.x = random.randint(20, 580)
        # self.x = 580
        self.y = 0
        self.FALL_SPEED = game.GIFT_FALL_SPEED
        self.COLORs = (255, 0, 0), (0, 173, 173), (0, 0, 255), (0, 255, 0), (200, 0, 255), 
        self.score = random.randint(1, 3)
        self.appear = False
    
    def blitme(self):
        text = self.FONT.render(str(self.score), 0, self.get_color())
        self.screen.blit(text, (self.x, self.y))
        if self.y < 232:
            self.y += self.FALL_SPEED
    
    def get_color(self):
        t = pygame.time.get_ticks() % 4000
        if t < 800: i = 0
        elif t < 1600: i = 1
        elif t < 2400: i = 2
        elif t < 3200: i = 3
        elif t < 4000: i = 4
        return self.COLORs[i]

    def reset(self):
        self.y = 0
        self.x = random.randint(20, 580)
        # self.x = 580
        self.score = random.randint(1, 3)
        self.appear = True
