import pygame
from pathlib import Path

images = Path(__file__).parent.joinpath('images')


class Player:
    def __init__(self, game):
        # initialize player
        self.screen = game.screen
        self.speed = game.PLAYER_SPEED

        # load img
        self.standr = pygame.image.load(str(images.joinpath('stand-right.png'))).convert_alpha()
        self.standl = pygame.image.load(str(images.joinpath('stand-left.png'))).convert_alpha()
        self.run1r = pygame.image.load(str(images.joinpath('run1-right.png'))).convert_alpha()
        self.run2r = pygame.image.load(str(images.joinpath('run2-right.png'))).convert_alpha()
        self.run3r = pygame.image.load(str(images.joinpath('run3-right.png'))).convert_alpha()
        self.run1left = pygame.image.load(str(images.joinpath('run1-left.png'))).convert_alpha()
        self.run2left = pygame.image.load(str(images.joinpath('run2-left.png'))).convert_alpha()
        self.run3left = pygame.image.load(str(images.joinpath('run3-left.png'))).convert_alpha()
        
        self.image = self.standr
        self.rect = self.image.get_rect() # width and height = 40

        self.x = 285
        self.y = 216
        self.cx = 285 + self.rect.width/2 + 3
        self.cy = self.y + self.rect.height/2 + 3 
        # print(self.rect.width, self.SCREEN_WIDTH, self.rect.height, self.SCREEN_HEIGHT)
        self.face = 1
        self.is_moving = False
        self.die = False
        self.score = 0
        self.bonus = 0
        self.dir = 0

    
    def blitme(self):
        # draw player on the screen
        self.screen.blit(self.image, (self.x, self.y))


    def right(self, dir = 1):
        self.x += self.speed * dir
        self.cx += self.speed * dir
        self.image = self.get_move_image(self.run2r, self.run1r, self.run3r)
        self.face = 1
        self.is_moving = True
        self.dir = dir


    def left(self, dir = 1):
        self.x -= self.speed * dir
        self.cx -= self.speed * dir
        self.image = self.get_move_image(self.run2left, self.run1left, self.run3left)
        self.face = -1
        self.is_moving = True
        self.dir = dir
    
    # decide which side to face when stop moving
    def stop(self):
        if self.face == 1:
            self.image = self.standr
            self.is_moving = False
        else:
            self.image = self.standl
            self.is_moving = False
    

    # def reset(self):
    #     self.image = self.standr
    #     self.x = 285
    #     self.y = 215
    #     self.cx = 285 + self.rect.width/2 + 3
    #     self.cy = 215 + self.rect.height/2 + 3
    #     self.face = 1
    #     self.is_moving = False
    #     self.die = False
    #     self.bonus = 0
    #     self.dir = 0
        # self.model = Model(1+self.MAX_BALLS*5)
    
    # return moving image
    def get_move_image(self, im1, im2, im3):
        t = pygame.time.get_ticks() % 90
        if t < 30: return im1
        elif t < 60: return im2
        else: return im3

