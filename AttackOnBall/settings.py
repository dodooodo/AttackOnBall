# import random


class Settings:
    def __init__(self):
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 325

        self.BG_HEIGHT = 245
        self.BG_COLOR = (255, 245, 153)

        # start from 245
        self.FLOOR_HEIGHT = 80
        self.FLOOR_COLOR = (255, 74, 46)

        self.PLAYER_SPEED = 1 # 2.5

        self.MAX_BALLS = 15
        self.GIFT_FALL_SPEED = 1

        # self.BALL_SPEED = random.uniform(2.2, 2.5) / 9

        self.fps = 120