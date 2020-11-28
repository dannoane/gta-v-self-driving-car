from util import game_input

class GtaVInput(game_input.GameInput):
    def __init__(self):
        self.__w_key = 0x11
        self.__s_key = 0x1f
        self.__a_key = 0x1e
        self.__d_key = 0x20
        self.__space_key = 0x39
        self.__prev_key = 0

    def stop_press(self):
        self.release_key(self.__prev_key)

    def move_forward(self, release = False):
        self.press_key(self.__w_key)
        self.__prev_key = self.__w_key
        if release:
            self.stop_press()

    def move_backwards(self, release = False):
        self.press_key(self.__s_key)
        self.__prev_key = self.__s_key
        if release:
            self.stop_press()

    def steer_left(self, release = False):
        self.press_key(self.__a_key)
        self.__prev_key = self.__a_key
        if release:
            self.stop_press()

    def steer_right(self, release = False):
        self.press_key(self.__d_key)
        self.__prev_key = self.__d_key
        if release:
            self.stop_press()

    def hand_break(self, release = False):
        self.press_key(self.__space_key)
        self.__prev_key = self.__space_key
        if release:
            self.stop_press()
        