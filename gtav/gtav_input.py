from util import game_input

class GtaVInput(game_input.GameInput):
    def __init__(self):
        self.__w_key = 0x11
        self.__s_key = 0x1f
        self.__a_key = 0x1e
        self.__d_key = 0x20
        self.__space_key = 0x39
        self.__prev_keys = []

    def stop_press(self):
        while len(self.__prev_keys) > 0:
            self.release_key(self.__prev_keys.pop())

    def _push_to_stack(self, key):
        if len(self.__prev_keys) > 0 and self.__prev_keys[-1] == key:
            return
        self.__prev_keys.append(key)

    def move_forward(self, release = False):
        if release:
            self.stop_press()
        self.press_key(self.__w_key)
        self._push_to_stack(self.__w_key)

    def move_backwards(self, release = False):
        if release:
            self.stop_press()
        self.press_key(self.__s_key)
        self._push_to_stack(self.__s_key)

    def steer_left(self, release = False):
        if release:
            self.stop_press()
        self.press_key(self.__a_key)
        self._push_to_stack(self.__a_key)

    def steer_right(self, release = False):
        if release:
            self.stop_press()
        self.press_key(self.__d_key)
        self._push_to_stack(self.__d_key)

    def hand_break(self, release = False):
        if release:
            self.stop_press()
        self.press_key(self.__space_key)
        self._push_to_stack(self.__space_key)
        