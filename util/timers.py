import time

class Timers:
    def __init__(self):
        self.timers = dict()

    def start(self, label = ''):
        self.timers[label] = self.current_time_ms()

    def end(self, label = ''):
        if label not in self.timers:
            raise Exception('{} does not exist'.format(label))
        
        current_time = self.current_time_ms()
        print('{}ms {}'.format(current_time - self.timers[label], label))
        self.timers[label] = current_time

    def current_time_ms(self):
        return int(round(time.time() * 1000))

