import torch
import math

class Scheduler():
    def __init__(self, func='linear', ilr=0.1, flr=None, epochs=100, wepochs=3):
        assert func in ['linear']

        self.epochs = epochs - wepochs
        self.wepochs = wepochs
        self.ilr = ilr
        self.flr = flr if flr else ilr * 0.01
        self.wlr = self.flr * 0.1
        self.func = getattr(self, func)

    def warmup(self, epoch):
        return self.wlr + (self.ilr - self.wlr) * (epoch/(self.wepochs))

    def linear(self, epoch):
        return self.flr + (self.ilr - self.flr) * (1 - epoch/(self.epochs-1))

    def __call__(self, epoch):
        if epoch < self.wepochs:
            return self.warmup(epoch)
        else:
            return self.func(epoch - self.wepochs)