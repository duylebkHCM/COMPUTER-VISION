import numpy as np

class LearningRateDecay:
    def __init__(self, factor, initAlpha, dropEvery):
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def step_decay(self, epoch):
        # initialize the base initial learning rate, drop factor, and
        # epochs to drop every
        initAlpha = self.initAlpha
        factor = self.factor
        dropEvery = self.dropEvery 
        # compute learning rate for the current epoch
        alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
        # return the learning rate
        return float(alpha)