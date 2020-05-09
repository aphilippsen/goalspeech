import numpy as np

class Metric:
    def __init__(self, m = 2):
        self.m = m

    def distance(self, a, b):
        # default: euc distance
        return np.power(np.sum(np.power(self.preprocess(a) - self.preprocess(b), self.m),axis=1), 1./self.m)

    def preprocess(self, a):
        if a.dtype == object:
            a = np.concatenate(np.reshape(a, (-1,1)), axis=0)

        if a.ndim == 1:
            a = np.reshape(a, (1,-1))
        return a

class ScaledEucDistance(Metric):

    def __init__(self, s = 1, m = 2):
        Metric.__init__(self, m)
        self.s = s

    def distance(self, a, b):
        # default: euc distance
        return np.power(np.sum(np.power((self.preprocess(a) - self.preprocess(b))/self.s, self.m),axis=1), 1./self.m)
