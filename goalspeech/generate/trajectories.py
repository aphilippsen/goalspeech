import numpy as np

class DMP():

    def __init__(self, num_bfs, num_dim, k, sigma, tau = 1):
        self.num_dim = num_dim        # dimension of controlled variables

        self.num_bfs = num_bfs                       # number of basis functions
        self.sigma = sigma                           # width of basis functions
        self.centers = np.arange(1, 0, -1/num_bfs)   # basis function centers
        self.weights = np.zeros((num_dim, num_bfs))    # basis function weights

        self.tau = tau                # time constant
        self.alpha = 0.1              # decay of the canonical system
        self.k = k                    # spring constant
        self.d = 2 * np.sqrt(self.k)  # damping constant

        self.minV = 1e-3                   # convergence criterion for movgen method
        self.maxSteps = 10000              # convergence criterion for movgen method

        self.g = np.zeros((1, self.num_dim))           # goal position/offset
        self.x = np.zeros((1, self.num_dim))           # position
        self.v = np.zeros((1, self.num_dim))           # velocity
        self.a = np.zeros((1, self.num_dim))           # acceleration
        self.c = 1                                   # canonical system

        self.start = []
        self.stop = []
        self.userdata = []

    def initialize(self, x = None, g = None):
        if not x is None:
            self.x = x
        if not g is None:
            self.g = g
        self.c = 1
        self.a = np.zeros((1, np.size(self.x)))
        self.v = np.zeros((1, np.size(self.x)))

    def update(self):
        self.updateCanonicalSystem()
        self.updateTransformationSystem()

    def updateCanonicalSystem(self):
        self.c = self.c - self.alpha
        if self.c < 0:
            self.c = 0


    def updateTransformationSystem(self):
        self.a = self.k * (self.g - self.x) - self.d * self.v + self.f()
        self.v = self.v + (1 / self.tau) * self.a
        self.x = self.x + (1 / self.tau) * self.v

    def f(self):
        h = self.calcBasisFunctionResponses()
        y = np.dot(h, self.weights)
        return y

    def calcBasisFunctionResponses(self):
        if self.c > 0:
            h = np.exp(-1/(2*(self.sigma**2)) * (self.c - self.centers)**2)
            h = h / np.sum(h)
        else:
            h = np.zeros((1, self.num_bfs))
        return h

    def prepareDMPTraining(self, X):
        # set canonical system decay according to number of trajectory timesteps
        self.alpha = 1 / (np.size(X,0) + 1)
        self.c = 1

    def getTarget(self, x, v, a):
        ydes = -self.k * (self.stop - x) + self.d * v + self.tau * a
        return ydes

    def train(self, X, evaluate = False):
        num_timesteps = X.shape[0]

        # calculate derivatives
        V = np.append(np.diff(X,axis=0), np.zeros((1, self.num_dim)),axis=0)
        A = np.append(np.diff(V, axis=0), np.zeros((1, self.num_dim)),axis=0)

        # preparations
        self.start = np.reshape(X[0,:], (1,len(X[0,:])))
        self.stop = np.reshape(X[-1,:], (1,len(X[-1,:])))
        self.prepareDMPTraining(X)
        H = np.zeros((num_timesteps, self.num_bfs))
        Y = np.zeros((num_timesteps, self.num_dim))
        C = np.zeros((num_timesteps, 1)) # for evaluation only

        # collect data
        for i in range(num_timesteps):
            H[i,:] = self.calcBasisFunctionResponses()
            Y[i,:] = self.getTarget(X[i,:], V[i,:], A[i,:])
            C[i] = self.c
            self.updateCanonicalSystem()

        # determine weights by regression
        self.weights = np.linalg.lstsq(np.dot(np.transpose(H),H), np.dot(np.transpose(H),Y), rcond=None)[0]

        if evaluate:
            self.evaluate(X, C, H, Y)

    def movgen(self, weights, x = None, g = None):
        self.weights = weights
        if x is None:
            x = self.start
        if g is None:
            g = self.stop

        X = []
        self.initialize(x, g)
        step = 1
        while (np.sum(np.abs(self.v)) > self.minV or step < 10) and (step < self.maxSteps):
            self.update()
            X.append(self.x)
            step += 1

        return np.concatenate(X)

    def evaluate(self, X, C, H, Y):
        pass





