import numpy as np

from goalspeech.ml.clustering import ITM
from goalspeech.ml.learner import Learner

class OnlineWeightedRBF(Learner):
    def __init__(self, inpDim, outDim, invmodelSpecs = {}):
        Learner.__init__(self, inpDim, outDim, invmodelSpecs)

        # set required parameters (if not defined in invmodelSpecs)
        self.radius = 0.1
        try:
            self.radius = float(invmodelSpecs.pop('radius'))
        except:
            pass
        
        self.weightThreshold = 0.1
        try:
            self.weightThreshold = float(invmodelSpecs.pop('weightthreshold'))
        except:
            pass

        # underlying clustering algorithm to partition input space
        self.center = ITM(0.01, self.radius)

    def init(self, X, Y):
        """
        Initialize with a single known X, Y training pair
        """
        self.center.cluster(X)

        if Y.ndim == 2:
            Y = np.reshape(Y, (-1,))
        self.centerOutputs = [Y]

    def train(self, X, Y, weights = []):

        if X.dtype == object:
            X = np.concatenate(X, axis=0)
        if Y.dtype == object:
            Y = np.concatenate(X, axis=0)

        if X.ndim == 1:
            X = np.reshape(X, (1,-1))
        if Y.ndim == 1:
            Y = np.reshape(Y, (1,-1))
        numSamples = np.size(X,0)

        if len(weights) == 0:
            weights = np.ones(numSamples)

        if all(weights == 0):
            return

        p = np.random.permutation(numSamples)
        for i in range(numSamples):

            if weights[p[i]] > self.weightThreshold:

                # how the learner currently estimates X
                yhat = self.apply(X[p[i],:])
                yhat = yhat[0] # only one sample

                # update clustering
                self.center.cluster(X[p[i],:])

                # if new center was added in clustering
                if np.size(self.center.C,0) > len(self.centerOutputs):
                    # set own estimation
                    self.centerOutputs.append(yhat)
                    print("OnlineWeightedRBF: new neuron " + str(len(self.centerOutputs)) + " added")

                # adapt local model according to estimation error
                yhat = self.apply(X[p[i],:])
                # after this, network activations (self.h) are set
                yerror = Y[p[i],:] - yhat
                # update centers
                for j in range(np.size(self.center.C,0)):
                    self.centerOutputs[j] += self.lrate * self.h[j] * weights[p[i]] * yerror[0,:]

    def apply(self, X):
        """
        """
        if X.dtype == object:
            X = np.concatenate(X, axis=0)

        if X.ndim == 1:
            X = np.reshape(X, (1,-1))
        numSamples = np.size(X,0)

        yhat = np.zeros(shape=(numSamples, self.outDim))
        for i in range(numSamples):
            # normed distances of cluster centers to data point
            dists = self.center.clusters() - np.repeat(np.reshape(X[i,:], (1,-1)), self.center.numClusters(), axis=0)
            self.a = np.sum(np.power((dists/self.radius),2), axis=1)
            self.h = np.exp(-self.a)

            if self.softmax:
                self.n = np.sum(self.h)
                if self.n != 0:
                    self.h = self.h / self.n

            # create yhat as a sum of local estimations
            for j in range(self.center.numClusters()):
                # xlocal = (X[i,:] - self.center.C[j,:]) / self.radius
                yhat[i,:] += self.h[j] * self.centerOutputs[j]

        return yhat
