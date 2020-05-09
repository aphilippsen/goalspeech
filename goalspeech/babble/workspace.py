import numpy as np

from goalspeech.utils.metric import ScaledEucDistance
from goalspeech.ml.clustering import ITM

class WorkspaceModel:

    def __init__(self, initialTask, targetDist, radius = 0.1, wThreshold = 0.5):
        self.metric = ScaledEucDistance(1)

        self.radius = radius
        self.wThrNodeCreation = wThreshold

        self.S = ITM(0, self.radius)
        self.S.cluster(initialTask)

        # history gathered for each node in the WSM
        # TODO could also be gathered separately for the targetDist clusters?!
        # list of lists of [timestep, value] pairs
        self.wHistory = [[[0,0]]]
        self.cHistory = [[[0,0]]]
        self.priorsHistory = []

        self.targetDist = targetDist
        self.scalingFactor = 1
        for i in range(targetDist.n_components):
            for j in range(targetDist.n_components):
                if i != j:
                    newDist = self.metric.distance(self.targetDist.means_[i], self.targetDist.means_[j])
                    self.scalingFactor = min(self.scalingFactor, newDist)

        if self.scalingFactor == 0:
            print("WorkspaceModel: two targetDist clusters appear to be the same?! Set scalingFactor=1!")
            self.scalingFactor = 1
        #else:
        #    self.scalingFactor = self.scalingFactor[0]

        self.alpha = 1e-3
        self.historyLength = 25


    def update(self, timestep, tRolloutsExpl, weights = [], competences = []):
        numSamples = np.size(tRolloutsExpl,0)
        if len(weights) == 0:
            weights = np.ones(numSamples)

        for i in range(numSamples):
            d = self.metric.distance(np.array(self.S.clusters()), tRolloutsExpl[i,:])
            minDist = np.min(d)
            sIdx = np.argmin(d)

            # if distance to current WSM is large enough
            if minDist > self.radius:
                # supervised and weight-dependent node creation
                if weights[i] < self.wThrNodeCreation:
                    print("WorkspaceModel: node not created because of small weights")
                else:
                    print("WorkspaceModel: new node")
                    self.S.cluster(tRolloutsExpl[i,:])
                    self.cHistory.append([])
                    self.wHistory.append([])
                    sIdx = self.S.numClusters()-1 # link to last node

            # update history
            self.wHistory[sIdx].append([timestep, weights[i]])
            try:
                self.cHistory[sIdx].append([timestep, competences[i]])
            except:
                self.cHistory[sIdx].append([timestep, 0])


    # distance to WSM (for adaptive noise)
    def calcDistanceToWSM(self, point):
        wsClusters = self.S.clusters()

        d = self.metric.distance(np.array(wsClusters), point)
        minDist = np.sort(d)[0] # minimal dist
        minIdx = np.argsort(d)[0] # index of node with minimal dist

        return (minDist/self.scalingFactor, minIdx)

    def getActionNoiseLevel(self, point):
        (dist, idx) = self.calcDistanceToWSM(point)
        #return -(np.exp(-2 * dist) - 1)
        return (1 - np.exp(-4 * dist))

    # active target selection

    def activeTargetSelection(self, drawFromTargetDist, activeSelection):
        """
            drawFromTargetDist: if True, draw from target dist, if False,
                draw from WS cluster nodes.
            activeSelection: if False, select randomly, if True, use
                active selection
        """

        if drawFromTargetDist:
            # draw from target distribution

            if activeSelection:
                # adapt priors according to competence in this region
                # get the distance to the WSM and treat is as a probability
                for n in range(self.targetDist.means_.shape[0]):
                    dist, _ = self.calcDistanceToWSM(self.targetDist.means_[n])
                    self.targetDist.weights_[n] = (1 - np.exp(-4 * dist))
                # adjust priors (just change them in place)

            self.targetDist.weights_ /= np.sum(self.targetDist.weights_)
            print("Updated target distribution weights to: " + str(self.targetDist.weights_))
            distribution = self.targetDist

        else:
            # draw from WS cluster nodes

            priors = np.ones(self.S.numClusters())
            if activeSelection:
                print("Warning: active selection from WS clusters not yet implemented")
                # TODO adjust priors variable, make sure no prior is zero

            distribution = GaussianMixture(n_components = self.S.numClusters(), weights_init=priors)
            distribution.fit(self.S.clusters())


        (seed, currentCluster) = distribution.sample()
        print("Drawn new sample for exploration " + str(seed) + " (cluster " + str(currentCluster) + ")")
        return (seed, int(currentCluster))

