import numpy as np
import os

import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
from matplotlib import pyplot as plt

from goalspeech.utils.normalize import Normalizer
from goalspeech.generate.trajectories import DMP

class ArticulatoryData:
    """
        Holds articulatory shapes or trajectories, function to rollout
        and care for correct normalization.
        arNorm: normalized, one version of each speech sound
        ar: normalized and vector-reshaped (one sample per line), including all noisy variations
    """

    def __init__(self, arParamsOn, arParamsOff):
        # which parameters should be ignored?
        self.arParamsOn = arParamsOn
        self.arParamsOff = arParamsOff

        # automatically ignore changes in parameters with a variance < self.threshold
        self.threshold = 1e-10

    # functions to be implemented by child classes
    def doNormalization(self):
        """
            call this the first time you normalize something with this instance
        """
        pass

    def normalize(self, original):
        """
            afterwards, this function maps something into normalized space...
        """
        pass

    def denormalize(self, normed):
        """
            ... and back into non-normalized space
        """
        pass

    def generateArVariations(self, numSamplesPerSequence, arNoise):
        """
            vary the articulatory parameters slightly to create noisy samples
        """
        pass

    def rolloutArData(self, data):
        """
            rollout the parameter configuration in time to create a trajectory
        """
        pass


# For speech sounds defined by a single articulatory shape, i.e. vowels
class ArticulatoryShapeData(ArticulatoryData):

    def __init__(self, arParamsOn, arParamsOff, soundDuration = 500, arSamplingRate = 200):
        """
            soundDuration in ms
            arSamplingRate in frames/s
        """
        ArticulatoryData.__init__(self, arParamsOn, arParamsOff)

        # required to rollout trajectories from articulatory shapes
        self.arSamplingRate = arSamplingRate
        self.soundDuration = soundDuration

    def setArticulatoryData(self, arOrig, neutralIdx):
        """
            arOrig: a numpy object array of matrices of size (numParams,)
            neutralIdx: index of the ar shapes used as home gesture
        """
        # original articulatory data as received from VTL
        self.arOrig = arOrig

        self.doNormalization(arOrig)

        self.arHome = self.arNorm[neutralIdx]
        self.arNoise = np.repeat(0.01, np.size(self.arNorm[0]))

        self.ar_dim = self.arNorm[0].shape

    # normalization

    def doNormalization(self, arOrig):
        """ initialize normalizer """
        self.normalizer = Normalizer()
        if np.size(arOrig,0) > 1:
            self.arNorm = self.normalizer.normalizeData(arOrig, margin=0.1)
        elif np.size(arOrig,0) == 1:
            print("ArticulatoryData: Not enough articulatory data for normalization!")
            import sys; sys.exit()
            # TODO: use vocal tract boundaries instead!
        else:
            raise Exception("Error in ArticulatoryData: no articulatory data provided!")

        # remove static articulators
        self.variance = np.var(arOrig)
        self.statics = self.arNorm[0]
        # do NOT use these paramters:
        self.variance[self.arParamsOff > 0] = 0
        # DO use these parameters:
        self.variance[self.arParamsOn > 0] = self.arParamsOn[self.arParamsOn > 0]

        for n in range(np.size(self.arNorm)):
            self.arNorm[n] = self.arNorm[n][self.variance > self.threshold]

    def normalize(self, original):
        if original.dtype == object:
            res = np.empty(np.size(original), dtype=object)
            for i in range(np.size(original,0)):
                res[i] = self.normalizer.range2norm(original[i])
                res[i] = res[i][self.variance > self.threshold]
            return res
        else:
            res = self.normalizer.range2norm(original)
            if original.ndim == 1:
                return res[self.variance > self.threshold]
            else:
                return res[:,self.variance > self.threshold]

    def denormalize(self, normed):

        if normed.dtype == object:
            res = np.empty(np.size(normed), dtype=object)
            for i in range(np.size(normed,0)):
                x = self.denormalize(normed[i])
                res[i] = x
            return res
        else:
            if normed.ndim == 1:
                normed = np.reshape(normed, (1,-1))

            res = np.zeros(shape=(np.size(normed,0), len(self.variance)))
            norm_i = 0
            for i in range(np.size(self.variance)):
                if self.variance[i] < self.threshold:
                    res[:,i] = self.statics[i]
                else:
                    res[:,i] = normed[:,norm_i]
                    norm_i += 1

            return self.normalizer.norm2range(res)

    # rollout
    def rolloutArData(self, data):
        numTimeSamples = self.soundDuration / 1000.0 * self.arSamplingRate
        if data.dtype == object:
            rolledOut = np.empty(np.size(data), dtype = object)
            for i in range(np.size(data)):
                denormed = self.denormalize(data[i])
                rolledOut[i] = np.repeat(denormed, numTimeSamples, axis=0)
        else:
            if data.ndim == 1 or (data.ndim == 2 and np.size(data,0) == 1):
                rolledOut = np.repeat(self.denormalize(data), numTimeSamples, axis=0)
            else:
                # matrix numSample x numParams
                rolledOut = np.empty(np.size(data,0), dtype = object)
                for i in range(np.size(data,0)):
                    denormed = self.denormalize(data[i,:])
                    rolledOut[i] = np.repeat(denormed, numTimeSamples, axis=0)

        return rolledOut

    def generateArVariations(self, numSamplesPerSequence, arNoise = {}):
        numParams = np.size(self.arNorm[0])
        self.numSamplesPerSequence = numSamplesPerSequence
        if arNoise:
            self.arNoise = arNoise
        try:
            assert(np.size(self.arNoise) == numParams)
        except AssertionError:
            raise Exception("ArticulatoryData: expected arNoise to be of size " + str(numParams))

        self.ar = np.empty(shape = (np.size(self.arNorm)*self.numSamplesPerSequence, numParams))

        for i in range(np.size(self.arNorm)):
            startIdx = i * self.numSamplesPerSequence
            endIdx = startIdx + numSamplesPerSequence

            copied = np.repeat(np.reshape(self.arNorm[i], (1,-1)), self.numSamplesPerSequence, axis=0)
            randNoise = np.repeat(np.reshape(self.arNoise, (1,-1)), self.numSamplesPerSequence, axis=0) * np.random.randn(self.numSamplesPerSequence, numParams)

            self.ar[startIdx:endIdx,:] = copied + randNoise


class ArticulatoryTrajectoryData:
    """
        For articulatory trajectories which change in time, described via dynamic movement primitives (DMP).
    """

    def __init__(self, arParamsOn, arParamsOff, dmp_params):
        """
            soundDuration in ms
            arSamplingRate in frames/s
        """
        ArticulatoryData.__init__(self, arParamsOn, arParamsOff)
        self.dmp_params = dmp_params

    def setArticulatoryData(self, arOrig, neutralIdx, print_dmp_results = False, picture_path = '.'):
        """
            arOrig: a numpy object array of matrices of size (numParams,)
            neutralIdx: index of the ar shapes used as home gesture
        """
        # original articulatory data as received from VTL
        self.arOrig = arOrig

        # normalize the trajectory data
        self.doNormalization(arOrig)

        self.ar_dim = self.arNorm[0].shape[1]

        # create DMP representation
        arDmp = self.createDMPdata(self.arNorm, self.arNorm[0].shape[1])

        # evaluate
        if print_dmp_results:
            undo_arDmp = np.empty(np.size(arDmp), dtype=object)
            for i in range(len(arDmp)):
                undo_arDmp[i] = self.playDMP(arDmp[i][1:1+self.dmp_params['num_bfs']], arDmp[i][1,:], arDmp[i][-1,:])
            self.plotDMPrepresentations(self.arNorm, undo_arDmp, picture_path = picture_path)

        # normalization method 1
        arDmp_reshaped = np.empty(np.size(arDmp), dtype=object)
        for i in range(len(arDmp)):
            arDmp_reshaped[i] = arDmp[i].reshape((1,-1))
        # finally, normalize the DMP data again
        self.dmpNormalizer = Normalizer()
        self.arNorm = self.dmpNormalizer.normalizeData(arDmp_reshaped)

        # normalization method 2
        # self.dmpNormalizer = Normalizer()
        # self.arNorm = self.dmpNormalizer.normalizeData(arDmp)

        self.arHome = self.arNorm[neutralIdx]
        self.arNoise = np.tile(0.001, self.arNorm[0].shape)

    # normalization

    def doNormalization(self, arOrig):
        """ initialize normalizer """
        self.normalizer = Normalizer()
        if np.size(arOrig,0) > 1:
            self.arNorm = self.normalizer.normalizeData(arOrig, margin=0.1)

        elif np.size(arOrig,0) == 1:
            print("ArticulatoryData: Not enough articulatory data for normalization, use vocal tract boundaries instead!")
            # TODO
        else:
            raise Exception("Error in ArticulatoryData: no articulatory data provided!")

        # remove static articulators
        self.variance = np.var(np.concatenate(arOrig),axis=0)
        self.statics = self.arNorm[0]
        # do NOT use these paramters:
        self.variance[self.arParamsOff > 0] = 0
        # DO use these parameters:
        self.variance[self.arParamsOn > 0] = self.arParamsOn[self.arParamsOn > 0]

        for n in range(np.size(self.arNorm)):
            self.arNorm[n] = self.arNorm[n][:,self.variance > self.threshold]

    def createDMPdata(self, original, num_dim):
        if original.dtype == object:
            arDmp = np.empty(np.size(original), dtype=object)
            for i in range(len(original)):
                arDmp[i] = self.createDMPdata(original[i], num_dim)
        else:
            self.dmp = DMP(self.dmp_params['num_bfs'], num_dim, self.dmp_params['k'], self.dmp_params['sigma'], self.dmp_params['tau'])
            self.dmp.train(original)
            arDmp = np.concatenate((self.dmp.start, self.dmp.weights, self.dmp.stop))
        return arDmp

    def playDMP(self, weights, x, g):
        #self.dmp.x = x
        #self.dmp.g = g
        #elf.dmp.weights = weights
        traj = self.dmp.movgen(weights, x, g)
        return traj

    def plotDMPrepresentations(self, orig_data, dmp_data, picture_path = '.'):
        fig = plt.figure()
        for i in range(len(orig_data)):
            ax = fig.add_subplot(len(orig_data),2,(i*2)+1)
            for p in range(orig_data[0].shape[1]):
                ax.plot(np.arange(orig_data[i].shape[0]), orig_data[i][:,p])
            ax = fig.add_subplot(len(orig_data),2,(i*2)+2)
            for p in range(dmp_data[0].shape[1]):
                ax.plot(np.arange(dmp_data[i].shape[0]), dmp_data[i][:,p])
        plt.title('Originals (left) and DMP approximations (right)')
        plt.savefig(os.path.join(picture_path, 'arData_dmp.png'))
        # TODO: showing plot? plt.show(block=False)


    def normalize(self, original):
        if original.dtype == object:
            res = np.empty(np.size(original), dtype=object)
            for i in range(np.size(original,0)):
                res[i] = self.normalizer.range2norm(original[i])
                res[i] = res[i][self.variance > self.threshold]
            return res
        else:
            res = self.normalizer.range2norm(original)
            if original.ndim == 1:
                return res[self.variance > self.threshold]
            else:
                return res[:,self.variance > self.threshold]

    def denormalize(self, normed):

        if normed.dtype == object:
            res = np.empty(np.size(normed), dtype=object)
            for i in range(np.size(normed,0)):
                x = self.denormalize(normed[i])
                res[i] = x
            return res
        else:
            if normed.ndim == 1:
                normed = np.reshape(normed, (1,-1))

            res = np.zeros(shape=(np.size(normed,0), len(self.variance)))
            norm_i = 0
            for i in range(np.size(self.variance)):
                if self.variance[i] < self.threshold:
                    res[:,i] = np.interp(np.linspace(0, 1, len(res[:,i])), np.linspace(0, 1, len(self.statics[:,i])), self.statics[:,i])
                else:
                    res[:,i] = normed[:,norm_i]
                    norm_i += 1

            return self.normalizer.norm2range(res)

    # rollout
    def rolloutArData(self, data):
        if data.dtype == object:
            # this is arData.arNorm format
            result = np.empty(np.size(data), dtype=object)
            for i in range(len(data)):
                result[i] = self.rolloutArData(data[i])
            return result

        elif data.ndim == 2 and data.shape[1] > self.ar_dim:
            # this is arData.ar format, one sample is given per row
            result = np.empty(data.shape[0], dtype=object)
            for i in range(data.shape[0]):
                reshaped = data[i,:].reshape((self.dmp_params['num_bfs']+2,-1))
                result[i] = self.rolloutArData(reshaped)
            return result

        else:
            # from here we only have to handle (dmp_dim+2 X num_ar_params) arrays

            # To undo this:
            # arDmp_reshaped = np.empty(np.size(arDmp), dtype=object)
            # for i in range(len(arDmp)):
            #     arDmp_reshaped[i] = arDmp[i].reshape((1,-1))

            # normalization method 1
            data = self.dmpNormalizer.norm2range(data.reshape((1,-1))).reshape(data.shape)
            # normalization method 2
            # data = self.dmpNormalizer.norm2range(data)

            rolledOut = self.playDMP(data[1:1+self.dmp_params['num_bfs']], data[0,:], data[-1,:])
            denormRolledOut = self.denormalize(rolledOut)
            return denormRolledOut

    def generateArVariations(self, numSamplesPerSequence, arNoise = {}):

        if arNoise:
            if len(arNoise) == 1:
                self.arNoise = np.tile(arNoise, self.arNorm[0].shape)
            elif np.ndim(arNoise) == 2 and arNoise.shape == self.arNorm[0].shape:
                self.arNoise = arNoise
            elif np.size(arNoise) == self.ar_dim:
                self.arNoise = np.tile(arNoise, (1,int(self.arNorm[0].shape[1]/self.ar_dim)))
            else:
                print("Given arNoise format not understood. Expected: " + str(self.arNoise.shape) + " or [" + str(self.arNoise.shape[1]) + "] or [1]. Ignore!")
            print("Use arNoise: " + str(self.arNoise))

        self.numSamplesPerSequence = numSamplesPerSequence

        # size of the vector-reshaped trajectory data
        numParams = np.size(self.arNorm[0])

        self.ar = np.empty(shape = (np.size(self.arNorm)*self.numSamplesPerSequence, numParams))

        for i in range(np.size(self.arNorm)):
            startIdx = i * self.numSamplesPerSequence
            endIdx = startIdx + numSamplesPerSequence

            copied = np.repeat(np.reshape(self.arNorm[i], (1,-1)), self.numSamplesPerSequence, axis=0)
            randNoise = np.repeat(np.reshape(self.arNoise, (1,-1)), self.numSamplesPerSequence, axis=0) * np.random.randn(self.numSamplesPerSequence, numParams)

            self.ar[startIdx:endIdx,:] = copied + randNoise

