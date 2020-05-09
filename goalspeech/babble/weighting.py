import numpy as np
from dtw import dtw

class WeightingScheme:
    def __init__(self, gb):
        self.gb = gb

    def getWeights(self):
        return self.calcWeights()

    def calcWeights(self):
        pass

class AlwaysOnWeighting(WeightingScheme):

    def __init__(self, gb):
        WeightingScheme.__init__(self, gb)

    def calcWeights(self):
        return np.ones(self.gb.numRollouts)

class LoudnessWeighting(WeightingScheme):
    """
        Rates the overall loudness of the sound, given a refence
        loadness and an evaluation function.
    """

    def __init__(self, gb, ref, fct = (lambda x: np.median(np.abs(x)))):
        """
            Reference ref can be e.g. the value that a typically good
            speech signal achieves in the data set.
            The default evaluation function for loudness takes the median
            of the absolute values of the signal and may be overwritten.
        """
        WeightingScheme.__init__(self, gb)
        self.ref = ref
        self.fct = fct

    def calcWeights(self):
        w = np.zeros(self.gb.numRollouts)
        for i in range(self.gb.numRollouts):
            w[i] = self.fct(self.gb.allWavs[i])
            if w[i] < (0.5*self.ref):
                w[i] = 0
            else:
                w[i] = min(w[i] / (1.5*self.ref), 1)
        return w

class TargetWeighting(WeightingScheme):
    """
        Weights the quality of the babbled speech sound according to the
        euclidean distance of the desired and the actually achieved goal
        space positions.
        The weights are scaled within one batch, therefore, it is
        resistant to single outliers. The threshold ensures stable
        behavior in case of several outliers.
    """

    def __init__(self, gb, threshold = 1.5):
        """
            Default for hard threshold for weight 0 as measured as
            euclidean distance in goal space is 1.5.
        """
        WeightingScheme.__init__(self, gb)
        self.threshold = threshold

    def calcWeights(self):
        w = np.zeros(self.gb.numRollouts)
        targetError = np.sqrt(np.sum((self.gb.tRolloutsExpl - self.gb.tRolloutsDesired)**2, axis=1))

        targetError = [targetError[i] if targetError[i] < self.threshold else self.threshold for i in range(len(targetError))]
        w = 1 - targetError / np.repeat(np.max(targetError), w.shape)
        #w = np.exp(-2 * np.asarray(targetError) / np.repeat(np.max(targetError), w.shape))
        return w

class SyllableStructureWeighting(WeightingScheme):

    def __init__(self, gb, ambSp, min_limit = 5e6, max_limit = 10e6):
        WeightingScheme.__init__(self, gb)
        self.eval_window_width = 100
        self.gb = gb
        self.min_limit = min_limit
        self.max_limit = max_limit

        self.compare_sounds = []
        for i in range(0, len(ambSp.sounds), np.int(np.floor(len(ambSp.sounds)/len(ambSp.sequences)))):
            self.compare_sounds.append(ambSp.sounds[i])

        self.compare_structures = []
        for i in range(len(self.compare_sounds)):
            structure_home = np.abs(self.compare_sounds[i][0])
            h = []
            for j in range(0, len(structure_home) - len(structure_home)%self.eval_window_width-self.eval_window_width, self.eval_window_width):
                h.append(np.sum(structure_home[j:j+self.eval_window_width]))
            self.compare_structures.append(np.array(h, dtype='float'))

    def calcWeights(self):
        w = np.zeros(self.gb.numRollouts)

        dists = np.zeros(len(self.gb.allWavs))
        for i in range(len(self.gb.allWavs)):
            structure = np.abs(self.gb.allWavs[i])
            s = []
            for j in range(0, len(structure) - len(structure)%self.eval_window_width-self.eval_window_width, self.eval_window_width):
                s.append(np.sum(structure[j:j+self.eval_window_width]))

            #import sounddevice as sd; sd.play(self.gb.allWavs[i])
            euclidean_norm = lambda x, y: np.abs(x - y)

            try:
                # d, cost_matrix, acc_cost_matrix, path = dtw(np.array(s,dtype='float'), h, dist=euclidean_norm) # w=20
                min_d = self.max_limit * 32767
                for k in [0]:#range(len(self.compare_structures)):
                    new_d, cost_matrix, acc_cost_matrix, path = dtw(np.array(s,dtype='float'), self.compare_structures[k], dist=euclidean_norm) # w=20
                    if new_d < min_d:
                        min_d = new_d

            except:
                min_d = self.max_limit * 32767
            dists[i] = min_d
            #print(d)
            #import pdb; pdb.set_trace()

        print(dists)
        # clip to range
        dists = dists / 32767 # scale to max sound amplitude of 1
        dists[dists>self.max_limit] = self.max_limit
        dists[dists<self.min_limit] = self.min_limit
        dists = dists - self.min_limit
        w = 1 - dists/np.max(dists)
        
        return w

