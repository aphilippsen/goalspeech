from goalspeech.utils.normalize import Normalizer

class Learner:

    def __init__(self, inpDim, outDim, spec = {}):
        self.inpDim = inpDim
        self.outDim = outDim

        self.inpNormalization = 1 # (de)activates normalization of input
        self.outNormalization = 1 # (de)activates normalization of output
        inpNormalizer = {}
        outNormalizer = {}

        for k in spec.keys():
            # set member parameters as defined in the dictionary spec.
            # used because classes subclassed from this class can have
            # various other parameters that cannot be specified here
            val = eval("spec['" + k + "']")
            try:
                val = float(val)
            except:
                pass

            exec("self." + k + " = val")


    def normalizeIO(self, X, Y):
        # TODO how to handle normalization?
        pass

    def init(self, X):
        pass

    def train(self, X, Y):
        pass

    def apply(self, X):
        pass


class FakeLearner(Learner):

    def __init__(self, inpDim, outDim, outputIndices = [], preprocessingFct = None):
        Learner.__init__(self, inpDim, outDim, {})

        if outputIndices:
            self.outputIndices = outputIndices
        else:
            self.outputIndices = [o for o in range(outDim)]

        self.preprocessingFct = preprocessingFct

    def apply(self, X):
        if self.preprocessingFct:
            X = self.preprocessingFct(X)

        if self.outputIndices:
            if X.ndim == 1:
                return X[self.outputIndices]
            else:
                return X[:,self.outputIndices]
        else:
            raise ValueError("FakeLearner: outputIndices not defined or empty!")


class DRFakeLearner(Learner):
    '''
    A Learner class implementation that does not actually learn the relation between input and output, but instead executes existing Dimension Reduction mappings obtained from PCA or LDA (classes from sklearn).
    '''

    def __init__(self, inpDim, outDim, dr_object, normalizer, prescaling = None, premapping_dr = None):
        Learner.__init__(self, inpDim, outDim, {})
        self.dr = dr_object
        self.normalizer = normalizer
        self.premapping_dr = premapping_dr
        self.prescaling = prescaling

    def apply(self, X):
        X = self.prescaling.transform(X)

        if self.premapping_dr:
            X = self.premapping_dr.transform(X)

        Y = self.dr.transform(X)
        Y = self.normalizer.range2norm(Y)
        return Y
