import datetime
import numpy as np
import os
import random
import scipy
import torch

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# learning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture

from goalspeech.features.features import calcAcousticFeatures
from goalspeech.utils.normalize import Normalizer
from goalspeech.ml.learner import DRFakeLearner, FakeLearner
from goalspeech.ml.esn import ESN

def loudness(sound):
    return np.median(abs(sound))


class AmbientSpeech:

    def __init__(self, config, speech_sound_type='vowel'):
        self.speech_sound_type = speech_sound_type

        # extract important parameters from config
        self.speakerType = config['vocalTract']['speakerType']
        self.audioSamplingRate = int(config['vocalTract']['audioSamplingRate'])

        self.sequences = config['ambientSpeech']['sequences'].split()
        self.featureMode = config['ambientSpeech']['featureMode']

        try:
            octave_scripts_dir = config['locations']['octavescripts']
            from oct2py import Oct2Py
            self.octave_binding = Oct2Py()
            self.octave_binding.addpath(octave_scripts_dir)
        except:
            print("Warning: octave binding could not be set up!")
            self.octave_binding = None

        try:
            self.useModelSpace = int(config['ambientSpeech']['usemodelspace'])
        except:
            self.useModelSpace = 0
        try:
            self.esn_seed = int(config['dynamics']['esnseed'])
        except:
            self.esn_seed = 1
        try: # dimensionality of ESN model space
            self.esn_dim = int(config['dynamics']['esndim'])
        except:
            self.esn_dim= 10
        try:
            self.esn_reg = int(config['dynamics']['esnreg'])
        except:
            self.esn_reg = 1

        self.minmax = None

        self.dataPath = os.path.join(os.path.abspath(config['locations']['dataDir']), config['vocalTract']['lib'])

        self.targetDim = int(config['goalSpace']['targetDim'])
        self.embMethod = config['goalSpace']['embMethod']
        if self.embMethod == 'lda':
            self.lda_preprojection_dim = int(config['goalSpace']['ldaPreprojectionDim'])
        self.learnerMethod = config['goalSpace']['learnerMethod']
        self.targetlocmethod = config['goalSpace']['targetlocmethod']

        # create tmp directory to save intermediate results in feature calculation
        now = datetime.datetime.now()
        self.tmpDir = os.environ['XDG_RUNTIME_DIR'] + "/babbling_" + str(now.year) + "-" + str(now.month) + "-" + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute) + "_" + str(now.microsecond) + "_" + str(random.randint(1, 10000)) + "/"
        os.makedirs(self.tmpDir, exist_ok=True)

        # if no model space is used, this variable will be used to make sure that there is the same number of frames in every sound feature vector
        self.min_frame_num = 0

        # used as name for the feature file
        try:
            dmpstr = "-DMBPF-" + str(config["dynamics"]["dmpbfs"])
        except Exception:
            dmpstr = ""
        self.num_samples_per_sequence = int(config["ambientSpeech"]["numsamplespersequence"])
        self.featureString = "features-" + "".join(self.sequences) + "-" + str(self.num_samples_per_sequence) + dmpstr + '-' + config["ambientSpeech"]["arNoise"].replace(" ", "-") + "-" + self.featureMode

    def setAmbientSpeech(self, soundsNorm, classifications, minimum_energy = None):

        self.sounds = soundsNorm
        self.classifications = classifications

        # load/calculate raw features
        os.makedirs(self.dataPath, exist_ok=True)
        featureFile = os.path.join(self.dataPath, self.featureString + ".mat")
        if os.path.exists(featureFile):
            print("Load features from file: " + featureFile)

            contents = {}
            scipy.io.loadmat(featureFile, contents)
            # contents.keys() lists all variables
            self.rawFeatures = contents['rawFeatures'][0]
            self.signalEnergy = contents['signalEnergy'][0]

            numSamples = np.size(soundsNorm, 0)

        else:
            numSamples = np.size(soundsNorm, 0)

            self.rawFeatures = self.calcFeatures(soundsNorm)
            print("raw features have shape of " + str(self.rawFeatures[0].shape))

            self.signalEnergy = np.empty(numSamples)
            for i in range(numSamples):
                self.signalEnergy[i] = loudness(soundsNorm[i][0])

            
            # save features to file
            contents = {'rawFeatures': self.rawFeatures, 'signalEnergy': self.signalEnergy}
            scipy.io.savemat(featureFile, contents)

        # filter out invalid speech sounds
        self.invalids = np.zeros(numSamples)
        """
        hist, bins = np.histogram(self.signalEnergy)
        print("Histogram of signal energy:")
        print(hist)
        print(bins)
        minimum_energy = bins[:-1][np.cumsum(hist)>100][0]
        """
        if minimum_energy is None:
            # try to determine it automatically
            #minimum_energy = (np.median(self.signalEnergy[0:self.num_samples_per_sequence])-2*np.std(self.signalEnergy[0:self.num_samples_per_sequence]))
            minimum_energy = np.median(self.signalEnergy)-2*np.std(self.signalEnergy)
        
        self.invalids[self.signalEnergy<minimum_energy] = 1
        #self.invalids[self.signalEnergy<(np.mean(self.signalEnergy[0:self.num_samples_per_sequence])-2*np.std(self.signalEnergy[0:self.num_samples_per_sequence]))] = 1

        

        # calculate feature normalization
        self.normFeatures = self.normalizeFeatures(self.rawFeatures)

        # get rid of time dependency
        self.normFeatures = self.applyModelSpaceRepresentation(self.normFeatures)

    def calcFeatures(self, sounds):
        if sounds.dtype == object:
            # a list of sounds
            features = np.empty(np.size(sounds,0), dtype=object)
            features[0], minmax = calcAcousticFeatures(sounds[0], self.audioSamplingRate, self.featureMode, self.speakerType, self.tmpDir, speech_sound_type = self.speech_sound_type, octave_binding = self.octave_binding)
            for i in range(1,np.size(sounds, 0)):
                features[i], _ = calcAcousticFeatures(sounds[i], self.audioSamplingRate, self.featureMode, self.speakerType, self.tmpDir, speech_sound_type = self.speech_sound_type, octave_binding = self.octave_binding)
        else:
            if sounds.ndim == 1:
                sounds = np.reshape(sounds, (1,-1))
            # only a single sound
            (features, minmax) = calcAcousticFeatures(sounds, self.audioSamplingRate, self.featureMode, self.speakerType, self.tmpDir, speech_sound_type = self.speech_sound_type, octave_binding = self.octave_binding)
        if not minmax is None:
            self.minmax = minmax # save specific normalization boundaries according to feature setting

        return features

    # acoustic feature normalization
    # self.rawFeatures <=> self.normFeatures
    def normalizeFeatures(self, features):
        self.normalizer = Normalizer()
        return self.normalizer.normalizeData(features, self.minmax, margin=0.2)

    def normalize(self, x):
        if x.dtype == object:
            res = np.empty(np.size(x), dtype=object)
            for i in range(np.size(x,0)):
                res[i] = self.normalizer.range2norm(x[i])
            return res
        else:
            return self.normalizer.range2norm(x)

    def denormalize(self, x):
        if x.dtype == object:
            res = np.empty(np.size(x), dtype=object)
            for i in range(np.size(x,0)):
                res[i] = self.normalizer.norm2range(x[i])
            return res
        else:
            return self.normalizer.norm2range(x)

    # process features to get rid of time dependency (each sample a single vector)
    def applyModelSpaceRepresentation(self, normFeatures):

        # if normFeatures is not packed
        if normFeatures.ndim == 2:
            normFeatures = np.array([normFeatures])

        if normFeatures[0].ndim == 1:
            feature_dim = normFeatures[0].shape
        else:
            feature_dim = normFeatures[0].shape[1]

        if self.useModelSpace:

            modelspace_features = features = np.empty(np.size(normFeatures,0), dtype=object)

            self.esn = ESN(feature_dim, self.esn_dim, feature_dim, weight_seed=self.esn_seed)

            for i in range(normFeatures.shape[0]):

                # prepare input and output for one-step-ahead-prediction
                X = torch.Tensor(normFeatures[i][0:-1,:])
                Y = torch.Tensor(normFeatures[i][1:,:])
                # feed into ESN
                self.esn(X, Y)
                # compute readout weights
                self.esn.calculate_readout()

                modelspace_features[i] = self.esn.readout.weight.view((1,-1)).detach().numpy()

            normFeatures = np.asarray(modelspace_features)

        else:

            if normFeatures.dtype == object:
                # convert normFeature entrys to 2-d arrays of size 1 x (timeSteps*params)
                # cut if more than the minimum number of frames
                if self.min_frame_num == 0:
                    self.min_frame_num = np.min([f.shape[0] for f in normFeatures])
                for i in range(np.size(normFeatures,0)):
                    normFeatures[i] = np.reshape(normFeatures[i][0:self.min_frame_num,:], (1,-1)) # -1: auto determine size
            else:
                # only a matrix
                if normFeatures.ndim == 3 and normFeatures.shape[0] == 1:
                    normFeatures = normFeatures[0,:,:]
                if normFeatures.shape[0] >= self.min_frame_num:
                    # convert to vector, dropping frames potentially
                    normFeatures = np.reshape(normFeatures[0:self.min_frame_num,:], (1,-1))
                else:
                    # add new frames before converting to vector
                    normFeatures = np.concatenate((normFeatures, np.tile(normFeatures[-1,:], (self.min_frame_num-normFeatures.shape[0],1))),axis=0)
                    normFeatures = np.reshape(normFeatures, (1,-1))

        return normFeatures

    # create goal space from normFeatures using the defined dimred method
    def generateGoalSpace(self, learnerSpecs = {}):
        # mappedData matrix
        self.mappedData = np.zeros(shape=(np.size(self.normFeatures,0), self.targetDim))

        # taking away the invalids, what is the maximum number of samples?
        num_sequences = len(self.sequences)
        non_invalids_max = int(self.num_samples_per_sequence - np.max([np.sum(self.invalids[x*self.num_samples_per_sequence:x*self.num_samples_per_sequence+self.num_samples_per_sequence]) for x in np.arange(num_sequences)]))
        print("Maximum number of valid samples per sequence: " + str(non_invalids_max))
        if non_invalids_max == 0:
            raise ValueError('Not enough valid samples per sequence are available, they were likely filtered out as "invalid". Try to inspect AmbientSpeech.signalEnergy and adjust the minimum_energy parameter in the setAmbientSpeech function.')

        # clean up ambient speech by reducing it, leaving out invalids
        self.normFeatures_for_GS = np.empty((int(num_sequences * non_invalids_max),), dtype=object)
        self.classifications_for_GS = np.empty((int(num_sequences * non_invalids_max),), dtype=object)
        for i in range(num_sequences):
            from_idx = i*self.num_samples_per_sequence
            to_idx = i*self.num_samples_per_sequence+self.num_samples_per_sequence
            list_of_valids = [i for i, x in enumerate(self.invalids[from_idx:to_idx]) if not x]
            random_set = random.sample(list(list_of_valids), non_invalids_max)

            real_idcs = np.asarray(random_set)+i*self.num_samples_per_sequence
            self.normFeatures_for_GS[i*non_invalids_max:i*non_invalids_max+non_invalids_max] = self.normFeatures[real_idcs]
            self.classifications_for_GS[i*non_invalids_max:i*non_invalids_max+non_invalids_max] = self.classifications[real_idcs]
        
        # TODO check this: why should i use rawFeatures here? I delete them for now to save space
        if self.embMethod == 'f':
            # use features directly
            #indices = 1:self.targetDim
            if self.featureMode == 'formants':
                for i in range(np.size(self.normFeatures_for_GS,0)):
                    self.mappedData[i,:] = self.normFeatures_for_GS[i][0,0:self.targetDim]
                self.mappingLearner = FakeLearner(np.size(self.normFeatures_for_GS[0]), self.targetDim)
            elif self.featureMode == 'formants_full':
                for i in range(np.size(self.normFeatures_for_GS,0)):
                    # take the mean of all formants
                    self.mappedData[i,:] = self.normalize(np.mean(self.rawFeatures[i],0))[0:self.targetDim]
                self.mappingLearner = FakeLearner(np.size(self.normFeatures_for_GS[0]), self.targetDim, [], lambda x: np.mean(x,0))
            else:
                raise Exception("Error in generateGoalSpace(): featureMode " + self.featureMode + " does not support goal space generation method " + self.embMethod)

        elif self.embMethod == 'pca':
            # Use PCA for dimension reduction

            # all features in a matrix
            allFeatures = np.concatenate(self.normFeatures_for_GS)

            # with whiten = True (http://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
            # seems to make difference only in scaling. As I scale to [-1,1] range anyway, no difference!

            scaler = StandardScaler()
            scaler.fit(allFeatures)
            scaledFeatures = scaler.transform(allFeatures)

            pca = PCA(n_components = self.targetDim)
            pca.fit(scaledFeatures)
            self.mappedData = pca.transform(scaledFeatures)

            normalizerMapped = Normalizer()
            self.mappedData = normalizerMapped.normalizeData(self.mappedData)
            # save learner for project other data as well later on
            self.mappingLearner = DRFakeLearner(np.size(self.normFeatures_for_GS[0]), self.targetDim, pca, normalizerMapped, prescaling = scaler)

        elif self.embMethod == 'lda':
            # Use LDA for dimension reduction
            # For efficiency, a preprojection via PCA is performed
            allLabels = np.array([self.sequences.index(cl) for cl in self.classifications_for_GS])

            # all features in a matrix
            allFeatures = np.concatenate(self.normFeatures_for_GS)
            origDim = np.size(allFeatures,1)

            scaler = StandardScaler()
            scaler.fit(allFeatures)
            scaledFeatures = scaler.transform(allFeatures)

            # keep 10 eigenvectors or more depending on original dimensionality
            if self.lda_preprojection_dim == 0:
                self.prePcaDim = origDim
            else:
                self.prePcaDim = min(self.lda_preprojection_dim, origDim)
                print("PCA preprojection from " + str(origDim) + " to " + str(self.prePcaDim) + " dimensions.")

            # pre PCA for more efficient calculation
            if self.prePcaDim < origDim:
                pre_pca = PCA(n_components = self.prePcaDim)
                pre_pca.fit(scaledFeatures)
                allFeaturesPCA = pre_pca.transform(scaledFeatures)
            else:
                pre_pca = None
                allFeaturesPCA = scaledFeatures

            lda = LDA(n_components = 2)
            lda.fit(allFeaturesPCA, allLabels)
            self.mappedData = lda.transform(allFeaturesPCA)

            normalizerMapped = Normalizer()
            self.mappedData = normalizerMapped.normalizeData(self.mappedData)
            # TODO save learner
            self.mappingLearner = DRFakeLearner(np.size(self.normFeatures_for_GS[0]), self.targetDim, lda, normalizerMapped, prescaling = scaler, premapping_dr = pre_pca)

    # calculate normalized features from sound (2-d array or 1-d array with dtype=object)
    def soundToFeatures(self, sound):
        rawFeatures = self.calcFeatures(sound)
        normFeatures = self.normalize(rawFeatures)
        normFeatures = self.applyModelSpaceRepresentation(normFeatures)
        return normFeatures

    # project sound into goal space (2-d array or 1-d array with dtype=object)
    # returns 2-d array numSamples x targetDim
    def soundToGoalSpace(self, sound):
        normF = self.soundToFeatures(sound)
        if self.mappingLearner:
            if normF.dtype == object:
                gsPos = np.zeros(shape=(np.size(normF,0), self.targetDim))
                for i in range(np.size(normF,0)):
                    gsPos[i] = self.mappingLearner.apply(normF[i])
            else:
                gsPos = self.mappingLearner.apply(normF)
        else:
            raise Exception("Error: mapping to goal space undefined, call generateGoalSpace() first!")

        return gsPos

    def locateTargets(self):
        if self.targetlocmethod == 'fake':
            # automatically set Gaussians on the correct places
            numSamples = int(self.mappedData.shape[0]/len(self.sequences))
            desired_means = [np.mean(self.mappedData[i*numSamples:i*numSamples+numSamples,:],axis=0) for i in range(len(self.sequences))]
            desired_covs = [np.cov(np.transpose(self.mappedData[i*numSamples:i*numSamples+numSamples,:])) for i in range(len(self.sequences))]
            if np.size(self.mappedData) > 0:
                # data distribution of Gaussians
                self.targetDist = GaussianMixture(n_components = len(self.sequences), covariance_type = 'full', means_init=desired_means) # weights_init=
                try:
                    self.targetDist.fit(self.mappedData)
                except:
                    print("Warning: Could not automatically detect clusters via Gaussian Mixture Model. Probably more data is required!")
                # set desired mean and covariances
                self.targetDist.means_ = np.asarray(desired_means)
                self.targetDist.covariances_ = np.asarray(desired_covs)
            else:
                raise Exception("Error: mappedData not defined, call generateGoalSpaceFirst()!")

        else:
            # determine Gaussians automatically from data
            if np.size(self.mappedData) > 0:
                # data distribution of Gaussians
                self.targetDist = GaussianMixture(n_components = len(self.sequences), covariance_type = 'full') # weights_init=
                self.targetDist.fit(self.mappedData)
            else:
                raise Exception("Error: mappedData not defined, call generateGoalSpaceFirst()!")

    # sort self.sequence such that it fits with the GMs
    def getNewSequenceOrder(self):
        '''
            Returns a pair (newOrder, classes) where newOrder contains the indices of the original sequences in the new array, classes contains the reordered sequences list.
        '''
        newOrder = []
        for i in range(np.size(self.mappedData,0)):
            # get GM index
            newIdx = np.argmin(scipy.linalg.norm(self.mappedData[i] - self.targetDist.means_, axis=1))

            # TODO somewhat ugly solution, but at least doesn't make
            # assumption that every class has the same amount of samples
            # BUT mappedData has to be sorted by class
            try:
                # already added
                newOrder.index(newIdx)
            except ValueError:
                # add
                newOrder = newOrder + [newIdx]

        return (newOrder, [self.sequences[newOrder.index(s)] for s in range(len(self.sequences))])

    def getTargetPositions(self):
        '''
            Returns positions where GM components were located. Adjusts the sequence of the target positions means and covariances to the original sequence order.
        '''
        (newOrder, classes) = self.getNewSequenceOrder()
        return (self.targetDist.means_[newOrder],self.targetDist.covariances_[newOrder])

    def plot(self, save_directory = None):
        allLabels = np.array([self.sequences.index(cl) for cl in self.classifications_for_GS])

        fig = plt.figure('Proactive results', figsize=(10, 10))
        plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
        ax = fig.add_subplot(111)
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'darkorange', 'navy', 'yellow', 'black']
        lw = 2

        # plot mappedData
        for (i, color, target_name) in zip(range(len(self.sequences)), colors, self.sequences):
            ax.scatter(self.mappedData[allLabels == i, 0], self.mappedData[allLabels == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title("Features: " + self.featureMode + "  Dim.red.method: " + self.embMethod)

        # plot targetDist
        if self.targetDist:
            means = self.targetDist.means_
            covs = self.targetDist.covariances_

            for (mean, covar) in zip(means, covs):
                v,w = scipy.linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / scipy.linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, edgecolor = 'black', facecolor='none')
                #ell.set_clip_box(plt.bbox)
                ell.set_alpha(1)
                plt.gca().add_patch(ell)

        if save_directory:
            plt.savefig(os.path.join(save_directory, "gs.pdf"))

        return (fig, ax) #plt.show(block = False)

    # prepare for pickling and unpickling
    def __getstate__(self):
        d = dict(self.__dict__)
        if self.featureMode == 'formants_full' and self.embMethod == 'f':
            d['mappingLearner'].preprocessingFct = None

        return d

    def __setstate__(self, d):
        if d['featureMode'] == 'formants_full' and d['embMethod'] == 'f':
            d['mappingLearner'].preprocessingFct = lambda x: np.mean(x,0)
        self.__dict__.update(d)
        
        

