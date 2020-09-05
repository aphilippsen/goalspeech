import numpy as np

from .weighting import AlwaysOnWeighting, LoudnessWeighting, TargetWeighting, SyllableStructureWeighting
from .workspace import WorkspaceModel
from goalspeech.ml.rbf import OnlineWeightedRBF
from goalspeech.evaluate import competence

import scipy.io.wavfile
import os

class GoalBabbling:
    """
        Goal Babbling object, has forward and inverse model
        self.tskHome, self.actHome, self.soundHome
    """

    def __init__(self, fwdmodel, invmodelSpecs = {'class': 'OnlineWeightedRBF', 'lrate': 0.9, 'radius': 0.1, 'softmax': 1, 'weightThreshold': 0.01}):
        self.fwdmodel = fwdmodel
        self.invmodelSpecs = invmodelSpecs

        self.iteration = -1

    def init(self, actHome):
        self.actHome = actHome
        (self.tskHome, action, self.soundHome) = self.fwdmodel(self.actHome)
        print("Home action: " + str(self.actHome))
        print("Home goal space position: " + str(self.tskHome))
        assert(not np.isnan(self.tskHome).any())

        self.actDim = np.size(self.actHome)
        self.tskDim = np.size(self.tskHome)

        # find out which learner to use (and delete class element from dict)
        try:
            learnerClass = self.invmodelSpecs.pop('class')
        except:
            print("Element class from inverse model specs already removed?")

        # initialize inverse model
        self.invmodel = eval(learnerClass + '(self.tskDim, self.actDim, self.invmodelSpecs)')
        self.invmodel.init(self.tskHome, self.actHome)

    def explorationStep(self):
        pass
    def calculateWeights(self):
        pass
    def adaptationStep(self):
        pass

    def babble(self):
        self.iteration += 1
        print("Explore...")
        self.explorationStep()
        print("Calc weights...")
        self.calculateWeights()
        for i in range(len(self.wschemes)):
            print(str(self.wschemes[i]) + ":\n" + str(self.weights[-1][i]))
        print("Adapt learner...")
        self.adaptationStep()


class SkillGoalBabbling(GoalBabbling):

    def __init__(self, fwdmodel, invmodelSpecs = {}):
        if invmodelSpecs:
            GoalBabbling.__init__(self, fwdmodel, invmodelSpecs)
        else:
            GoalBabbling.__init__(self, fwdmodel)

        # default parameters

        # exploration:
        self.numRollouts = 10
        self.tskNoiseAmplitude = 0.05 # task space noise
        self.actNoiseAmplitude = 0.3 # action space noise; single noise value or (1 x actDim) for dimension-dependent noise
        self.actNoiseAuto = False # set to True if actNoiseAmplitude==0: determine action space noise automatically
        self.actNoiseFactor = 0.55 # maximum amount of noise for automatic noise determination

        # WSM:
        self.wsm = None
        self.tskThreshold = 0.1 # workspace model radius
        self.wThrNodeCreation = 0.5 # minimum weight for creation of new nodes in WSM

        self.drawFromTargetDist = True # whether drawing new targets from targetDist or from WSM cluster centers
        #self.cSIdx = 1 # if targets are drawn from the WSM, this contains currently selected target

        # babbling
        self.tskCurrent = [] # current task seed (selected at the end of babble(), so this contains usually the next planned task!)
        self.actCurrent = [] # result of invmodel(tskCurrent)

        self.currentClusterIdx = 0 # index of prototype that was currently selected (only valid if drawFromTargetDistribution)
        # legacy from matlab code where ground truth was not available
        self.currentClusterIdxEstimation = 0

        self.currentWeights = np.zeros(self.numRollouts) # weights for current tRolloutsDesired, dim: numRollouts
        self.weights = [] # current weights, separately stored for each wscheme, dim: numWSchemes x numRollouts
        self.allWavs = None
        self.currentWav = [] # holds sound of that sample of the batch that scored the highest weight

        self.active = False
        # CMA: cma, C

    def configure(self, config):
        self.numRollouts = int(config['babbling']['rollouts'])
        self.tskNoiseAmplitude = float(config['babbling']['tskNoiseAmplitude'])
        self.actNoiseAmplitude = float(config['babbling']['actNoiseAmplitude'])
        self.actNoiseAuto = (config['babbling']['actNoiseAuto'] == 'True')
        self.actNoiseFactor = float(config['babbling']['actNoiseFactor'])

        self.tskThreshold = float(config['wsm']['tskThreshold'])
        self.wThrNodeCreation = float(config['wsm']['wThrNodeCreation'])
        self.active = (config['babbling']['active'] == 'True')

    def init(self, actHome, targetDistribution, wschemes = ['tar', 'sal'], wschemeParams = {}, ambSp = None):
        """
            ambSp only required as reference for 'syl' weighting scheme.
        """
        GoalBabbling.init(self, actHome)

        # TODO create wschemes

        # set initial seed for exploration
        self.tskCurrent = self.tskHome
        self.actCurrent = self.actHome

        # set noise level
        if self.actNoiseAmplitude == 0:
            # not fixed, but automatic adaptation
            self.actNoiseAuto = True
            self.actNoiseAmplitude = self.actNoiseFactor # as default level

        # create/init workspace model
        self.wsm = WorkspaceModel(self.tskHome, targetDistribution, self.tskThreshold, self.wThrNodeCreation)

        # init wschemes
        self.wschemes = np.empty(len(wschemes), dtype=object)
        for (scheme, i) in zip(wschemes, range(len(wschemes))):
            # create scheme
            if scheme == 'one':
                wsch = AlwaysOnWeighting(self)
                # TODO add all weighting schemes
            elif scheme == 'sal':
                try:
                    wsch = LoudnessWeighting(self, float(wschemeParams['refSignalEnergy']))
                except:
                    error("LoadnessWeighting requires a reference value of signal energy!")
            elif scheme == 'tar':
                try:
                    distanceThreshold = float(wschemeParams['distanceThreshold'])
                    wsch = TargetWeighting(self, distanceThreshold)
                except:
                    wsch = TargetWeighting(self)
            elif scheme == 'syl':
                assert(ambSp is not None)
                min_limit = float(wschemeParams['syllableMinDist'])
                max_limit = float(wschemeParams['syllableMaxDist'])
                wsch = SyllableStructureWeighting(self, ambSp, min_limit, max_limit)
            else:
                raise ValueError("SkillGoalBabbling: no such weighting scheme defined: " + scheme)
            # register scheme
            self.wschemes[i] = wsch

        # init babbling history log (internal evaluation)
        numTargets = targetDistribution.n_components
        self.competenceHistory = np.empty(numTargets, dtype=object)
        for x in range(numTargets):
            self.competenceHistory[x] = []

    def explorationStep(self):
        """
            Explore around tskCurrent, store results in tRolloutsDesired.
        """

        # tsk exploration: create numRollouts noisy samples of
        # tskCurrent and store in tRolloutsDesired
        copiedTskCurrent = np.repeat(np.reshape(self.tskCurrent, (1,-1)), self.numRollouts, axis=0)
        self.tRolloutsDesired = copiedTskCurrent + self.tskNoiseAmplitude * np.random.randn(self.numRollouts, self.tskDim)

        # legacy: estimate which cluster is currently explored
        self.currentClusterIdxEstimation = np.argmin(self.wsm.metric.distance(self.tskCurrent, self.wsm.targetDist.means_))
        print("Target in cluster " + str(self.currentClusterIdxEstimation))

        # get inv estimates of tRolloutsDesired
        print("Get inverse estimates")
        self.aRolloutsOrig = self.invmodel.apply(self.tRolloutsDesired)

        # add exploratory noise in action space
        print("Add exploratory noise")
        self.aRolloutsExpl = self.aRolloutsOrig + self.actNoiseAmplitude * np.random.randn(self.numRollouts, self.actDim)

        # observe actual outcomes
        print("Apply forward model")
        (self.tRolloutsExpl, self.aRolloutsExpl, self.allWavs) = self.fwdmodel(self.aRolloutsExpl)

    def calculateWeights(self):
        self.weights.append(np.reshape(np.concatenate([scheme.getWeights() for scheme in self.wschemes]), (-1, self.numRollouts)))
        # replace NaNs by zeros
        #self.weights[-1]

        # product of columns
        self.currentWeights = np.prod(self.weights[-1], 0)
        maxIdx = np.argmax(self.currentWeights)
        self.currentWav = self.allWavs[maxIdx]

    def adaptationStep(self):

        # adapt inverse model
        self.invmodel.train(self.tRolloutsExpl, self.aRolloutsExpl, self.currentWeights)

        # store competence result in history
        currentCompetences = competence(self.tRolloutsDesired, self.tRolloutsExpl)
        currentCompetence = np.nanmean(currentCompetences)
        self.competenceHistory[self.currentClusterIdx].append([self.iteration, currentCompetence, self.actNoiseAmplitude])

        # update task space model
        self.wsm.update(self.iteration, self.tRolloutsExpl, self.currentWeights, currentCompetences)

        # decide target and action noise for next iteration
        # --------------------------------

        (self.tskCurrent, self.currentClusterIdx) = self.wsm.activeTargetSelection(self.drawFromTargetDist, self.active)
        self.actCurrent = self.invmodel.apply(self.tskCurrent)

        if self.actNoiseAuto:
            # automatic noise determination
            self.actNoiseAmplitude = self.actNoiseFactor * self.wsm.getActionNoiseLevel(self.tskCurrent)
            print("Adjusted action noise amplitude to " + str(self.actNoiseAmplitude))

    def get_babbled_sounds(self):
        sound = np.zeros((5000,))
        for s in self.allWavs:
            sound = np.concatenate((sound, s))
            sound = np.concatenate((sound, np.zeros(5000,)))
        sound = np.concatenate((sound, np.zeros(5000,)))
        return sound

    def play_babbled_sounds(self):
        sound = self.get_babbled_sounds()

        import sounddevice as sd
        sd.default.samplerate = 22050
        sd.play(sound, blocking=True)

    def store_babbled_sounds(self, filename):
        sound = self.get_babbled_sounds()
        scipy.io.wavfile.write(filename, 22050, sound.astype('int16'))


