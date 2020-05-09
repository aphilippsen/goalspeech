import numpy as np

class ForwardModel:

    def __init__(self, speechProduction, ambientSpeech, arData = {}):
        self.sp = speechProduction
        self.ambSp = ambientSpeech
        self.arData = arData

    def execute(self, action):
        """
            Action is a 2-d array of numSamples x numParams
        """

        if action.dtype == object:
            action = np.concatenate(action,axis=0)

        if action.ndim == 1:
            action = np.reshape(action, (1,-1))

        # roll out trajectory => object array
        print("Roll-out trajectories")
        rolledOutAction = self.arData.rolloutArData(action)
        # produce sounds
        print("Produce sound")
        signalNorm = self.sp.produceSound(rolledOutAction)
        # map into goal space
        print("Map into goal space")
        gsPos = self.ambSp.soundToGoalSpace(signalNorm)

        return (gsPos, action, signalNorm)
        
