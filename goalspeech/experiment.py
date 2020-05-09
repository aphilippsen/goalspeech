import copy
import pickle
import os

from goalspeech.babble.forward import ForwardModel
from goalspeech.generate.vtl import VTLSpeechProduction
from goalspeech.babble.weighting import LoudnessWeighting

def load_experiment(location, name, sp = None, ambSp = None, arData = None):
    expData = ExperimentData()
    expData.load(location, name, sp=sp, ambSp=ambSp, arData=arData)
    return expData

def save_experiment(location, name, config, gb, ambSp=None, arData=None, errorData=None, compData=None, weights=None):
    expData = ExperimentData(config, gb, ambSp, arData, errorData, compData, weights)
    expData.save(location, name)
    
    print(expData.gb.fwdmodel)
    return expData

class ExperimentData:
    def __init__(self, config=None, gb=None, ambSp=None, arData=None, errorData=None, compData=None, weights=None):
        self.config = config
        self.gb = copy.deepcopy(gb) # changes for pickling only local
        self.ambSp = ambSp # changes for pickling only local
        self.arData = arData
        self.errorData = errorData
        self.compData = compData
        self.weights = weights

    def save(self, location, name):
        # remove lambda functions, they cannot be pickled
        self.gb.fwdmodel = None
        for i in range(len(self.gb.wschemes)):
            if type(self.gb.wschemes[i]).__name__ == 'LoudnessWeighting':
                self.gb.wschemes[i].fct = None

        # remove octave binding temporarily
        try:
            self.ambSp.octave_binding = None
        except:
            pass

        if not name.endswith(".pickle"):
            name = name + ".pickle"

        f = open(os.path.join(location, name), 'wb')
        pickle.dump(self, f)

        # add again temporarily removed object
        try:
            if self.ambSp.featureMode == "gbfb":
                octave_scripts_dir = self.config['locations']['octavescripts']
                from oct2py import Oct2Py
                while True:
                    try:
                        self.ambSp.octave_binding = Oct2Py()
                        break
                    except:
                        continue
                self.ambSp.octave_binding.addpath(octave_scripts_dir)
        except:
            pass
            


    def load(self, location, name, sp = None, ambSp = None, arData = None):
        if not name.endswith('.pickle'):
            name = name + '.pickle'
        f = open(os.path.join(location, name), 'rb')
        expDataObject = pickle.load(f)

        self.config = expDataObject.config
        self.gb = copy.deepcopy(expDataObject.gb)
        self.ambSp = copy.deepcopy(expDataObject.ambSp)
        self.arData = copy.deepcopy(expDataObject.arData)
        self.errorData = copy.deepcopy(expDataObject.errorData)
        self.compData = copy.deepcopy(expDataObject.compData)
        try: # TODO backward compabilibity
            self.weights = copy.deepcopy(expDataObject.weights)
        except:
            print("Weight history could not be loaded!")
            pass

        if sp:
            self.sp = sp
        else:
            lib = self.config['vocalTract']['lib']
            speaker = self.config['vocalTract']['speaker']
            self.sp = VTLSpeechProduction(lib, speaker)

        if self.sp and self.ambSp and self.arData:
            fwdmodel = ForwardModel(self.sp, self.ambSp, self.arData)
            self.gb.fwdmodel = lambda x: fwdmodel.execute(x)
            for i in range(len(self.gb.wschemes)):
                if type(self.gb.wschemes[i]).__name__ == 'LoudnessWeighting':
                    self.gb.wschemes[i] = LoudnessWeighting(self.gb, self.gb.wschemes[i].ref)

        if ambSp:
            self.ambSp = ambSp
        
        if self.ambSp.featureMode == "gbfb":
            octave_scripts_dir = self.config['locations']['octavescripts']
            from oct2py import Oct2Py
            self.ambSp.octave_binding = Oct2Py()
            self.ambSp.octave_binding.addpath(octave_scripts_dir)

        # create valid tmp directory (in case evaluation is done e.g. on another PC)
        import datetime
        import random
        now = datetime.datetime.now()
        self.ambSp.tmpDir = os.environ['XDG_RUNTIME_DIR'] + "/eval-babbling_" + str(now.year) + "-" + str(now.month) + "-" + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute) + "_" + str(now.microsecond) + "_" + str(random.randint(1, 10000)) + "/"

