import configparser
from goalspeech.generate.vtl import VTLSpeechProduction

if not 'config' in locals():
    syllableConfig = "config/syllableLearning.cfg"
    config = configparser.ConfigParser()
    config.read(syllableConfig)

# create object for speech production
lib = config['vocalTract']['lib']
speaker = config['vocalTract']['speaker']

# VTLSpeechProduction may only be instantiated once! Otherwise, it stops
# working properly! (returns ints instead of floats!?)
# Therefore, in ipython use: '%run -i test'
try:
    sp.audioSamplingRate
    print("'sp' already defined.")
except:
    print("No 'sp' object exists!")

if not 'sp' in locals():
    print("create new VTLSpeechProduction object")
    sp = VTLSpeechProduction(lib, speaker)
#(tr, gl) = sp.getExampleArTraj()
#audio = sp.produceSound(tr, gl)

arNoise = [float(x) for x in config['ambientSpeech']['arNoise'].split()]

sequences = config['ambientSpeech']['sequences'].split()
neutral = config['ambientSpeech']['neutral']
numSamplesPerSequence = int(config['ambientSpeech']['numSamplesPerSequence'])

maxIterations = int(config['experiment']['maxIterations'])
runs = int(config['experiment']['runs'])
rollouts = int(config['babbling']['rollouts'])
stopThreshold = float(config['experiment']['stopThreshold'])

tskThreshold = float(config['wsm']['tskThreshold'])
wThrNodeCreation = float(config['wsm']['wThrNodeCreation'])

wschemes = config['invmodel']['wschemes'].split()
wschemeParams = config['wschemeParams']

tskNoiseAmplitude = float(config['babbling']['tskNoiseAmplitude'])
actNoiseAmplitude = float(config['babbling']['actNoiseAmplitude'])
actNoiseAuto = (config['babbling']['actNoiseAuto'] == 'True')
actNoiseFactor = float(config['babbling']['actNoiseFactor'])
active = (config['babbling']['active'] == 'True')

invmodelSpecs = config['invmodel']

dmp_bfs = int(config['dynamics']['dmpbfs'])
dmp_k = float(config['dynamics']['dmpk'])
dmp_sigma = float(config['dynamics']['dmpsigma'])
dmp_tau = float(config['dynamics']['dmptau'])
