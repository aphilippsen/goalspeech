import configparser
import os

config = configparser.ConfigParser()
config['vocalTract'] = {'lib': 'vtl-2.1',
                        'speaker': 'JD2',
                        'speakerType': 'male',
                        'audioSamplingRate': '22050'}
config['ambientSpeech'] = {'sequences': 'a e i o u @',
                        'neutral': '@',
                        'arNoise': '0.1 0.1 0.1 0.1 0.1 0.1 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01',
                        'duration': '500', # in ms
                        'numSamplesPerSequence': '100',
                        'featureMode': 'formants_full',
                        'useModelSpace': '0'}
config['goalSpace'] = {'targetDim': '2',
                        # how to map into goal space (pca, lda, f: take features directly (e.g. formants))
                        'embMethod': 'f',
                        # method to use to train the mapping from normFeatures to goal space
                        # original: use not learn mapping, but use it directly ('original', 'ESN'…)
                        'learnerMethod': 'original',
                        # how to discover GMM clusters from the mapped goal space data ('em', 'fake')
                        # fake: no clustering, directly infer from data
                        'targetLocMethod': 'fake'}
config['invmodel'] = {'class': 'OnlineWeightedRBF',
                        'lrate': '0.9',
                        'radius': '0.15',
                        'softmax': '1',
                        'weightThreshold': '0.1',
                        'wschemes': 'one'}
config['wsm'] = {'tskThreshold': '0.1',
                 'wThrNodeCreation': '0.5'}

config['experiment'] = {'runs': '5',
                        'maxIterations': '500', # per run
                        'stopThreshold': '0.1'}

config['babbling'] = {'rollouts': '10', # per iteration
                      'tskNoiseAmplitude': '0.05', # task space noise
                      'actNoiseAmplitude': '0', # action space noise; single noise value or (1 x actDim) for dimension-dependent noise
                      'actNoiseAuto': 'True', # set to True if actNoiseAmplitude==0: determine action space noise automatically
                      'actNoiseFactor': '0.55', # maximum amount of noise for automatic noise determination
                      'active': False} # active target selection

config['wschemeParams'] = {}

config['locations'] = {'dataDir': 'data/ambientSpeech',
                    'configDir': 'config',
                    'resultsDir': 'data/results'}

def load_config(configName):
    c = configparser.ConfigParser()
    c.read(os.path.join(config['locations']['configDir'], vowelConfig))
    return c

def save_config(config, configName, location = None):
    if not location:
        location = config['locations']['configDir']
    confName = os.path.join(location, configName)
    #vowelConfig = 'config/vowelLearning.cfg'
    with open(confName, 'w') as configfile:
        config.write(configfile)

def generate_config(changes = {}):
    """ Create a default config with changes as defined in the given
        dictionary.
    """
    c = config
    for d in changes:
        try:
            v = str(changes[d])
        except:
            raise TypeError("Values must be strings!")
        ks = d.split('/')
        config[ks[0]][ks[1]] = v

    return c

def print_config(c):
    print("Config:")
    for config_entries in c:
        print("[" + config_entries + "]")
        for e in c[config_entries]:
            print("\t" + e + ": " + c[config_entries][e])
