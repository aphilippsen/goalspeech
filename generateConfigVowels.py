import configparser

vowelConfig = "config/vowelLearning-3d.cfg"

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
                        'featureMode': 'formants',
                        'useModelSpace': '0'}
config['goalSpace'] = {'targetDim': '3',
                        # how to map into goal space (pca, lda, f: take features directly (e.g. formants))
                        'embMethod': 'pca',
                        # in case of LDA: specify to how many dimensions a PCA preprojection should be performed
                        #'ldaPreprojectionDim': '500',
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
                        'wschemes': 'tar sal'}
config['wsm'] = {'tskThreshold': '0.1',
                 'wThrNodeCreation': '0.5'}

config['experiment'] = {'runs': '10',
                        'maxIterations': '500', # per run
                        'stopThreshold': '0.1'}

config['babbling'] = {'rollouts': '10', # per iteration
                      'tskNoiseAmplitude': '0.05', # task space noise
                      'actNoiseAmplitude': '0', # action space noise; single noise value or (1 x actDim) for dimension-dependent noise
                      'actNoiseAuto': 'True', # set to True if actNoiseAmplitude==0: determine action space noise automatically
                      'actNoiseFactor': '0.55', # maximum amount of noise for automatic noise determination
                      'active': False} # active target selection

config['wschemeParams'] = {'distanceThreshold': '1.5'} # maximum goal space distance for TargetWeighting

config['locations'] = {'dataDir': 'data/ambientSpeech',
                    'configDir': 'config',
                    'resultsDir': 'data/results'}

with open(vowelConfig, 'w') as configfile:
    config.write(configfile)
