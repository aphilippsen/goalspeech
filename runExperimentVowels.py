import datetime
import numpy as np
import os
import random
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

from goalspeech.generate.acoustics import AmbientSpeech
from goalspeech.generate.articulation import ArticulatoryShapeData
from goalspeech.babble.forward import ForwardModel
from goalspeech.babble.goalbabbling import SkillGoalBabbling
from goalspeech.evaluate import competence, distance
from goalspeech.experiment import save_experiment

import goalspeech.config

# articulatory data as shapes, realized as 500~ms long vowels

shapes = sp.getArticulatoryShapes(sequences)

# >0 values: use this value as variance when normalizing
arParamsOn = np.zeros(sp.numVocalTractParams + sp.numGlottisParams)
# 1: switch off / ignore
arParamsOff = np.array([0, 0, 0, 0, 0,       # HX, HY, JX, JA, LP
                           0, 0, 0, 0, 0,       # LD, VS, VO, WC, TCX
                           0, 0, 0, 0, 0,       # TCY, TTX, TTY, TBX, TBY
                           1, 1, 0, 0, 0,       # TRX, TRY, TS1, TS2, TS3
                           0, 0, 0, 0,          # TS4, MA1, MA2, MA3
                           1, 1, 1, 1, 1, 1])   # F0, Pressure, lower_rest_displacement upper_rest_displacement ary_area aspiration_strength

dataPath = os.path.join(os.path.abspath(config['locations']['dataDir']), config['vocalTract']['lib'])
os.makedirs(dataPath, exist_ok=True)
arDataFile = os.path.join(dataPath, "arData-" + "".join(sequences) + "-" + str(numSamplesPerSequence) + "-" + config["ambientSpeech"]["arNoise"].replace(" ", "-") + ".mat")
audiosamplingrate = int(config['vocalTract']['audiosamplingrate'])

#soundDict = {}
#scipy.io.loadmat("/home/anja/acarmap/v1/data/ambientSpeechData/vtl-2.1/arData-sounds_aeioun_100_0.1-0.1-0.1-0.1-0.1-0.1-0.01-0.01-0.01-0.01-0.01-0.01-0.01-0.01-0.01-0.01-0.01-0.01.mat", soundDict)
#soundsNorm = soundDict['soundsNorm']
## convert 600x1 Matrix to 600-Array of 1x~13000 Matrices
#soundsNorm = soundsNorm[:,0]
## would create a 2D matrix of sounds, but sounds might be of different length, so rather not!
## soundsNorm = [s[0,:] for s in soundsNorm]

print("Create articulatory trajectories…")
arData = ArticulatoryShapeData(arParamsOn, arParamsOff, sequenceDuration, sp.arSamplingRate)
arData.setArticulatoryData(shapes, sequences.index(neutral))

print("Try to load articulatory data…")
if os.path.exists(arDataFile):
    contents = {}
    scipy.io.loadmat(arDataFile, contents)
    soundsNorm = contents['soundsNorm'][0]
    originalArVariations = contents['originalArVariations']
    arData.ar = arData.normalize(originalArVariations)
    print("Loaded ar and sound data from file: " + arDataFile)
else:
    print("Generate articulatory data…")
    arData.generateArVariations(numSamplesPerSequence, arNoise)
    originalArVariations = arData.denormalize(arData.ar)
    soundsNorm = sp.produceSound(arData.rolloutArData(arData.ar))
    contents = {'originalArVariations': originalArVariations, 'soundsNorm': soundsNorm}
    scipy.io.savemat(arDataFile, contents)

print("Load or generate ambient speech…")
ambSp = AmbientSpeech(config, speech_sound_type = 'vowel')
classifications = np.repeat(sequences, numSamplesPerSequence)

# calculates raw acoustic features, the apropriate normalization params,
# and potentially the model space representation
ambSp.setAmbientSpeech(soundsNorm, classifications)
print("Done.")

# FOR EACH RUN

now = datetime.datetime.now()
expStr = str(now.year) + "-" + str(now.month) + "-" + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute) + "_" + str(now.microsecond).zfill(7)
main_directory = os.path.join(config['locations']['resultsDir'], expStr)
os.makedirs(main_directory)

goalspeech.config.save_config(config, "config.txt", main_directory)

evaluation_interval = int(config['experiment']['evaluationInterval'])
store_experiment_each_it = False
store_best_iteration = True

for r in range(runs):
    print("Run " + str(r))
    save_directory = os.path.join(main_directory, 'run-' + str(r))
    os.makedirs(save_directory)
    
    it_eval = 0 # index for arrays storing current competence/error evaluation values
    current_min_error = np.Infinity

    # generate low-dimensional goal space of acoustic data
    ambSp.generateGoalSpace()
    # cluster goal space data => GMMs, later serving as target distribution
    ambSp.locateTargets()
    ambSp.plot(save_directory = save_directory)

    # define forward model
    fwdmodel = ForwardModel(sp, ambSp, arData)
    fwd = lambda x: fwdmodel.execute(x)

    # create babbling object and initialize
    sgb = SkillGoalBabbling(fwd, dict(invmodelSpecs))
    sgb.configure(config)
    wschemeParams['refSignalEnergy'] = str(np.mean(ambSp.signalEnergy))
    sgb.init(arData.arHome, ambSp.targetDist, wschemes, wschemeParams)

    # initialize arrays for saving external evaluation
    numTargets = ambSp.targetDist.n_components
    evalErrorHistory = np.zeros((maxIterations, numTargets))
    evalCompetenceHistory = np.zeros((maxIterations, numTargets))

    weightsHistory = np.empty((maxIterations,len(wschemes)), dtype=object)
    
    # babbling per iteration
    for it in range(maxIterations):
        print("Iteration: " + str(it))

        # babbling
        sgb.babble()
        #sgb.play_babbled_sounds()
        sgb.store_babbled_sounds(os.path.join(save_directory, "currently_babbling-run-" + str(r) + "_it-" + str(it) + ".wav"))
        
        for sch in range(len(wschemes)):
            weightsHistory[it,sch]=sgb.weights[-1][sch]

        print("Babbled")

        # plot which task space coordinates were explored
        (fig, ax) = ambSp.plot()
        ax.plot(sgb.tRolloutsDesired[:,0], sgb.tRolloutsDesired[:,1], 'k*')
        for (expl, weight) in zip(sgb.tRolloutsExpl, sgb.currentWeights):
            if weight == 0:
                ax.plot(expl[0], expl[1], 'g*')
            else:
                ax.plot(expl[0], expl[1], color='purple', marker='*', linestyle='None')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        fig.savefig(os.path.join(save_directory, 'exploration-' + str(it) + '.pdf'))
        plt.close()

        # plot the current basic functions of the inverse model learner
        (fig, ax) = ambSp.plot()
        for i in range(len(sgb.invmodel.center.C)):
            circ = mpl.patches.Circle((sgb.invmodel.center.C[i][0], sgb.invmodel.center.C[i][1]), radius=sgb.invmodel.center.radius, edgecolor = 'black', facecolor='orange', alpha = 0.5)
            plt.gca().add_patch(circ)
        fig.savefig(os.path.join(save_directory, 'invmodel-' + str(it) + '.pdf'))
        plt.close()

        # plot the current workspace state
        (fig, ax) = ambSp.plot()
        for i in range(len(sgb.wsm.S.C)):
            circ = mpl.patches.Circle((sgb.wsm.S.C[i][0], sgb.wsm.S.C[i][1]), radius=sgb.wsm.S.radius, edgecolor = 'black', facecolor='darkgrey', alpha = 0.5)
            plt.gca().add_patch(circ)
        fig.savefig(os.path.join(save_directory, 'wsm-' + str(it) + '.pdf'))
        plt.close()

		# write information to file
        with open(os.path.join(save_directory, "training-progress.txt"), 'a') as f:
            f.write("iteration " + str(it) + "\n")
            f.write(str(sgb.weights[-1]))
            f.write("\n\n")


        # external evaluation
        if it%evaluation_interval == 0 or it == maxIterations:
            (targetMeans, targetCovs) = ambSp.getTargetPositions() # ordered target positions
            invmodelEstimates = sgb.invmodel.apply(targetMeans)
            (gsPoss, actions, sounds) = sgb.fwdmodel(invmodelEstimates)
            evalErrorHistory[it_eval, :] = distance(targetMeans, gsPoss)
            evalCompetenceHistory[it_eval, :] = competence(targetMeans, gsPoss)

            # store evaluation information
            with open(os.path.join(save_directory, "training-progress.txt"), 'a') as f:
                f.write('evaluation error: ' + str(evalErrorHistory[it_eval,:]) + '\n')
                f.write('evaluation competence: ' + str(evalCompetenceHistory[it_eval,:]) + '\n')
                f.write("\n\n")

            # store sounds
            sound = np.zeros((5000,))
            for s in sounds:
                sound = np.concatenate((sound, s))
                sound = np.concatenate((sound, np.zeros(5000,)))
            sound = np.concatenate((sound, np.zeros(5000,)))
            scipy.io.wavfile.write(os.path.join(save_directory, 'evaluation-sounds-' + str(it) + '.wav'), audiosamplingrate, sound.astype('int16'))

            # stopping condition
            print("Error: " + str(distance(targetMeans, gsPoss)), "Competence: " + str(competence(targetMeans, gsPoss)))
            if it > 5 and np.max(evalErrorHistory[it_eval,:]) < stopThreshold:
                print("Sufficient competence achieved!")
                break

            # save regularly current results
            if store_experiment_each_it:
                save_experiment(save_directory, "results-" + str(it_eval*evaluation_interval), config, sgb, ambSp, arData, evalErrorHistory, evalCompetenceHistory, weightsHistory)
            else:
                save_experiment(save_directory, "results", config, sgb, ambSp, arData, evalErrorHistory, evalCompetenceHistory, weightsHistory)
            it_eval += 1
            
            if store_best_iteration:
                if np.mean(evalErrorHistory[it_eval, :]) < current_min_error:
                    current_min_error = np.mean(evalErrorHistory[it_eval, :])
                    save_experiment(save_directory, "results-best", config, sgb, ambSp, arData, evalErrorHistory, evalCompetenceHistory, weightsHistory)
                    with open(os.path.join(save_directory, "training-progress.txt"), 'a') as f:
                        f.write("New best iteration " + str(it) + " with error " + str(current_min_error))

    print("now save")
    save_experiment(save_directory, "results", config, sgb, ambSp, arData, evalErrorHistory, evalCompetenceHistory, weightsHistory)
    print("done")
