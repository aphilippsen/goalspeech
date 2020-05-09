import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile

from goalspeech.experiment import load_experiment
from goalspeech.evaluate import competence, distance

main_results_dir = "/home/anja/repos/goalspeech/data/results/vowels/act-ad/"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/vowels/act-fi/"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/vowels/noa-ad/"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/vowels/noa-fi/"
runs=30

#main_results_dir = "/home/anja/repos/goalspeech/data/results/syllables_aa_baa_maa/"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/2020-5-3_17-51_2610_act-fi"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/2020-5-4_21-22_4451_noa-ad"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/2020-5-6_20-31_4362"
#runs=3

#main_results_dir = "/home/anja/repos/goalspeech/data/results/2020-5-1_13-26_3295/"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/2020-5-3_17-51_2610"
#main_results_dir = "/home/anja/repos/goalspeech/data/results/2020-5-4_21-22_4451"
#runs=1


# TODO read it from config file
max_epochs = 500
evaluation_interval = 10

# which evaluations to perform
generate_sounds = False
generate_generalization_sounds = False
generate_interpolations = True

interp_factor = 0.1
interpolations = np.arange(0, 1+interp_factor, interp_factor)

allSounds = np.empty((runs,),dtype=object)
evalError = np.empty((runs,), dtype=object)
evalCompetence = np.empty((runs), dtype=object)
arNoise = np.empty((runs,),dtype=object)
weights = np.empty((runs,),dtype=object)

target_error = np.empty((runs,),dtype=object)
target_comp = np.empty((runs,),dtype=object)
generalization_error = np.empty((runs,),dtype=object)
generalization_comp = np.empty((runs,),dtype=object)

interp_sounds = np.empty((runs,),dtype=object)
ar_interp = np.empty((runs,),dtype=object)
gs_interp = np.empty((runs,),dtype=object)

# which pattern was explored in which iteration?
class_per_ep = np.empty((runs,), dtype=object)

# load evaluation data from all runs
for r in range(runs):
    print('Evaluate run ' + str(r))
    results_dir = os.path.join(main_results_dir, 'run-' + str(r))
    expData = load_experiment(results_dir, "results.pickle")
    
    #rawFeatures = np.load(os.path.join(main_results_dir, "ambSp_rawFeatures.npy"), allow_pickle=True)
    #expData.ambSp.rawFeatures = rawFeatures

    evalError[r] = expData.errorData[:int(max_epochs/evaluation_interval),:]
    evalCompetence[r] = expData.compData[:int(max_epochs/evaluation_interval),:]
    try:
        weights[r] = expData.weights
    except:
        pass

    assert(evalError[r].shape == evalCompetence[r].shape)

    # fill up potential zeros if learning stopped earlier than at max_epochs
    for its in range(evalError[r].shape[0]):
        if (evalError[r][its,:]==0).all():
            evalError[r][its,:] = evalError[r][its-1,:]
        if (evalCompetence[r][its,:]==0).all():
            evalCompetence[r][its,:] = evalCompetence[r][its-1,:]

    # find the iteration with the minimum error and take it as the final epoch
    min_error_it = np.argmin(np.sum(evalError[r], axis=1))
    for it in range(min_error_it+1, evalError[r].shape[0]):
        evalError[r][it,:] = evalError[r][it-1,:]
        evalCompetence[r][it,:] = evalCompetence[r][it-1,:]

    # reload experiment of best iteration if available
    try:
        # TODO fix! even if loading it, the epochs are not correct later.... use min_error_it maybe
        expData2 = load_experiment(results_dir, "results-best.pickle", sp=expData.sp, ambSp=expData.ambSp, arData=expData.arData)
        if expData2.gb.fwdmodel:
            expData.gb = expData2.gb
        else:# load gb object from newly loaded file, but maintain original fwdmodel
            expData2.gb.fwdmodel = expData.gb.fwdmodel
            expData.gb = expData2.gb
    except:
        # loaded results are already best or no individually stored results
        if str(min_error_it*evaluation_interval<max_epochs):
            print("Best results' network cannot be loaded.")

    # store which classes were explored at which iteration
    class_per_ep[r] = np.zeros((max_epochs,))
    for i in range(len(expData.gb.competenceHistory)): # number of classes
        for j in range(len(expData.gb.competenceHistory[i])): # number of explorations
            class_per_ep[r][expData.gb.competenceHistory[i][j][0]] = i

    # collect articulatory noise value per iteration
    numShapes = expData.gb.competenceHistory.shape[0]
    arNoise[r] = np.zeros((max_epochs, numShapes))

    for sh in range(numShapes):
        for i in range(len(expData.gb.competenceHistory[sh])):
            arNoise[r][expData.gb.competenceHistory[sh][i][0],sh] = expData.gb.competenceHistory[sh][i][2]

    arNoise[r][0,:] = np.max(arNoise[r][0,:])
    for j in range(arNoise[r].shape[1]):
        for k in range(arNoise[r].shape[0]):
            if arNoise[r][k,j] == 0:
                arNoise[r][k,j] = arNoise[r][k-1,j]

    if generate_sounds:
        (targetMeans, targetCovs) = expData.ambSp.getTargetPositions() # ordered target positions
        invmodelEstimates = expData.gb.invmodel.apply(targetMeans)
        (gsPoss, actions, allSounds[r]) = expData.gb.fwdmodel(invmodelEstimates)
        target_error[r] = distance(targetMeans, gsPoss)
        target_comp[r] = competence(targetMeans, gsPoss)

    # generalization in goal space
    if generate_generalization_sounds:
        shift_factor = 0.1
        shifts = np.concatenate((expData.ambSp.getTargetPositions()[0]+np.tile([shift_factor, 0], (len(expData.ambSp.getTargetPositions()[0]),1)),
                    expData.ambSp.getTargetPositions()[0]+np.tile([-shift_factor, 0], (len(expData.ambSp.getTargetPositions()[0]),1)),
                    expData.ambSp.getTargetPositions()[0]+np.tile([0, shift_factor], (len(expData.ambSp.getTargetPositions()[0]),1)),
                    expData.ambSp.getTargetPositions()[0]+np.tile([0, -shift_factor], (len(expData.ambSp.getTargetPositions()[0]),1)), 
                    ))
        invmodelEstimates = expData.gb.invmodel.apply(shifts)
        (gsPos_generaliz, _, _) = expData.gb.fwdmodel(invmodelEstimates)

        generalization_error[r] = distance(shifts, gsPos_generaliz)
        generalization_comp[r] = competence(shifts, gsPos_generaliz)

    if generate_interpolations:
        # get the target positions
        (targetMeans, targetCovs) = expData.ambSp.getTargetPositions() # ordered target positions
        
        ar_interp[r] = []
        gs_interp[r] = []
        interp_sounds[r] = []
        
        neutralIdx = expData.ambSp.sequences.index(expData.config["ambientSpeech"]["neutral"]) 
        # for all non-neutral sequences
        for s in range(len(expData.ambSp.sequences)):
            if s != neutralIdx:
                # get interpolations between neutral and target positions
                targetInterp = []
                for ip in interpolations:
                    targetInterp.append(ip * targetMeans[s] + (1-ip) * targetMeans[neutralIdx])
                
                # invmodel: estimate corresponding motor commands
                invmodelEstimates = expData.gb.invmodel.apply(np.asarray(targetInterp))
                (gsPoss, actions, sounds) = expData.gb.fwdmodel(invmodelEstimates)
                ar_interp[r].append(actions)
                gs_interp[r].append(gsPoss)
                interp_sounds[r].append(sounds)

if generate_sounds:
    sound = np.zeros((5000,))
    for runSounds in allSounds:
        if runSounds is None:
            continue
        for s in runSounds:
            sound = np.concatenate((sound, s))
            sound = np.concatenate((sound, np.zeros(5000,)))
        sound = np.concatenate((sound, np.zeros(5000,)))
    scipy.io.wavfile.write(main_results_dir + "/evaluation-sounds.wav", expData.sp.audioSamplingRate, sound.astype('int16'))

meanError = np.mean(evalError)
stdError = np.std(evalError)
meanComp = np.mean(evalCompetence)
stdComp = np.std(evalCompetence)

np.save(os.path.join(main_results_dir, "evalError.npy"), evalError)
np.save(os.path.join(main_results_dir, "evalCompetence.npy"), evalCompetence)

# write in text format for plotting


if generate_sounds:
    np.save(os.path.join(main_results_dir, "target_error.npy"), target_error)
    np.save(os.path.join(main_results_dir, "target_comp.npy"), target_comp)

if generate_generalization_sounds:
    np.save(os.path.join(main_results_dir, "generalization_error.npy"), generalization_error)
    np.save(os.path.join(main_results_dir, "generalization_comp.npy"), generalization_comp)

if generate_interpolations:
    np.save(os.path.join(main_results_dir, "ar_interpolations.npy"), ar_interp)
    np.save(os.path.join(main_results_dir, "gs_interpolations.npy"), gs_interp)
    
    for r in range(len(interp_sounds)):
        runSounds = interp_sounds[r]
        if runSounds is None:
            continue
        sound_interp = np.zeros((5000,))
        for pat in range(len(runSounds)):
            for ip in runSounds[pat]:
                sound_interp = np.concatenate((sound_interp, ip))
                sound_interp = np.concatenate((sound_interp, np.zeros(5000,)))
            sound_interp = np.concatenate((sound_interp, np.zeros(5000,)))
        scipy.io.wavfile.write(main_results_dir + "/evaluation-sound_interp_" + str(r) + ".wav", expData.sp.audioSamplingRate, sound_interp.astype('int16'))
    

lw = 3
fig = plt.figure('Per epoch error', figsize=(15, 10))
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
ax = fig.add_subplot(111)
for i in range(meanError.shape[1]):
    ax.plot(np.arange(0,max_epochs,evaluation_interval), meanError[:,i], label=expData.ambSp.sequences[i], linewidth=lw)
    ax.fill_between(np.arange(0,max_epochs,evaluation_interval), meanError[:,i] - stdError[:,i], meanError[:,i] + stdError[:,i],
    alpha=0.3)
plt.xlabel("training epoch")
plt.ylabel("mean square error")
plt.legend()
plt.savefig(os.path.join(main_results_dir, "error-statistic.pdf"))
plt.close()

fig = plt.figure('Per epoch competence', figsize=(15, 10))
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
ax = fig.add_subplot(111)
for i in range(meanComp.shape[1]):
    ax.plot(np.arange(0,max_epochs,evaluation_interval), meanComp[:,i], label=expData.ambSp.sequences[i], linewidth=lw)
    ax.fill_between(np.arange(0,max_epochs,evaluation_interval), meanComp[:,i] - stdComp[:,i], meanComp[:,i] + stdComp[:,i],
    alpha=0.3)
plt.xlabel("training epoch")
plt.ylabel("competence")
plt.legend()
plt.savefig(os.path.join(main_results_dir, "competence-statistic.pdf"))
plt.close()

# plot the decrease of articulatory noise during babbling
meanArNoise = np.mean(arNoise)
stdArNoise = np.std(arNoise)

fig = plt.figure('Per epoch articulatory noise level', figsize=(15, 10))
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
ax = fig.add_subplot(111)
for i in range(meanArNoise.shape[1]):
    ax.plot(np.arange(max_epochs), meanArNoise[:,i], label=expData.ambSp.sequences[i], linewidth=lw)
    ax.fill_between(np.arange(max_epochs), meanArNoise[:,i] - stdArNoise[:,i], meanArNoise[:,i] + stdArNoise[:,i],
    alpha=0.3)
plt.xlabel("training epoch")
plt.ylabel("articulatory noise amplitude")
plt.legend()
plt.savefig(os.path.join(main_results_dir, "arNoise-statistic.pdf"))
plt.close()

# plot the weights of the babbled speech sounds over time
colors=['red', 'green', 'blue', 'cyan', 'magenta', 'darkorange', 'navy', 'yellow', 'black', 'purple']

fig = plt.figure('weights history', figsize=(15,10))
plt.rcParams.update({'font.size':20, 'legend.fontsize':20})
ax = fig.add_subplot(111)
weight_max_ep = np.zeros((runs,))
if not weights[0] is None:
    for r in range(len(weights)): # number of runs
        for ep in range(weights[r].shape[0]): # number of epochs
            try:
                weights_ep = np.prod(weights[r][ep,:])
                ax.scatter(np.tile(ep, (len(weights_ep),)), weights_ep, color=colors[int(class_per_ep[r][ep])])
            except:
                weight_max_ep[r] = ep-1
                print("Weight history available until iteration " + str(ep-1))
                break
plt.legend(expData.ambSp.sequences)
plt.savefig(os.path.join(main_results_dir, "weights-history.pdf"))
plt.close()

"""
fig = plt.figure('weights history per weighting scheme', figsize=(15,10))
plt.rcParams.update({'font.size':20, 'legend.fontsize':20})
ax = fig.add_subplot(111)
for r in range(len(weights)):
    for i in range(len(expData.gb.wschemes)):
        weight_sum = [np.median(weights[r][x,:][i]) for x in range(len(weights[r]))]
        ax.plot(np.arange(int(weight_max_ep[r])), weight_sum, color=colors[i], label='weighting scheme '+str(i+1))
    plt.legend()
    plt.savefig(os.path.join(main_results_dir, "weighting-scheme-sum.pdf"))
plt.close()
"""



# generate text file with error values
iterations = np.arange(0,max_epochs,evaluation_interval)+evaluation_interval
with open(os.path.join(main_results_dir, "iterations-errors.txt"), "w") as f:
    f.write('t')
    for i in expData.ambSp.sequences:
        f.write("\tmean" + i + "\tstd" + i)
    f.write('\n')
    for t in range(meanError.shape[0]):
        f.write(str(iterations[t]))
        for i in range(len(expData.ambSp.sequences)):
            f.write("\t" + str(meanError[t,i]) + "\t" + str(stdError[t,i]))
        f.write("\n")

iterations = np.arange(0,max_epochs,evaluation_interval)+evaluation_interval
with open(os.path.join(main_results_dir, "iterations-comps.txt"), "w") as f:
    f.write('t')
    for i in expData.ambSp.sequences:
        f.write("\tmean" + i + "\tstd" + i)
    f.write('\n')
    for t in range(meanComp.shape[0]):
        f.write(str(iterations[t]))
        for i in range(len(expData.ambSp.sequences)):
            f.write("\t" + str(meanComp[t,i]) + "\t" + str(0.5*stdComp[t,i]))
        f.write("\n")

# store variance instead of standard deviation
iterations = np.arange(0,max_epochs,evaluation_interval)+evaluation_interval
with open(os.path.join(main_results_dir, "iterations-comps-var.txt"), "w") as f:
    f.write('t')
    for i in expData.ambSp.sequences:
        f.write("\tmean" + i + "\tstd" + i)
    f.write('\n')
    for t in range(meanComp.shape[0]):
        f.write(str(iterations[t]))
        for i in range(len(expData.ambSp.sequences)):
            f.write("\t" + str(meanComp[t,i]) + "\t" + str(stdComp[t,i]*stdComp[t,i]))
        f.write("\n")
        
        
# write ar noise to text file
iterations = np.arange(0, max_epochs)
with open(os.path.join(main_results_dir, "iterations-arNoise.txt"), "w") as f:
    f.write('t')
    for i in expData.ambSp.sequences:
        f.write("\tmean" + i)
    f.write('\n')
    for t in np.arange(0, meanArNoise.shape[0], evaluation_interval):
        f.write(str(iterations[t]))
        for i in range(len(expData.ambSp.sequences)):
            f.write("\t" + str(np.mean(meanArNoise[t:t+evaluation_interval,i])))
        f.write("\n")

        
        
        
### evaluation ###

"""
Checking the generalization capability of the different vowel settings

err_actad = np.load("/home/anja/repos/goalspeech/data/results/vowels/act-ad/target_comp.npy", allow_pickle=True)
gen_actad = np.load("/home/anja/repos/goalspeech/data/results/vowels/act-ad/generalization_comp.npy", allow_pickle=True)

err_actfi = np.load("/home/anja/repos/goalspeech/data/results/vowels/act-fi/target_comp.npy", allow_pickle=True)
gen_actfi = np.load("/home/anja/repos/goalspeech/data/results/vowels/act-fi/generalization_comp.npy", allow_pickle=True)

err_noaad = np.load("/home/anja/repos/goalspeech/data/results/vowels/noa-ad/target_comp.npy", allow_pickle=True)
gen_noaad = np.load("/home/anja/repos/goalspeech/data/results/vowels/noa-ad/generalization_comp.npy", allow_pickle=True)

err_noafi = np.load("/home/anja/repos/goalspeech/data/results/vowels/noa-fi/target_comp.npy", allow_pickle=True)
gen_noafi = np.load("/home/anja/repos/goalspeech/data/results/vowels/noa-fi/generalization_comp.npy", allow_pickle=True)


mean_err_actad = np.mean(np.concatenate(err_actad).reshape((-1,6)),axis=0)
std_err_actad = np.std(np.concatenate(err_actad).reshape((-1,6)),axis=0)
mean_gen_actad = np.mean(np.concatenate(gen_actad).reshape((-1,6)),axis=0)
std_gen_actad = np.std(np.concatenate(gen_actad).reshape((-1,6)),axis=0)

mean_err_actfi = np.mean(np.concatenate(err_actfi).reshape((-1,6)),axis=0)
std_err_actfi = np.std(np.concatenate(err_actfi).reshape((-1,6)),axis=0)
mean_gen_actfi = np.mean(np.concatenate(gen_actfi).reshape((-1,6)),axis=0)
std_gen_actfi = np.std(np.concatenate(gen_actfi).reshape((-1,6)),axis=0)

mean_err_noaad = np.mean(np.concatenate(err_noaad).reshape((-1,6)),axis=0)
std_err_noaad = np.std(np.concatenate(err_noaad).reshape((-1,6)),axis=0)
mean_gen_noaad = np.mean(np.concatenate(gen_noaad).reshape((-1,6)),axis=0)
std_gen_noaad = np.std(np.concatenate(gen_noaad).reshape((-1,6)),axis=0)

mean_err_noafi = np.mean(np.concatenate(err_noafi).reshape((-1,6)),axis=0)
std_err_noafi = np.std(np.concatenate(err_noafi).reshape((-1,6)),axis=0)
mean_gen_noafi = np.mean(np.concatenate(gen_noafi).reshape((-1,6)),axis=0)
std_gen_noafi = np.std(np.concatenate(gen_noafi).reshape((-1,6)),axis=0)

"""


### interpolation evaluation ###

# plot acquired goal space positions
#plt.figure()
#plt.plot(gs_interp[0][0][:,0], gs_interp[0][0][:,1])
#plt.plot(gs_interp[0][1][:,0], gs_interp[0][1][:,1])
#plt.plot(gs_interp[0][2][:,0], gs_interp[0][2][:,1])
#plt.plot(gs_interp[0][3][:,0], gs_interp[0][3][:,1])
#plt.plot(gs_interp[0][4][:,0], gs_interp[0][4][:,1])
#plt.show()

try:
    gs_interp = np.load(os.path.join(main_results_dir, "gs_interpolations.npy"), allow_pickle=True)
    neutralIdx = expData.ambSp.sequences.index(expData.config["ambientSpeech"]["neutral"]) 

    # get distances to target positions
    dists = np.zeros((runs, len(gs_interp[0]),  len(gs_interp[0][0])))
    for run in np.arange(0,runs):
        results_dir = os.path.join(main_results_dir, 'run-' + str(run))
        expData = load_experiment(results_dir, "results.pickle")
        (targetMeans, targetCovs) = expData.ambSp.getTargetPositions()

        targets = []
        for i in range(len(targetMeans)):
            if i != neutralIdx:
                targets.append(targetMeans[i])

        with open(os.path.join(main_results_dir, "interpolation-gs-dist-to-target_run-" + str(run) + ".txt"), "w") as f:
            f.write('t')
            for i in range(len(expData.ambSp.sequences)):
                if i != neutralIdx:
                    f.write("\t" + expData.ambSp.sequences[i])
            f.write('\n')
            for t in range(len(gs_interp[0][0])):
                f.write(str(interpolations[t]))
                for s in range(len(gs_interp[0])):
                    dists[run, s, t] = distance(targets[s], gs_interp[run][s][t,:])[0]
                    f.write("\t" + str(dists[run, s, t]))
                f.write("\n")

    # take the mean across different runs
    mean_interp = np.mean(dists, axis=0)
    std_interp = np.std(dists, axis=0)
    with open(os.path.join(main_results_dir, "interpolation-gs-dist-to-target.txt"), "w") as f:
        f.write('t')
        for i in range(len(expData.ambSp.sequences)):
            if i != neutralIdx:
                f.write("\t" + expData.ambSp.sequences[i] + "\tstd" + expData.ambSp.sequences[i])
        f.write('\n')
        for t in range(len(gs_interp[0][0])):
            f.write(str(interpolations[t]))
            for s in range(len(gs_interp[0])):
                f.write("\t" + str(mean_interp[s, t]) + "\t" + str(0.5*std_interp[s, t]))
            f.write("\n")


    # get the distance to the articulatory pattern acquired for the target
    ar_interp = np.load(os.path.join(main_results_dir, "ar_interpolations.npy"), allow_pickle=True)
    dists = np.zeros((runs, len(ar_interp[0]),  len(ar_interp[0][0])))
    for run in np.arange(0,runs):
        #results_dir = os.path.join(main_results_dir, 'run-' + str(run))
        #expData = load_experiment(results_dir, "results.pickle")
        #(targetMeans, targetCovs) = expData.ambSp.getTargetPositions()
        #targets = []
        #for i in range(len(targetMeans)):
        #    if i != neutralIdx:
        #        targets.append(targetMeans[i])

        with open(os.path.join(main_results_dir, "interpolation-ar-dist-to-target_run-" + str(run) + ".txt"), "w") as f:
            f.write('t')
            for i in range(len(expData.ambSp.sequences)):
                if i != neutralIdx:
                    f.write("\t" + expData.ambSp.sequences[i])
            f.write('\n')
            for t in range(len(ar_interp[0][0])):
                f.write(str(interpolations[t]))
                for s in range(len(ar_interp[0])):
                    dists[run, s, t] = distance(ar_interp[run][s][-1,:], ar_interp[run][s][t,:])[0]
                    f.write("\t" + str(dists[run, s, t]))
                f.write("\n")

    mean_interp = np.mean(dists, axis=0)
    std_interp = np.std(dists, axis=0)
    with open(os.path.join(main_results_dir, "interpolation-ar-dist-to-target.txt"), "w") as f:
        f.write('t')
        for i in range(len(expData.ambSp.sequences)):
            if i != neutralIdx:
                f.write("\t" + expData.ambSp.sequences[i] + "\tstd" + expData.ambSp.sequences[i])
        f.write('\n')
        for t in range(len(ar_interp[0][0])):
            f.write(str(interpolations[t]))
            for s in range(len(ar_interp[0])):
                f.write("\t" + str(mean_interp[s, t]) + "\t" + str(0.5*std_interp[s, t]))
            f.write("\n")
except:
    print("Interpolation could not be evaluated.")



"""
# determine articulatory effort:
#main_results_dir = '/home/anja/repos/goalspeech/data/results/vowels/noa-fi/'
#ar_interp = np.load('/home/anja/repos/goalspeech/data/results/vowels/noa-fi/ar_interpolations.npy', allow_pickle=True)                                                                                                  

effort = np.zeros((runs, len(ar_interp[0])))

for run in np.arange(0, runs):
    results_dir = os.path.join(main_results_dir, 'run-' + str(run))
    expData = load_experiment(results_dir, "results.pickle")
    neutralShape = expData.sp.getArticulatoryShapes('@')[0]
    for s in range(len(ar_interp[run])):
        effort[run, s] = distance(expData.arData.denormalize(ar_interp[run][s][-1,:]), neutralShape)
np.mean(np.mean(effort,axis=0))
"""

"""
# example weight history
with open(os.path.join(main_results_dir, "example-weights.txt"), "w") as f:
    f.write('t\tw1\tw2\n')
    
    for it in np.arange(0, max_epochs, evaluation_interval):
        f.write(str(it+evaluation_interval))
        for w in range(2):
            f.write("\t" + str(np.mean(np.mean(weights[0][it:it+10,w]))))
        f.write("\n")
"""

