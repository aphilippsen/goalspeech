import copy

import matplotlib.pyplot as plt
import numpy as np

from goalspeech.utils.metric import ScaledEucDistance

d = ScaledEucDistance()

# CALCULATIONS

def distance(desired, reached):
    return d.distance(desired, reached)

def competence(desired, reached):
    return np.exp(-distance(desired, reached))

def external_evaluation(ambSp, gb):
        (targetMeans, targetCovs) = ambSp.getTargetPositions()
        invmodelEstimates = gb.invmodel.apply(targetMeans)
        (gsPoss, actions, sounds) = gb.fwdmodel(invmodelEstimates)
        extEvalError = distance(targetMeans, gsPoss)
        extEvalComp = competence(targetMeans, gsPoss)
        return (extEvalError, extEvalComp)

# PLOTTING

plot_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'darkorange', 'navy', 'yellow', 'black']

def plot_internal_competence(sgb, sequence_order):
    plt.figure()

    for (i, color, target_name) in zip(range(np.size(sgb.competenceHistory)), plot_colors, sequence_order):
        iters = [x[0] for x in sgb.competenceHistory[i]]
        comps = [x[1] for x in sgb.competenceHistory[i]]
        actN = [x[2] for x in sgb.competenceHistory[i]]

        plt.plot(iters, comps, color=color, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show(block = False)

def plot_external_evaluation(externalHistory, sequences):
    plt.figure()

    for (i, color, target_name) in zip(range(np.shape(externalHistory)[1]), plot_colors, sequences):
        plt.plot(range(np.shape(externalHistory)[0]), externalHistory[:,i], color=color, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show(block = False)


# INTERACTING

def imitate_target(sgb, target):
    action = sgb.invmodel.apply(target)
    [reachedPos, action, sig] = sgb.fwdmodel(action)
    return (reachedPos, sig)

def imitate_all(sgb):
    desired = copy.deepcopy(sgb.wsm.targetDist.means_)
    reached = np.zeros(desired.shape)
    sounds = np.empty(shape=(np.size(desired,0)), dtype=object)
    for i in range(np.size(sgb.wsm.targetDist.means_,0)):
        (reached[i], sounds[i]) = imitate_target(sgb, desired[i])
    return (desired, reached, sounds)

def interaction_loop(sgb):

    plt.figure()
    lw = 2

    # plot mappedData
    for (i, color, target_name) in zip(range(6), colors, self.sequences):
        plt.scatter(self.mappedData[allLabels == i, 0], self.mappedData[allLabels == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("Goal space of ambient speech\nFeatures: " + self.featureMode + "  Dim.red.method: " + self.embMethod)

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
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color = 'black')
            #ell.set_clip_box(plt.bbox)
            ell.set_alpha(1)
            plt.gca().add_patch(ell)

    plt.show(block = False)
