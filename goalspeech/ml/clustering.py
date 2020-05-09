import numpy as np
import matplotlib.pyplot as plt

from goalspeech.utils.metric import ScaledEucDistance

class ClusteringAlg:
    def __init__(self, metric = ScaledEucDistance(1)):
        self.metric = metric
        self.C = []

    def cluster(self, X):
        return np.array(self.C)

    def distance(self, a, b):
        """ Calculate distance according to self.metric.
        """
        return self.metric.distance(a, b)

    def nearestNeighbors(self, x, k):
        d = self.distance(x, np.array(self.C))
        dsorted = np.sort(d)
        n = np.argsort(d)
        return (n[0:k], dsorted[0:k])



class ITM(ClusteringAlg):

    def __init__(self, lrate = 0.01, radius = 0.1):
        ClusteringAlg.__init__(self)
        # self.metric = ScaledEucDistance(1) # or something
        self.lrate = lrate
        self.radius = radius # radius determines maximal VQ error
        self.deletion = False # allow deletion of nodes (from self.C)

        # for prototypes/nodes:
        self.C = []
        # for edges between prototypes
        self.E = []

    def numClusters(self):
        return len(self.C)

    def clusters(self):
        return self.C

    def cluster(self, X):
        """ Add clusters according to data points in X.
            X is a 2-d array numSamples x dim.
        """

        if X.ndim == 1:
            X = np.reshape(X, (1,-1))

        for i in range(np.size(X,0)):

            if not self.C:
                # this is the first input ever seen => add
                self.C.append(X[i,:])
                self.E.append([])

            elif self.numClusters() == 1:
                # only one prototype exists, add depending on distance
                d = self.distance(X[i,:], np.array(self.C))
                if d > self.radius:
                    # add prototype
                    self.C.append(X[i,:])
                    self.E.append([])
                    # connect with edges
                    self.E[0].append(1)
                    self.E[1].append(0)
            else:
                # get first two nearest neighbors
                (indices, dists) = self.nearestNeighbors(X[i,:], 2)
                n = indices[0] # nearest
                s = indices[1] # second nearest
                dist_to_n = dists[0] # dist to nearest
                dist_to_s = dists[1] # dist to second nearest

                # adapt best match
                self.C[n] = self.C[n] + self.lrate * (X[i,:] - self.C[n])

                # update edges
                # new link between best and second best match
                if not s in self.E[n]:
                    self.E[n].append(s)
                    self.E[s].append(n)

                # for each node that is connected with n
                edges_n = self.E[n]
                new_edges_n = []
                v_s_to_n = self.C[n] - self.C[s]
                for m in range(len(edges_n)):
                    # check if C[s] lies inside Thales sphere of C[n] and C[m]
                    v_s_to_m = self.C[edges_n[m]] - self.C[s]
                    isInThalesCircle = (np.dot(v_s_to_n, v_s_to_m) < 0)
                    if isInThalesCircle:
                        # delete n in the list of edges of m
                        self.E[edges_n[m]].remove(n)
                        # do NOT add m to the new edge list of n
                    else:
                        new_edges_n.append(edges_n[m])
                self.E[n] = new_edges_n

                if self.deletion:
                    # remove edgeless nodes
                    for m in range(self.numClusters()):
                        if not self.E[m]:
                            # m has no edges anymore
                            del(self.C[m])
                            del(self.E[m]) # TODO not in matlab impl???
                        if m < n:
                            n -= 1
                        if m < s:
                            s -= 1

                v_X_to_n = self.C[n] - X[i,:]
                v_X_to_s = self.C[s] - X[i,:]
                outsideThalesCircle = np.dot(v_X_to_n, v_X_to_s) > 0


                original = False

                if original:
                    # TODO: alteratively, in original [Jockusch99instantaneous]:
                    # 4. wenn X[i,:] außerhalb des thaleskreises um C[n] und C[s]
                    # und außerhalb eines kreises um C[n] liegt mit radius Abstand
                    # new node y mit C[y] = X[i,:]
                    # connect y and n
                    # if C[n] and C[s] closer than 0.5 * radius, remove s

                    if outsideThalesCircle and dist_to_n > self.radius:
                        self.E[n].append(self.numClusters())
                        self.C.append(X[i,:])
                        self.E.append([n])
                    if self.deletion and self.distance(self.C[n], self.C[s]) < 0.5 * self.radius:
                        # remove node and according edge list
                        del(self.C[s])
                        del(self.E[s])
                        # remove all references to node s
                        for m in range(len(self.E)):
                            self.E[m].remove(s)

                else:
                    # matlab implementation from mlt toolbox [Reinhart et al.]

                    # if X lies outside of circle around n
                    if dist_to_n > self.radius and dist_to_s > self.radius:
                        # create new node and connect n to it
                        self.E[n].append(self.numClusters())
                        self.C.append(X[i,:])
                        self.E.append([n])

                        if not outsideThalesCircle:
                            # n-x replaces n-s
                            self.E[n].remove(s)
                            self.E[s].remove(n)

                    if self.deletion and self.distance(self.C[n], self.C[s]) < 0.25 * self.radius:
                        # remove node and according edge list
                        del(self.C[s])
                        del(self.E[s])
                        # remove all references to node s
                        for m in range(len(self.E)):
                            self.E[m].remove(s)

    def plot(self):
        if self.C:
            if len(self.C[0]) != 2:
                raise NotImplementedError("Plotting for dimensionality != 2 not supported yet.")

            Cx = np.array(self.C)[:,0]
            Cy = np.array(self.C)[:,1]

            plt.figure()
            plt.scatter(Cx, Cy)

            for i in range(len(self.E)):
                Estart = self.C[i]
                print("from " + str(Estart[0]) + "," + str(Estart[1]))
                for j in range(len(self.E[i])):
                    Eend = self.C[self.E[i][j]]
                    print("to " + str(Eend[0]) + "," + str(Eend[1]))
                    plt.plot([Estart[0], Eend[0]], [Estart[1], Eend[1]], color='k', linestyle='-', linewidth=2)

            plt.show(block = False)
