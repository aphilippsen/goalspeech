import numpy as np

class Normalizer:
    # offset, range, minmax
    def __init__(self):
        self.threshold = 1e-10 # if valrange < threshold, do not scale this parameter!

    def normalizeData(self, dataOrig, minmax = None, margin = 0):
        ''' Initializes offset, range and minmax
            If dtype==object, assume [[]] entries
        '''

        if dataOrig.dtype == object:
            # cell2mat:
            if dataOrig.ndim == 2:
                # TODO care for all cases!
                dataOrig = np.concatenate(dataOrig)
                data = np.concatenate(np.concatenate(dataOrig))
            elif dataOrig.ndim == 1:
                if dataOrig[0].ndim == 1:
                    data = np.concatenate([sh.reshape(1,-1) for sh in dataOrig])
                else:
                    data = np.concatenate(dataOrig)
        else:
            data = dataOrig

        #if np.ndim(data)
        self.paramDims = np.size(data,1)


        # columnwise mean values
        self.offset = np.mean(data, axis=0)

        # shift data
        shiftedData = data - self.offset

        # columnwise minmax values
        mins = np.min(shiftedData,axis=0)
        maxs = np.max(shiftedData,axis=0)

        rr = maxs - mins
        
        mins[rr >= self.threshold] = mins[rr >= self.threshold] - (maxs[rr >= self.threshold]-mins[rr >= self.threshold])*margin
        maxs[rr >= self.threshold] = maxs[rr >= self.threshold] + (maxs[rr >= self.threshold]-mins[rr >= self.threshold])*margin
        
        rr[rr < self.threshold] = 0
        rr[rr >= self.threshold] = maxs[rr >= self.threshold] - mins[rr >= self.threshold]
        
        mins = maxs - rr
        self.range = np.array([mins, maxs])

        self.minmax = minmax
        if minmax is None:
            self.minmax = np.array([np.repeat([-1], self.paramDims), np.repeat([1], self.paramDims)])

        # scale data
        if dataOrig.dtype == object:
            #if dataOrig.ndim == 2:
            #res = np.empty(np.size(dataOrig[0]), dtype=object)
            #for i in range(np.size(dataOrig[0],0)):
            #    res[i] = self.range2norm(dataOrig[0][i])
            #elif dataOrig.ndim == 1:
            res = np.empty(np.size(dataOrig), dtype=object)
            for i in range(np.size(dataOrig,0)):
                res[i] = self.range2norm(dataOrig[i])

            return res
        else:
            return self.range2norm(dataOrig)


    def range2norm(self, x):
        shiftedData = x - self.offset

        oldBase = self.range[0,:]
        newBase = self.minmax[0,:]

        oldRange = self.range[1,:] - self.range[0,:]
        oldRange[oldRange == 0] = 1
        newRange = self.minmax[1,:] - self.minmax[0,:]

        y = newRange * (shiftedData - oldBase)  / oldRange + newBase
        return y

    def norm2range(self, y):

        oldBase = self.range[0,:]
        newBase = self.minmax[0,:]

        oldRange = self.range[1,:] - self.range[0,:]
        newRange = self.minmax[1,:] - self.minmax[0,:]
        newRange[newRange == 0] = 1

        x = oldRange * (y - newBase) / newRange + oldBase + self.offset
        return x
