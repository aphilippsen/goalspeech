import numpy as np

# http://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range
def scale1d(dat, out_range=(-1, 1)):
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))
    
#def scale2d(dat, out_range=(-1, 1)):
#    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

#    def interp(x):
#        return out_range[0] * (1.0 - x) + out_range[1] * x

#    def uninterp(x):
#        b = 0
#        if (domain[1] - domain[0]) != 0:
#            b = domain[1] - domain[0]
#        else:
#            b =  1.0 / domain[1]
#        return (x - domain[0]) / b

#    res = np.empty((np.size(dat,0), np.size(dat,1)))
#    for i in range(np.size(dat,0)):
#        res[i] = interp(uninterp(dat[i]))
    
