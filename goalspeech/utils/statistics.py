import numpy as np
import scipy.stats

def get_mean_and_confidence_interval(data, confidence=0.95, axis=None):
    m, se = np.mean(data, axis=axis), scipy.stats.sem(data, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return m, m-h, m+h
    
