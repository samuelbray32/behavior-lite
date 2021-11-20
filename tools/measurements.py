"""
Set of scalar measurements of pulse response that are compatible with bootstrapping
"""
import numpy as np

def adaptationTime(data):
    '''time for population response to come down to 1/2 it's peak value (see Uri's paper)'''
    x = np.median(data,axis=0)
    x -= np.percentile(x,5)#x.min()#
    x_max = x.max()
    loc = np.argmax(x)
    try:
        return np.where(x[loc:]/x.max()<=.5)[0][0]/120
    except: return 0
    
def responseDuration(data,thresh=.3):
    '''time for population traces to go below threshold value'''
    x = np.median(data,axis=0)
    try:
        return np.where(x<=thresh)[0][0]/120
    except: return 1/120 #return small value so don't get a divide by zero in comparison tests

def peakResponse(data):
    '''peak response of population trace'''
    return np.max(np.median(data,axis=0))


def totalResponse(data):
    '''peak response of population trace'''
    return np.mean(data,axis=1)


