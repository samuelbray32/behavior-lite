"""
Set of scalar measurements of pulse response that are compatible with bootstrapping
"""
import numpy as np

######################################
#       Trace measurements           #
######################################
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

def totalResponse_pop(data):
    '''total response of population trace'''
    return np.mean(np.median(data,axis=0))

def totalResponse(data):
    '''total response of individual trace'''
    return np.mean(data,axis=1)

######################################
#     Pulse train measurements       #
######################################
from scipy.stats import linregress
def pulsesPop(data,n,isi,integrate):
    y = np.median(data,axis=0)
    return np.array([y[i*isi:i*isi+integrate].mean() for i in range(n)])

def sensitization(data,**kwargs):
    ''' (population peak response- population first response)'''
    y = pulsesPop(data,**kwargs)
    return y[5]-y[0] #(y.max()-y[0])#/np.argmax(y)

def habituation(data,**kwargs):
    ''' (population peak response-population last response)'''
    y = pulsesPop(data,**kwargs)
    return y.max()-y[-1]#y.max()-y[-1]#/(y.size-np.argmax(y))



def sensitizationRate(data):
    ''' (peak response-first response)/t_peak'''
    #shape = (samples, pulses)
    t_peak = np.argmax(data,axis=1)
#     return (data.max(axis=1)-data[:,0])/(t_peak+1)
    values = []
    for i in range(data.shape[0]):
        xx = np.arange(max(t_peak[i],1))
        values.append(linregress(xx,data[i,:xx.size])[0])
    values = np.array(values)
    values[~np.isfinite(values)] = 0
    return values

def habituationRate(data):
    ''' (peak response-final response)/(t_final-t_peak+1)'''
    #shape = (samples, pulses)
    t_peak = np.argmax(data,axis=1)
#     return (data.max(axis=1)/data[:,-1])/(data.shape[1]-t_peak+1)
    values = []
    for i in range(data.shape[0]):
        xx = np.arange(max(data.shape[1]-t_peak[i],1))
        values.append(linregress(xx,data[i,-xx.size:])[0])
    values = -np.array(values)
    values[~np.isfinite(values)] = 0
    return values


# def sensitization(data):
#     ''' (peak response-first response)'''
#     #shape = (samples, pulses)
#     return (data.max(axis=1)-data[:,0])
#
# def habituation(data):
#     ''' (peak response-first response)'''
#     #shape = (samples, pulses)
#     return (data.max(axis=1)-data[:,-1])

def tPeak(data):
    ''' (peak response-first response)'''
    #shape = (samples, pulses)
    return data.argmax(axis=1)/data.shape[1]
