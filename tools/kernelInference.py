import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def data_of_interest(names,interest=[],exclude=[]):
    to_plot = []
    for dat in names:
        if dat in to_plot: continue
        for i in interest:
            if i in dat:
                keep = True
                for ex in exclude:
                    if ex in dat: keep = False
                #check double/triple knockdown
                if dat.count('+')>i.count('+'):
                    # print(dat)
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot


def prepareData(gene,pulses,steps,exclude=[],
                light_sample=(-9,15),step_off=True,augment=10000):
    UU = []
    ZZ = []
    '''get pulse data'''
    data_name = 'data/LDS_response_rnai.pickle'
    with open(data_name,'rb') as f:
        result = pickle.load(f)
    tp = result['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    #loop through pulses
    for j,pulse in enumerate(pulses):
        exclude_this=exclude.copy()
        if not 'p' in gene:
            exclude_this.append('p')
        tag = f'_{pulse}s'
        if pulse==5:
            tag=''
            exclude_this.append('_')
        #find datasets
        yp = []
        search=gene+tag
        to_plot = data_of_interest(result.keys(),[search],exclude_this)
        yp = [result[dat] for dat in to_plot]
        if len(yp)==0:
            print(gene,pulse,'[]',0)
            continue
        yp = np.concatenate(yp)
        print(gene, pulse, to_plot,yp.shape[0])
        #take median and append
        yp = np.median(yp,axis=0)
        u_i = np.zeros((tp.size))
        if pulse*2>=1:
            u_i[t_on:int(t_on+pulse*2)] = 1
        else:
            u_i[t_on:t_on+1] = pulse*2
        u_i = u_i[ind_t]
        ZZ.append(yp[None,ind_t])
        UU.append(u_i[None,:])
    '''get step data'''
    data_name = 'data/LDS_response_LONG.pickle'
    with open(data_name,'rb') as f:
        result_step = pickle.load(f)
    tp = result_step['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    for power in steps:
        exclude_this=exclude.copy()
        tag = f'_30m2h{power}bp'
        #find datasets
        yp = []
        search=gene+tag
        to_plot = data_of_interest(result_step.keys(),[search],exclude_this)
        yp = [result_step[dat] for dat in to_plot]
        if len(yp)==0:
            print(gene,power,'[]',0)
            continue
        yp = np.concatenate(yp)
        print(gene, power, to_plot,yp.shape[0])
        #take median and append
        yp = np.median(yp,axis=0)
        u_i = np.zeros((tp.size))
        u_i[t_on:t_on+30*120] = power/255
        ZZ.append(yp[None,ind_t])
        UU.append(u_i[None,ind_t])
        if step_off:
            ZZ.append(yp[None,ind_t+30*120])
            UU.append(u_i[None,ind_t+30*120])
    #prep and return data
    UU = np.concatenate(UU)[...,None]
    ZZ = np.concatenate(ZZ)

    if augment:
        UU = np.concatenate([UU for _ in range(augment)])
        ZZ = np.concatenate([ZZ for _ in range(augment)])
    ind = np.arange(UU.shape[0])
    np.random.shuffle(ind)
    UU = UU[ind]
    ZZ = ZZ[ind]
    return UU,ZZ

def inferKernel(interest, exclude=[], kernel_size = 5*120,
                pulses=[5,30], steps=[64]):
    kernels = []
    biases = []
    fig,ax = plt.subplots(figsize=(10,10))
    plt.plot([-kernel_size/120,0],[0,0],c='grey',ls=':')
    for gene in interest:
        exclude_this=exclude.copy()
        conditional_exclude = ['+','PreRegen','1F']
        for c_e in conditional_exclude:
            if not c_e in interest:
                exclude_this.append(c_e)
        UU, ZZ = prepareData(gene,pulses,steps,exclude_this,)
        """Build Model"""
        # frame_rate=120
        #import keras.backend as K
        U = keras.layers.Input((UU.shape[1],1))
        conv = keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')
        Z = conv(U)
        Z = keras.layers.Lambda(lambda x:K.squeeze(x,axis=-1))(Z)
        model = keras.models.Model(U,Z)
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt,loss='mse')
        model.fit(UU,ZZ[:,-Z.shape.as_list()[1]:],batch_size=23,epochs=5)
        #plot
        W = conv.get_weights()
        shift = np.squeeze(W[1])
        kernel = np.squeeze(W[0])
        tp = np.arange(-kernel.shape[0],0)/120
        ax.plot(tp,kernel,label=gene)
        kernels.append(kernel)
        biases.append(shift)
    #details
    ax.legend()
    plt.xlabel('time (min)')
    plt.ylabel('kernel value')
    plt.show()
    return fig, kernels, biases

def predictResponse(kernel,bias,stimulus,):
    Z = np.convolve(stimulus,np.flip(kernel),mode='valid')+bias
    buffer = np.zeros(kernel.size-1)*np.nan
    return np.append(buffer,Z)
