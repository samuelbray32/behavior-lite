import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap, timeDependentDifference
from .bootstrapTest import bootstrap_diff, bootstrap_relative
from .measurements import adaptationTime, peakResponse, totalResponse, totalResponse_pop, responseDuration
from .measurements import sensitization, habituation,sensitizationRate,habituationRate, tPeak
from tools.measurements import pulsesPop, sensitization, habituation
def data_of_interest(names,interest=[],exclude=[]):
    to_plot = []
    for dat in names:
        if dat in to_plot: continue
        for i in interest:
            if i in dat:
                keep = True
                for ex in exclude:
                    if ex in dat: keep = False
                if keep: to_plot.append(dat)
    return to_plot

##############################################################################################################

def rnai_response_regen(interest,exclude,n_boot=1e3,statistic=np.median,whole_ref=False,
                       ylim=(0,2),stat_testing=True,conf_interval=99):
    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    data = data_of_interest(result.keys(),interest,exclude)
    result_ref = result
    ref = []
    for d in data:
        if '30s' in d:
            ref.append('standard_30s2h')
        else:
            ref.append('standard_5s2h')
    if whole_ref:
        ref = []
        for d in data:
            if '30s' in d:
                ref.append('WT_30s')
            elif '10s' in d:
                ref.append('WT_10s')
            elif '1s' in d:
                ref.append('WT_1s')
            else:
                ref.append('WT')
        name = 'data/LDS_response_rnai.pickle'
        with open(name,'rb') as f:
            result_ref = pickle.load(f)
    
    if not(type(ylim[0]) is tuple):
        ylim = tuple([ylim for _ in range(len(data))])
                        
    day_shift = [0 for d in data]
    fig, ax = plt.subplots(ncols=len(data),nrows=len(result[(data[0])]),sharex=True,sharey='col',figsize=(4*len(data),12))
    if len(data)==1:
        ax=ax[:,None]
    if len (ref)==1:
        ref = [ref[0] for r in data]

    for i in range(len(data)):
        ax[0,i].set_title(data[i])
        print(ylim[i])
        ax[0,i].set_ylim(ylim[i])
        if whole_ref:
            print(ref[i])
            yp = result_ref[ref[i]]            
            y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)

        for j in range(len (result[(data[i])])):
            xp = result['tau']
            yp_=(result[data[i]][j])
            if yp.size==0:
                continue
            y_,rng_ = bootstrap_traces(yp_,n_boot=n_boot,statistic=statistic)
            ax[j+day_shift[i],i].plot(xp,y_,label=j, color=plt.cm.cool(j/7))
            ax[j+day_shift[i],i].fill_between(xp,*rng_,alpha=.3,edgecolor=None,facecolor=plt.cm.cool(j/7))
#         for j in range(len(result[ref[i]])):
            if not whole_ref:
                if len(result[ref[i]])<j: continue
                yp=(result_ref[ref[i]][j])
                y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
            xp_ref = result_ref['tau']
            ax[j,i].plot(xp_ref,y,color='grey',zorder=-1)
            ax[j,i].fill_between(xp_ref,*rng,alpha=.3,edgecolor=None,facecolor='grey',zorder=-2)
            
            #Do a time-dependent test on the reference and current condition
            loc=np.argmin(xp**2)
            ind_sig = np.arange(-5*120,10*120)+loc
            loc_ref=np.argmin(xp_ref**2)
            ind_sig_ref = np.arange(-5*120,10*120)+loc_ref
            if stat_testing:
                time_sig = timeDependentDifference(yp_[:,ind_sig],yp[:,ind_sig_ref],
                                                   n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp[ind_sig]
                y_sig = .1
                bott = np.zeros_like(x_sig)+ylim[i][1]
                ax[j,i].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=plt.cm.cool(j/7),alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[j,i].plot(x_sig,bott,**box_keys)
                ax[j,i].plot(x_sig,bott-y_sig,**box_keys)
                ax[j,i].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[j,i].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)
            sh=0
            if '30s' in data[i]: sh=.5
            if '5s' in data[i]: sh=5/60    
            ax[j,i].fill_between([-sh,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
            ax[j,i].spines['top'].set_visible(False)
            ax[j,i].spines['right'].set_visible(False)
            if i>0:
                ax[j,i].spines['left'].set_visible(False)
             
    fig.suptitle('regen control')
    #    ax[j].legend()
    plt.xlim(-3,20)
    #plt.yscale('log')
    ax[len(ax)//2,0].set_ylabel('Z')
    ax[-1,ax.shape[1]//2].set_xlabel('time (min)')

#     for a in ax:
#         a[0].set_yticks([0,1,2])
        
    return fig,ax

##############################################################################################################


def merge_regenerations(data,ref=None,day=None,conf_interval=99,n_boot=1e3,stat_testing=True,
                       ylim=(0,2)): 
    if day is None:
        day = [0 for _ in data]
    if ref is None:
        if '30s' in data[0]:
            ref = 'WT_30s'
        elif '10s' in data[0]:
            ref = 'WT_10s'
        elif '1s' in data[0]:
            ref = 'WT_1s'
        else:
            ref = 'WT'
    if '30s' in data[0]:
        sh=.5
    elif '10s' in data[0]:
        sh=10/60
    elif '1s' in data[0]:
        sh=1/60
    else:
        sh=5/60
    
    pl=25
    ph=75

    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_uvRange.pickle'
    ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
            result_ref = pickle.load(f)

    fig, ax = plt.subplots(nrows=9,sharex=True,sharey=True,figsize=(4,16))
    xx_ref=(result_ref[ref])
    if n_boot>1:
        yy_ref,rng_ref = bootstrap_traces(xx_ref,statistic=np.median,n_boot=n_boot,
                                          conf_interval=conf_interval)
    else:
        yy_ref=np.median(xx_ref,axis=0)
        rng_ref = (np.percentile(xx,pl,axis=0),np.percentile(xx,ph,axis=0))
    d=day.copy()
    for j in range(len(ax)):
        c = plt.cm.cool(j/len(ax))
        xx=[]
        for i,dat in enumerate(data):
            if d[i]>j or (j-d[i])>=len(result[dat]):
                continue
            xx.append((result[dat][j-d[i]]))
        xx = np.concatenate(xx)
        if xx.size==0: continue
        if n_boot>1:
            yy,rng = bootstrap_traces(xx,statistic=np.median,n_boot=n_boot,
                                      conf_interval=conf_interval)
            ax[j].plot(result['tau']-sh,yy,label=j, color=c)
            ax[j].fill_between(result['tau']-sh,*rng,alpha=.3,edgecolor=None,facecolor=c)

        else:
            ax[j].plot(result['tau']-sh,np.median(xx,axis=0),label=j, color=c)
            ax[j].fill_between(result['tau']-sh,np.percentile(xx,pl,axis=0),np.percentile(xx,ph,axis=0),alpha=.3,edgecolor=None,facecolor=c)
            xx_dat=xx.copy()
        
        ax[j].plot(result_ref['tau']-sh,yy_ref,color='grey',zorder=-1)
        ax[j].fill_between(result_ref['tau']-sh,*rng_ref,alpha=.3,edgecolor=None,facecolor='grey',zorder=-2)
        ax[j].fill_between([-sh,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
        ax[j].set_yticks([0,1,2])
        
        #Do a time-dependent test on the reference and current condition
        xp_ref = result_ref['tau']
        xp_ = result['tau']
        loc=np.argmin(xp_**2)
        ind_sig = np.arange(-5*120,10*120)+loc
        loc_ref=np.argmin(xp_ref**2)
        ind_sig_ref = np.arange(-5*120,10*120)+loc_ref
        if stat_testing:
                time_sig = timeDependentDifference(xx[:,ind_sig],xx_ref[:,ind_sig_ref],
                                                   n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp_[ind_sig]
                y_sig = .2
                bott = np.zeros_like(x_sig)+ylim[1]
                ax[j].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=plt.cm.cool(j/7),alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[j].plot(x_sig,bott,**box_keys)
                ax[j].plot(x_sig,bott-y_sig,**box_keys)
                ax[j].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[j].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)

        
    #    ax[j].legend()


    #plt.xlim(-20,60)
    #plt.yscale('log')    
    #ax[-1].set_ylim(0,.5)
    ax[-1].set_xlim(-1,7)
    ax[-1].set_ylim(ylim)
    ax[len(ax)//2].set_ylabel('Z')
    plt.xlabel('time (min)')
    #ax[0].set_title(data + ': Whole Worm control')
    return fig, ax