import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap, timeDependentDifference
from .bootstrapTest import bootstrap_diff, bootstrap_relative
from .measurements import adaptationTime, peakResponse, totalResponse, totalResponse_pop, responseDuration
from .measurements import sensitization, habituation,sensitizationRate,habituationRate, tPeak
from tools.measurements import pulsesPop, sensitization, habituation


def compareDelays(pulse1=5, pulse2=5, delay=[.5,1,3,5,30],
                  pop_measure=[totalResponse_pop,],n_boot=1e2,conf_interval=99,
                 plot_comparison=False,measure_compare='diff'):#responseDuration,peakResponse
    
    fig,ax = plt.subplots(nrows=len(pop_measure))
    width=.2 #width of violin plots
    if len(pop_measure)==1:
        ax = [ax]
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)
    #load data
    name = 'data/LDS_response_pairedPulse_uv-uv.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_uvRange.pickle'
    ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)
    
    #solve for the reference
    if pulse2==5:
        ref = 'WT'
    else:
        ref = f'WT_{pulse2}s'
    xp_ref = result_ref['tau']
    yp_ref = result_ref[ref]
    #calculate reference population statistics
    for n_m, M in enumerate(pop_measure):
        loc=np.argmin(xp_ref**2)
        bott = 0
        if measure_compare in DIFFERENCE:
            y,rng,significant = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
        elif measure_compare in RELATIVE:
            bott = 1
            y,rng,significant = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
        else:
            y,rng = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)
        x_loc = len(delay)
        if not plot_comparison:
            y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                   conf_interval=conf_interval, return_samples=True)
            v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
            ax[n_m].scatter([x_loc],[y-bott],color='grey')
            for pc in v['bodies']:
                pc.set_facecolor('grey')
            ax[n_m].plot([-.5,x_loc+.5],[y,y],c='grey',ls=':')
        else:
            ax[n_m].bar([x_loc],[y-bott],color='grey',width=width,bottom=[bott],alpha=.2)
            ax[n_m].plot([x_loc,x_loc],rng,color='grey')
        
    
    
    c='cornflowerblue'
    #loop through conditions    
    for i,d in enumerate(delay):
        test = f'WT_{pulse1}s{pulse2}s_{d}mDelay'
        if d<1:
            test = f'WT_{pulse1}s{pulse2}s_{int(d*60)}sDelay'
        xp = result['tau']
        yp = result[test]['secondary']
        
        #calculate population based measures
        for n_m, M in enumerate(pop_measure):
            loc_ref=np.argmin(xp_ref**2)
            loc=np.argmin(xp**2)
            bott = 0
            if measure_compare in DIFFERENCE:
                y,rng,significant = bootstrap_diff(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            elif measure_compare in RELATIVE:
                bott=1
                y,rng,significant = bootstrap_relative(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            else:
                y,rng = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)
                significant=False
            x_loc= i
            if not plot_comparison:
                y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                       conf_interval=conf_interval,return_samples=True)
                v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],
                                            showmeans=False,showextrema=False)
                ax[n_m].scatter([x_loc],[y],color=c)
                for pc in v['bodies']:
                    pc.set_facecolor(c)

            else:    
                ax[n_m].bar([x_loc],[y-bott],color=c,width=width,bottom=[bott],alpha=.2)
                ax[n_m].plot([x_loc,x_loc],rng,color=c)
            if significant:
                mark_sig(ax[n_m],x_loc,c=c,yy=rng[1]*1.1)
    ax[-1].set_xticks(np.arange(len(delay)+1))
    ax[-1].set_xticklabels(delay+[120])
    ax[-1].set_xlabel('Delay (min)')
    for a,M in zip(ax,pop_measure):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(M.__name__)
        a.set_xlim(-.5,len(delay)+.5)
    fig.suptitle(f'{pulse1}s pulse 1, {pulse2}s pulse 2, variable delay')
    return fig,ax

def compareFirstPulse(pulse1=[1,5,10], pulse2=5, delay=3,
                  pop_measure=[totalResponse_pop,],n_boot=1e2,conf_interval=99,
                 plot_comparison=False,measure_compare='diff'):#responseDuration,peakResponse
    
    fig,ax = plt.subplots(nrows=len(pop_measure))
    width=.2 #width of violin plots
    if len(pop_measure)==1:
        ax = [ax]
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)
    #load data
    name = 'data/LDS_response_pairedPulse_uv-uv.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_uvRange.pickle'
    ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)
    
    #solve for the reference
    if pulse2==5:
        ref = 'WT'
    else:
        ref = f'WT_{pulse2}s'
    xp_ref = result_ref['tau']
    yp_ref = result_ref[ref]
    #calculate reference population statistics
    for n_m, M in enumerate(pop_measure):
        loc=np.argmin(xp_ref**2)
        bott = 0
        if measure_compare in DIFFERENCE:
            y,rng,significant = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
        elif measure_compare in RELATIVE:
            bott = 1
            y,rng,significant = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
        else:
            y,rng = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)
        print(pulse1)
        x_loc = 0
        if not plot_comparison:
            y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                   conf_interval=conf_interval, return_samples=True)
            v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
            ax[n_m].scatter([x_loc],[y-bott],color='grey')
            for pc in v['bodies']:
                pc.set_facecolor('grey')
            ax[n_m].plot([-.5,len(pulse1)+.5],[y,y],c='grey',ls=':')
        else:
            ax[n_m].bar([x_loc],[y-bott],color='grey',width=width,bottom=[bott],alpha=.2)
            ax[n_m].plot([x_loc,x_loc],rng,color='grey')
    
    
    c='cornflowerblue'
    #loop through conditions    
    for i,p1 in enumerate(pulse1):
        test = f'WT_{p1}s{pulse2}s_{delay}mDelay'
        if delay<1:
            test = f'WT_{p1}s{pulse2}s_{int(delay*60)}sDelay'
        xp = result['tau']
        yp = result[test]['secondary']
        
        #calculate population based measures
        for n_m, M in enumerate(pop_measure):
            loc_ref=np.argmin(xp_ref**2)
            loc=np.argmin(xp**2)
            bott = 0
            if measure_compare in DIFFERENCE:
                y,rng,significant = bootstrap_diff(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            elif measure_compare in RELATIVE:
                bott=1
                y,rng,significant = bootstrap_relative(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            else:
                y,rng = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)
                significant=False
            x_loc= i+1
            if not plot_comparison:
                y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                       conf_interval=conf_interval,return_samples=True)
                v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],
                                            showmeans=False,showextrema=False)
                ax[n_m].scatter([x_loc],[y],color=c)
                for pc in v['bodies']:
                    pc.set_facecolor(c)

            else:    
                ax[n_m].bar([x_loc],[y-bott],color=c,width=width,bottom=[bott],alpha=.2)
                ax[n_m].plot([x_loc,x_loc],rng,color=c)
            if significant:
                mark_sig(ax[n_m],x_loc,c=c,yy=rng[1]*1.1)
    ax[-1].set_xticks(np.arange(len(pulse1)+1))
    ax[-1].set_xticklabels([0]+pulse1)
    ax[-1].set_xlabel('First pulse (s)')
    for a,M in zip(ax,pop_measure):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(M.__name__)
        a.set_xlim(-.5,len(pulse1)+.5)
    fig.suptitle(f'variable pulse 1, {pulse2}s pulse 2, {delay}m delay')
    return fig,ax
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        