import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces
from .bootstrapTest import bootstrapTest

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

def rnai_response(interest,exclude,n_boot=1e3,statistic=np.median,regeneration=False):
    if regeneration:
        return rnai_response_regen(interest,exclude,n_boot,statistic)
    name = 'data/LDS_response_rnai.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = data_of_interest(result.keys(),interest,exclude)
    
    tot=len(to_plot)
    fig, ax=plt.subplots(nrows=tot, sharex=True, sharey=True,figsize=(8,4*len(to_plot)))
    if tot==1:
        ax=[ax]
    num=0
    for i in to_plot:
        #plot control
        if i[-3:]=='30s':
            yp=result['WT_30s']
            xp=result['tau']-.5
        elif ('_1s' in i):# or i=='oct 20mM':
            print(i)
            yp=result['WT_1s']
            xp=result['tau']-5/60
        else:
            yp=result['WT']
            xp=result['tau']-5/60
        y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
        ax[num].plot(xp,y,lw=1,color='grey',zorder=-2,alpha=.6)
        ax[num].fill_between(xp,*rng,alpha=.2,color='grey',lw=0,edgecolor='None', zorder=-1)
        ax[num].set_ylim(0,2)
        ax[num].set_yticks([0,1,2])
        ax[num].plot([0,0],[0,5],c='k',ls=':')
        yp=result[i]
        y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
        ax[num].plot(xp,y,label=i,lw=1,color='r',)
        ax[num].fill_between(xp,*rng,alpha=.1,color='r',lw=0,edgecolor='None')
        ax[num].legend()
        num+=1
    plt.xlabel('time (min)')
    ax[len(ax)//2].set_ylabel('Z')
    plt.xlim(-3,10)
    return fig, ax

def rnai_response_regen(interest,exclude,n_boot=1e3,statistic=np.median):
    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    data = data_of_interest(result.keys(),interest,exclude)
    ref = []
    for d in data:
        if '30s' in d:
            ref.append('standard_30s2h')
        else:
            ref.append('standard_5s2h')
    day_shift = [0 for d in data]
    fig, ax = plt.subplots(ncols=len(data),nrows=len(result[(data[0])]),sharex=True,sharey=True,figsize=(8,12))
    if len(data)==1:
        ax=ax[:,None]
    if len (ref)==1:
        ref = [ref[0] for r in data]

    for i in range(len(data)):  
        ax[0,i].set_title(data[i])
        for j in range(len (result[(data[i])])):
            xp = result['tau']
            yp=(result[data[i]][j])
            if yp.size==0:
                continue
            y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
            ax[j+day_shift[i],i].plot(xp,y,label=j, color=plt.cm.cool(j/7))
            ax[j+day_shift[i],i].fill_between(xp,*rng,alpha=.3,edgecolor=None,facecolor=plt.cm.cool(j/7))
        for j in range(len(result[ref[i]])):   
            yp=(result[ref[i]][j])
            y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
            ax[j,i].plot(xp,y,color='grey',zorder=-1)
            ax[j,i].fill_between(xp,*rng,alpha=.3,edgecolor=None,facecolor='grey',zorder=-2)

            ax[j,i].spines['top'].set_visible(False)
            ax[j,i].spines['right'].set_visible(False)
            if i>0:
                ax[j,i].spines['left'].set_visible(False)
    fig.suptitle('regen control')    
    #    ax[j].legend()
    plt.xlim(-3,20)
    plt.ylim(-.1,2)
    #plt.yscale('log')    
    ax[len(ax)//2,0].set_ylabel('Z')
    ax[-1,ax.shape[1]//2].set_xlabel('time (min)')

    for a in ax:
        a[0].set_yticks([0,1,2])
    return fig,ax

def total_response_regeneration(interest,exclude,n_boot=1e3,statistic=np.median,
                               integrate=10*120,pool=12,color_scheme = None):
    name = 'data/LDS_response_regen_indPulse.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = ['WT_8hpa_30s2h','WT_8hpa_10s2h','WT_8hpa_5s2h','WT_8hpa_1s2h']
    to_plot.extend(data_of_interest(result.keys(),interest,exclude))

    reference = []
    if reference:
        ref_name = 'data/LDS_response_uvRange.pickle'
        with open(ref_name,'rb') as f:
            result_ref = pickle.load()
    max_t = 200
    isi = integrate
    integrate = isi
    pulse_list = [0,]
    sh = []
    
    n_pulse=len(pulse_list)
    fig, ax = plt.subplots(nrows=n_pulse,sharex=True,figsize=(12,8))
    if n_pulse==1:
        ax=[ax]
    for ii,cond in enumerate(to_plot):
        print(cond)
        for n_pulse, pulse in enumerate(pulse_list):
            if reference:
                response = result_ref[reference[ii]]
                tau = result_ref['tau'].copy()
                if sh:
                    tau -= sh[ii]
                data = bootstrapTest('reference', [response,], tau=tau,
                                      n_pulse=max(pulse_list)+1, isi=isi,
                                      integrate_time=integrate,experiment_list=None)
                y,rng = data.canonical_estimate([0,pulse],sample_size=None, 
                                                statistic=statistic, n_boot=n_boot)
                y_ref = y.copy()
                if not subtract_ref:
                    plt.plot(t_plot,np.ones_like(t_plot)*y,ls='--',zorder=-1,alpha=.5,c=c)
                    # plt.fill_between(t_plot, np.ones_like(t_plot)*rng[0], np.ones_like(t_plot)*rng[1],
                    #                  facecolor="grey",edgecolor=c,hatch='/////',lw=0,zorder=-1)
                    plt.fill_between(t_plot, np.ones_like(t_plot)*rng[0], np.ones_like(t_plot)*rng[1],
                                      facecolor="grey",alpha=.1,zorder=-1)
                    plt.fill_between(t_plot, np.ones_like(t_plot)*rng[0], np.ones_like(t_plot)*rng[1],
                                      facecolor=c,alpha=.1,zorder=-1)
            y = []
            lo = []
            hi = []
            t_plot = []
            response = result[cond]['data']
            response_pool = [np.concatenate(response[i:i+pool]) for i in range(len(response))]
            tau = result['tau'].copy()
            if sh:
                tau -= sh[ii]
            data = bootstrapTest('data', response_pool, tau=tau,
                                  n_pulse=max(pulse_list)+1, isi=isi,
                                  integrate_time=integrate,experiment_list=None)
            for t in range(min(max_t,len(response))):
                y_t, rng = data.canonical_estimate([t,pulse],sample_size=None,
                                                   statistic=statistic,n_boot=n_boot)
                y.append(y_t)
                lo.append(rng[0])
                hi.append(rng[1])
                t_plot.append(t)
            t_plot = np.array(t_plot) * result[cond]['spacing'] + result[cond]['initial'] 
            if color_scheme is None:
                if 'WT' in cond and not 'vibe' in cond:
                    x_st = cond.find('hpa_')
                    x_en = cond[x_st:].find('s')
                    c = plt.cm.gray_r((float(cond[x_st+4:x_st+x_en])+5)/40)

                else:
                    c=plt.cm.rainbow((ii)/10)
            else:
                c =color_scheme(1-((ii+1)/len(to_plot)))

#             if reference and subtract_ref:
#                 y -= y_ref
#                 lo -= y_ref
#                 hi -= y_ref
#                 # y = np.log10(y/y_ref)
#                 # lo = np.log10(lo/y_ref)
#                 # hi = np.log10(hi/y_ref)

            ax[n_pulse].plot(t_plot,y,color=c)
            ax[n_pulse].scatter(t_plot,y,color=c,label=f'{cond}, n={data.canonical_set_traces[0].shape[0]}')
            ax[n_pulse].fill_between(t_plot,lo,hi,facecolor=c,alpha=.2)

#     if subtract_ref:
#         plt.plot(t_plot,0*t_plot,c='k',alpha=.5,ls='-.' )
    plt.legend()   
    plt.xlabel('hours') 
    plt.ylabel(f'integrated activity 0-{isi/120} min')  
    # plt.ylabel(f'log10 [integrated activity 0-{isi/120} min] / [wholeworm value]')    
    plt.xlim(0,175)
    return fig, ax
