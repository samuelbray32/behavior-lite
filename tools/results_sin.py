import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap, timeDependentDifference
from .bootstrapTest import bootstrap_diff, bootstrap_relative

def data_of_interest(names,interest=[],exclude=[]):
    to_plot = []
    full_amp = True
    for i in interest:
        if 'amp' in i and not '127amp' in i:
            full_amp=False
    if full_amp:
        exclude.append('amp')
    for dat in names:
        if dat in to_plot: continue
        for i in interest:
            if i in dat:
                keep = True
                for ex in exclude:
                    if ex in dat: keep = False
                #check double/triple knockdown
                if dat.count('+')>i.count('+'):
                    print(dat)
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot

def calc_delta(x,w=2):
    if w%2==0:
        w+=1
    filt = np.ones(w)
    filt[:w//2]=-1
    filt[w//2]=0
    return np.convolve(x,filt,mode='same')/w

def response_sin(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,derivative=False,
                          visible=False):

    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey'):
      ax.scatter([loc,],12,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    fig, ax_all=plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True,figsize=(10,2*len(periods)))
    if len(periods)==1: ax_all = ax_all[None,:]
    if True: ax_all=ax_all[:,None]
    ax = ax_all

    for ii,interest in enumerate(interest_list):
        if ii==0:
            #step function reference
            name = 'data/LDS_response_LONG.pickle'
            with open(name,'rb') as f:
                result = pickle.load(f)
            xp = result['tau']
            ind_t = np.where((xp>=-3)&(xp<=40))[0]
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m2h{128}bp'],)
            if len(to_plot)==0:
                yp_ref=np.zeros((1,xp.size))
            else:
                yp_ref = np.concatenate([result[dat] for dat in to_plot])
                loc = np.argmin(xp**2)
                y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                for a in ax[:,0]:
                    c='grey'
                    a.plot(xp[ind_t],y,lw=1,color=c,zorder=-2,)
                    a.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            ind_t_ref = ind_t.copy()

        #loop the sin periods
        name = 'data/LDS_response_sinFunc.pickle'
        if visible:
            name = 'data/LDS_response_Vis_sinFunc.pickle'
        with open(name,'rb') as f:
            result = pickle.load(f)
        # print(result.keys())
        for i,p in enumerate(periods):
            xp = result['tau']
            ind_t = np.where((xp>=-3)&(xp<=40))[0]
            p_name = f'{p}m'
            if p<1 or p%1>0:
                p_name = f'{int(p*60)}s'

            print(f'{interest}_{duration}m_{p_name}')
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],)
            if len(to_plot)==0:
                continue
            print(to_plot)
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            # if derivative:
            #     print(yp[0].shape)
            #     yp = np.array([calc_delta(yy,derivative) for yy in yp])
            #     print(yp.shape)

            loc = np.argmin(xp**2)
            y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            if derivative:
                True #not interesting results :(
                # y = np.array([calc_delta(yy,derivative) for yy in y])
                # rng = np.percentile
                # def delta_meas(x,axis=0):
                #     xx = np.median(x,axis=0)
                #     return calc_delta(xx,derivative)
                # y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=delta_meas,conf_interval=conf_interval)

            c=plt.cm.Set1(ii/9)#'cornflowerblue'
            ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, ({yp.shape[0]})',)
            ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            ax[i,0].plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
            time_sig = timeDependentDifference(yp[:,ind_t],yp_ref[:,ind_t_ref],n_boot=n_boot,conf_interval=conf_interval)
            x_sig = xp[ind_t]#np.arange(0,10,1/120)
            y_sig = .1
        #                         if len(powers)-1>1:
        #                             y_sig=2*y_sig/len(interest_list)
            j=0
            bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
            ax[i,0].fill_between(x_sig,bott,bott-y_sig*time_sig,
                facecolor=c,alpha=.4)
            box_keys={'lw':1, 'c':'k'}
            ax[i,0].plot(x_sig,bott,**box_keys)
            ax[i,0].plot(x_sig,bott-y_sig,**box_keys)
            # ax[i,0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
            # ax[i,0].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
            # ax[i,0].set_title(f'{dur} min')
            ax[i,0].set_xlim(-3,40)
            ax[i,0].set_ylim(0,1.6)
            ax[i,0].legend()
    ax[-1,0].set_xlabel('time (min)')
    return fig,ax

def response_sin_amp(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,t_samp=(20,30)):


    #loop the sin periods
    if type(interest_list) is str:
        interest_list = [interest_list,]
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    for ii,interest in enumerate(interest_list):
        c = plt.cm.Set1(ii/9)
        if interest is 'WT':
            c='grey'
        for i,p in enumerate(periods):
            if len(interest_list)==1:
                c = plt.cm.magma(i/len(periods))
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
            p_name = f'{p}m'
            if p<1 or p%1>0:
              p_name = f'{int(p*60)}s'
            print(f'{interest}_{duration}m_{p_name}')
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],)
            if len(to_plot)==0:
                continue
            print(to_plot)
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            tau = np.arange(1,int(p*120))
            # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
            y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,U=result[to_plot[0]]['stim'][ind_t],tau=tau)
            print(np.array(y).shape,np.array(dist).shape)
            xx=np.linspace(0,2,np.array(y).size)
            ax[0].plot(xx,y,c=c)
            ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

            #amplitudes
            amp = np.nanmax(dist,axis=1)
            ax[1].scatter(p,amp.mean(),color=c,label=interest)
            v = ax[1].violinplot([amp],positions=[p],vert=True,widths=[.5],showextrema=False,)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)
            #phases
            amp = xx[np.nanargmax(dist,axis=1)]
            #rotate if near phi=0 boundary
            # if amp.mean()<.5 or amp.mean()>1.5:
            #     ind_rot = np.where(amp>1)
            #     amp[ind_rot] = amp[ind_rot]-2
            amp_alt = amp.copy()
            ind_rot = np.where(amp>1)
            amp_alt[ind_rot] = amp[ind_rot]-2
            if np.var(amp_alt)<np.var(amp):
                amp=amp_alt.copy()
            label=None
            if i==0:
                label=interest
            ax[2].scatter(p,amp.mean(),color=c,label=label)
            v = ax[2].violinplot([amp],positions=[p],vert=True,widths=[.5],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    ax[1].set_xlabel('period (min)')
    ax[1].set_ylabel('amplitude')
    ax[2].set_xlabel('period (min)')
    ax[2].set_ylabel('phase shift ($\pi$ rad)')
    ax[2].legend()
    return fig, ax

def response_sin_amp_vDriveamp(interest,exclude,periods =[2,3],drive_amp=[32,64,127],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,):


    #loop the sin periods
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)

    for i,p in enumerate(periods):
        c = plt.cm.Set1(i/9)
        for j,amp_j in enumerate(drive_amp):

            xp = result['tau']
            ind_t = np.where((xp>=20)&(xp<=30))[0]
            p_name = f'{p}m'
            if p<1:
              p_name = f'{int(p*60)}s'
            # print(f'{interest}_{duration}m_{p_name}')
            if amp_j<127:
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}Period_amp{amp_j}bp'],)
            else:
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}Period'],['amp'])
            print(f'{interest}_{duration}m_{p_name}Period_amp{amp_j}bp',to_plot)
            if len(to_plot)==0: continue
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            tau = np.arange(1,int(p*120))
            # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
            u=result[to_plot[0]]['stim'][ind_t]
            u -= u.min()#normalize the oscillations to the same range
            u /= u.max()
            y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,U=u,tau=tau)
            print(np.array(y).shape,np.array(dist).shape)
            xx=np.linspace(0,2,np.array(y).size)
            ax[0].plot(xx,y,c=c)
            ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

            #amplitudes
            plot_x = j + i/len(drive_amp)/2
            plot_w = 1/len(drive_amp)/2
            amp = np.max(dist,axis=1)
            label=None
            if j==0:
                label=p_name
            ax[1].scatter(plot_x,amp.nanmean(),color=c,label=label)
            v = ax[1].violinplot([amp],positions=[plot_x],vert=True,widths=[plot_w],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)
            #phases
            amp = xx[np.argmax(dist,axis=1)]
            ax[2].scatter(plot_x,amp.mean(),color=c,label=label)
            v = ax[2].violinplot([amp],positions=[plot_x],vert=True,widths=[plot_w],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    for a in ax[1:]:
        a.set_xlabel('drive amplitude')
        a.set_xticks(np.arange(len(drive_amp)))
        a.set_xticklabels(drive_amp)
        a.legend(loc=2)
    ax[1].set_ylabel('response amplitude')
    ax[2].set_ylabel('phase shift ($\pi$ rad)')
    return fig, ax


def response_sin_autocov(interest,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,tau=np.arange(1,120*10,10),statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,):


    #loop the sin periods
    fig,ax = plt.subplots(ncols=1,figsize=(6,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    for i,p in enumerate(periods):
        c = plt.cm.magma(i/len(periods))
        xp = result['tau']
        ind_t = np.where((xp>=15)&(xp<=30))[0]
        p_name = f'{p}m'
        if p<1:
          p_name = f'{int(p*60)}s'
        print(f'{interest}_{duration}m_{p_name}')
        to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],)
        print(to_plot)
        xp = result['tau']
        yp = np.concatenate([result[dat]['data'] for dat in to_plot])
        # tau = np.arange(1,int(p*120))
        # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
        from tools.measurements import cross_correlate_auto
        C = np.array([cross_correlate_auto(yy,tau) for yy in yp[:,ind_t]])
        def med(x):
            return np.mean(x,axis=0)
        y,rng,dist=bootstrap(C,statistic=med,conf_interval=conf_interval,n_boot=n_boot,
            return_samples=True,)
        print(np.array(y).shape,np.array(dist).shape)
        xx=tau/120
        ax.plot(xx,y,c=c,label=p)
        ax.fill_between(xx,*rng,facecolor=c,alpha=.2)


    ax.legend()
    # ax[1].set_xlabel('period (min)')
    # ax[1].set_ylabel('amplitude')
    return fig, ax
