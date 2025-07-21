"""
Starsim simulation routine

Created on Tuesday 18/02/2025 12.02.2025
by Sophie Stucki (stucki@ieec.cat)

"""
import sys
sys.path.insert(1, '/home/sophie-stucki/starsim_david/')

import starsim

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

from functools import partial

from generate_colormap import *

def simulation_init(Q, spot_size_list, latitude_list, longitude_list, conf_file_path, periods_nbr=1, point_nbr=1, dT_fc=None):
    """
    Iniialization of the starsim simulation for a particular set of parameters

    Params:
            -Q: facular_area_ratio
    Return:
            - ss: starsim object

    """
    
    #create the starsim object
    ss=starsim.StarSim(conf_file_path=conf_file_path)

    #set timeframe
    t=np.linspace(0,ss.rotation_period*periods_nbr,int(ss.rotation_period*periods_nbr*point_nbr))

    #set the Q parameter
    ss.facular_area_ratio=Q

    if dT_fc !=None:
        ss.facula_T_contrast = dT_fc

    #initialize the spot
    overlap=True

    #TODO: more modulable
    while overlap:
        Nspots=len(spot_size_list)
        ss.spot_map=np.zeros([Nspots,7])
        for j in range(Nspots):
            ss.spot_map[j][1]=200#lifetime spot
            ss.spot_map[j][0]=0#appearance time
            ss.spot_map[j][2]=latitude_list[j]#latitude (degrees) [0,180]
            ss.spot_map[j][3]=longitude_list[j]#longitude (degrees)	[0,360]
            ss.spot_map[j][4]=spot_size_list[j]#spot size (degrees)
        overlap=starsim.nbspectra.check_spot_overlap(ss.spot_map,Q) #true if spots are overlapping
        if overlap:
            print("ERROR: Spots are overlapping")
        #checks if the are overlapping spots. If true, find another set of spot parameters

    return ss


def update(frame, ax, fig, lags, scalar_map, t, rv_sim, CCF_p, RV, BIS, raw_xbis, raw_ybis, lc, path_grid):
    ax[1,0].cla()
    ax[1,1].cla()
    lag = lags[frame]

    color_val = scalar_map.to_rgba(rv_sim[lag])
    
    #plot RVs
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[0,2].plot(t[i],rv_sim[i], color=color_val,marker='.', markersize=12)
    ax[0,2].set_xlim(t.min(),t.max())
    ax[0,2].set_ylim(rv_sim.min()-np.abs(rv_sim.min())/10,rv_sim.max()+np.abs(rv_sim.max())/10)
    ax[0,2].plot(t[:lag],rv_sim[:lag],color='gray', alpha=0.2, linewidth= 1)
    
    #plot CCF
    ax[1,0].plot(RV[:],CCF_p[lag], color=color_val,alpha=0.9)
    for i in range(lag):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[1,0].plot(RV[:],CCF_p[i], color=color_val,alpha=0.1)


    #plot BIS (mean diff)
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[1,2].plot(t[i],BIS[i], color=color_val,marker='.', linestyle='None', markersize=12)
    ax[1,2].plot(t[:lag],BIS[:lag],color='gray', alpha=0.2, linewidth= 1)

    #plot BIS
    ax[1,1].plot(raw_xbis[lag] - (np.max(raw_xbis[lag]) + np.min(raw_xbis[lag])) / 2,raw_ybis[lag], color=color_val,alpha=1)
    for i in range(lag):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[1,1].plot(raw_xbis[i] - (np.max(raw_xbis[i]) + np.min(raw_xbis[i])) / 2,raw_ybis[i], color=color_val,alpha=0.1)
    ax[1,1].set_xlim(-40, 40)


    #plot flux intensity
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[0,1].plot(t[i],lc[i], color=color_val,marker='.', markersize=12)
    ax[0,1].set_xlim(t.min(),t.max())
    ax[0,1].plot(t[:lag],lc[:lag],color='gray', alpha=0.2, linewidth= 1)
        
    #generate stellar grid
    x=np.linspace(-0.999,0.999,1000)
    h=np.sqrt((1-x**2)/(np.tan(0)**2+1))

    
    color_dict = { 0:'#CA6702', 1:'#9B2226', 2:'#EE9B00', 3:'#0A9396'}
    #0: photosphere, 1: spot, 2: faculae, 3: planet
    
    vec_grid=np.load(path_grid+'vec_gridt{:.4f}.npy'.format(t[lag]))
    typ=np.load(path_grid+'typt{:.4f}.npy'.format(t[lag]))
    
    #identifies which type of grid element is covering the pixel: photosphere, spot, facualae or planet
    ax[0,0].scatter(vec_grid[:,1],vec_grid[:,2], color=[ color_dict[np.argmax(i)] for i in typ ],s=50, alpha=0.8)
    #ax[0,0].plot(x,h,'k')
    
    ax[1,0].set_ylabel('CCF power')
    ax[1,0].set_xlabel('RV (m/s)')
    ax[1,0].set_ylim(CCF_p.min()-np.abs(CCF_p.min())/10,CCF_p.max()+np.abs(CCF_p.max())/10)
    ax[0,2].set_ylabel('RV (m/s)')
    ax[0,2].set_xlabel('t (d)')
    ax[0,1].set_ylabel('f')
    ax[0,1].set_xlabel('t (d)')
    ax[1,2].set_ylabel('BIS span')
    ax[1,2].set_xlabel('t (D)')
    ax[1,1].set_ylabel('')
    ax[1,1].set_xlabel('BIS')

    ax[0,0].locator_params(axis='y',nbins=5)
    ax[1,0].locator_params(axis='x',nbins=5)

    ax[0,1].ticklabel_format(axis='both', style='sci')
    ax[1,0].ticklabel_format(axis='both', style='sci')

    ax[0,0].tick_params(grid_alpha=0.7)
    ax[0,1].tick_params(grid_alpha=0.7)
    ax[0,2].tick_params(grid_alpha=0.7)
    ax[1,0].tick_params(grid_alpha=0.7)
    ax[1,1].tick_params(grid_alpha=0.7)
    ax[1,2].tick_params(grid_alpha=0.7)

    fig.tight_layout()




def retrieve_observable(ss, t, pathdata, path_grid, output):
    """
    Load and save the #TODO

    Params:
            -ss: starsim object
            -t: timeframe
    
    """

    rv_sim=ss.results['rv']
    CCF=ss.results['CCF'][1:]
    RV=ss.results['CCF'][0]
    lc = ss.results['lc']
    BIS = ss.results['bis']
    raw_xbis = ss.results['raw_xbis']
    raw_ybis = ss.results['raw_ybis']


    '''
    Generate an animated GIF
    '''

    CCF_p =CCF[:] -np.mean(CCF,axis=0)

    lags = np.arange(len(t))


    # Normalize the continuous variable to map it to the range [0, 1] for the colormap
    norm = Normalize(vmin=-np.max((np.abs(rv_sim))), vmax=np.max((np.abs(rv_sim))))
    List = ['#104f5c', '#2c6570', '#477b83', '#639295', '#84a8a5', '#c7b7a0', '#e49a77', '#e5845f', '#de714d', '#d3603e', '#c65031']

    colormap = get_continuous_cmap(List)
    scalar_map = ScalarMappable(norm=norm, cmap=colormap)

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(30/1.5 + 2, 20/1.5))

    # Create an animation
    ani = FuncAnimation(fig, partial(update, ax=ax, fig=fig, lags=lags, scalar_map=scalar_map, t=t, rv_sim=rv_sim, CCF_p=CCF_p, RV=RV, BIS=BIS, raw_xbis=raw_xbis, raw_ybis=raw_ybis, lc=lc, path_grid=path_grid), frames=len(lags), repeat=False)

    # Save the animation as a video
    writer = FFMpegWriter(fps=3, metadata=dict(artist='Me'), bitrate=2400)

    if output == None:
        N_spots = np.shape(ss.spot_map)[0]
        if N_spots == 1:
            ani.save(pathdata+"Sun_demo_q_{}_spot_size_{}_lat_{}.gif".format(ss.facular_area_ratio, ss.spot_map[0][4], ss.spot_map[0][2]), writer=writer)
        else:
            ani.save(pathdata+"Sun_demo_q_{}_N_spots_{}.gif".format(ss.facular_area_ratio, N_spots), writer=writer)
    else:
        ani.save(output, writer=writer)

def simulation_routine(Q, spot_size_list, latitude_list, longitude_list, periods_nbr=1, point_nbr=1,conf_file_path= None,pathdata=None, path_grid=None, output=None, w='eq_2', dT_fc=None, crx=None):
    """
    Run the whole simulation and asve the observables #TODO
    """

    ss = simulation_init(Q, spot_size_list, latitude_list, longitude_list, conf_file_path, periods_nbr, point_nbr, dT_fc = dT_fc)
    
    t=np.linspace(0,ss.rotation_period *periods_nbr,int(ss.rotation_period *periods_nbr*point_nbr))

    if crx:
        ss.compute_forward(observables=['rv','lc', 'crx'],t=t, w=w)
    else:
        ss.compute_forward(observables=['rv','lc'],t=t, w=w)

    # plt.rcParams["animation.html"] = "jshtml"
    # plt.rcParams['figure.dpi'] = 150  
    # plt.rcParams['font.size'] = 24
    # plt.rcParams["axes.formatter.useoffset"]=False
    # plt.ioff()

    # # print('Filling factor spot:', ss.results['ff_sp'])
    # # print('Filling factor faculae:', ss.results['ff_fc'])
    # retrieve_observable(ss, t, pathdata, path_grid, output)

    return ss


def update_diff(frame, ax, fig, lags, scalar_map, t, diff_rv_sim, diff_CCF, RV, diff_BIS, diff_lc, path_grid):
    ax[1,0].cla()
    ax[1,1].cla()
    lag = lags[frame]

    color_val = scalar_map.to_rgba(diff_rv_sim[lag])
    
    #plot RVs
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(diff_rv_sim[i])
        ax[0,2].plot(t[i],diff_rv_sim[i], color=color_val,marker='.', markersize=12)
    ax[0,2].set_xlim(t.min(),t.max())
    ax[0,2].set_ylim(diff_rv_sim.min()-np.abs(diff_rv_sim.min())/10,diff_rv_sim.max()+np.abs(diff_rv_sim.max())/10)
    ax[0,2].plot(t[:lag],diff_rv_sim[:lag],color='gray', alpha=0.4, linewidth= 2)
    
    #plot CCF
    ax[1,0].plot(RV[:],diff_CCF[lag], color=color_val,alpha=0.9)
    for i in range(lag):
        color_val = scalar_map.to_rgba(diff_rv_sim[i])
        ax[1,0].plot(RV[:],diff_CCF[i], color=color_val,alpha=0.1)


    #plot BIS (mean diff)
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(diff_rv_sim[i])
        ax[1,2].plot(diff_rv_sim[i],diff_BIS[i], color=color_val,marker='.', linestyle='None', markersize=12)
    ax[1,2].set_xlim(diff_rv_sim.min(),diff_rv_sim.max())
    ax[1,2].plot(diff_rv_sim[:lag],diff_BIS[:lag],color='gray', alpha=0.4, linewidth= 2)



    #plot flux intensity
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(diff_rv_sim[i])
        ax[0,1].plot(t[i],diff_lc[i], color=color_val,marker='.', markersize=12)
    ax[0,1].set_xlim(t.min(),t.max())
    ax[0,1].plot(t[:lag],diff_lc[:lag],color='gray', alpha=0.4, linewidth= 2)
        
    
    
    ax[1,0].set_ylabel('diff CCF power')
    ax[1,0].set_xlabel('RV (m/s)')
    ax[0,2].set_ylabel('diff RV (m/s)')
    ax[0,2].set_xlabel('t (d)')
    ax[0,1].set_ylabel('diff f')
    ax[0,1].set_xlabel('t (d)')
    ax[1,2].set_ylabel('diff BIS indicator')
    ax[1,2].set_xlabel('RV (m/s)')


    ax[0,0].tick_params(grid_alpha=0.7)
    ax[0,1].tick_params(grid_alpha=0.7)
    ax[0,2].tick_params(grid_alpha=0.7)
    ax[1,0].tick_params(grid_alpha=0.7)
    ax[1,1].tick_params(grid_alpha=0.7)
    ax[1,2].tick_params(grid_alpha=0.7)

    fig.tight_layout()


def compare_eqs(Q, spot_size_list, latitude_list, longitude_list, periods_nbr=1, point_nbr=1,conf_file_path= None,pathdata=None, path_grid=None, output=None):
    """ 
    Compare the resulting observables between the 2 differential eqs
    
    """

    ss_1 = simulation_routine(Q, spot_size_list, latitude_list, longitude_list, periods_nbr, point_nbr,conf_file_path,pathdata, path_grid, output, w='eq_1')
    ss_2 = simulation_routine(Q, spot_size_list, latitude_list, longitude_list, periods_nbr, point_nbr,conf_file_path,pathdata, path_grid, output, w='eq_2')


    diff_rv = ss_1.results['rv'] - ss_2.results['rv']
    diff_ccf = ss_1.results['CCF'][1:] - ss_2.results['CCF'][1:]
    RV = ss_1.results['CCF'][0]
    diff_lc = ss_1.results['lc'] - ss_2.results['lc']
    diff_bis = ss_1.results['bis'] - ss_2.results['bis']
    diff_crx = ss_1.results['crx'] - ss_2.results['crx']

    
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(30/1.5 + 4, 20/1.5))

    # Create an animation
    ani = FuncAnimation(fig, partial(update_diff, ax=ax, fig=fig, lags=lags, scalar_map=scalar_map, t=t, diff_rv_sim=diff_rv, diff_CCF=diff_ccf, RV=RV, diff_BIS=diff_bis, diff_lc=diff_lc, path_grid=path_grid), frames=len(lags), repeat=False)

    # Save the animation as a video
    writer = FFMpegWriter(fps=3, metadata=dict(artist='Me'), bitrate=2400)

    ani.save(pathdata+"diff_eqs_q_{}_spot_size_{}_lat_{}.gif".format(ss.facular_area_ratio, ss.spot_map[0][4], ss.spot_map[0][2]), writer=writer)





    