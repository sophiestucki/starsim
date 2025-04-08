"""
Starsim simulation routine

Created on Tuesday 18/02/2025 12.02.2025
by Sophie Stucki (stucki@ieec.cat)

"""

import starsim

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

from functools import partial

def simulation_ini(Q, spot_size_list, latitude_list, longitude_list, conf_file_path, periods_nbr=1):
    """
    Iniialization of the starsim simulation for a particular set of parameters

    Params:
            -Q: facular_area_ratio
    Return:
            -ss: starsim object

    """
    #set timeframe
    t=np.linspace(0,28*periods_nbr,int(28*periods_nbr))
    
    #create the starsim object
    ss=starsim.StarSim(conf_file_path=conf_file_path)

    #set the Q parameter
    ss.facular_area_ratio=Q

    #initialize the spot
    overlap=True

    #TODO: more modulable
    while overlap:
        Nspots=len(spot_size_list)
        ss.spot_map=np.zeros([Nspots,7])
        for j in range(Nspots):
            ss.spot_map[j][1]=100#lifetime spot
            ss.spot_map[j][0]=0#appearance time
            ss.spot_map[j][2]=latitude_list[0]#latitude (degrees) [0,180]
            ss.spot_map[j][3]=longitude_list[0]#longitude (degrees)	[0,360]
            ss.spot_map[j][4]=spot_size_list[j]#spot size (degrees)
        overlap=starsim.nbspectra.check_spot_overlap(ss.spot_map,Q) #true if spots are overlapping
        #checks if the are overlapping spots. If true, find another set of spot parameters

    return ss


def update(frame, ax, fig, lags, scalar_map, t, rv_sim, CCF_p, RV, BIS, lc, path_grid):
    ax[0].cla()
    ax[1].cla()
    lag = lags[frame]
    
    #plot RVs
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[1].plot(t[i],rv_sim[i], color=color_val,marker='.')
    ax[1].set_xlim(t.min(),t.max())
    ax[1].set_ylim(rv_sim.min()-np.abs(rv_sim.min())/10,rv_sim.max()+np.abs(rv_sim.max())/10)
    ax[1].plot(t[:lag],rv_sim[:lag],color='gray')
    
    #plot CCF
    color_val = scalar_map.to_rgba(rv_sim[lag])
    ax[0].plot(RV[:],CCF_p[lag], color=color_val,alpha=1)
    for i in range(lag):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[0].plot(RV[:],CCF_p[i], color=color_val,alpha=0.1)


    #plot BIS
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(BIS[i])
        ax[4].plot(rv_sim[i],BIS[i], color=color_val,marker='.', linestyle='None')
    ax[4].set_xlim(rv_sim.min(),rv_sim.max())
    ax[4].plot(rv_sim[:lag],BIS[:lag],color='gray')


    #plot flux intensity
    for i in range(len(t)):
        color_val = scalar_map.to_rgba(rv_sim[i])
        ax[3].plot(t[i],lc[i], color=color_val,marker='.')
    ax[3].set_xlim(t.min(),t.max())
    ax[3].plot(t[:lag],lc[:lag],color='gray')
        
    #generate stellar grid
    x=np.linspace(-0.999,0.999,1000)
    h=np.sqrt((1-x**2)/(np.tan(0)**2+1))
    color_dict = { 0:'red', 1:'black', 2:'yellow', 3:'blue'}
    #0: photosphere, 1: spot, 2: faculae, 3: planet
    
    vec_grid=np.load(path_grid+'vec_gridt{:.4f}.npy'.format(t[lag]))
    typ=np.load(path_grid+'typt{:.4f}.npy'.format(t[lag]))
    
    #identifies which type of grid element is covering the pixel: photosphere, spot, facualae or planet
    ax[2].scatter(vec_grid[:,1],vec_grid[:,2], color=[ color_dict[np.argmax(i)] for i in typ ],s=50)
    ax[2].plot(x,h,'k')
    
    ax[0].set_ylabel('CCF power')
    ax[0].set_xlabel('RV (m/s)')
    ax[0].set_ylim(CCF_p.min()-np.abs(CCF_p.min())/10,CCF_p.max()+np.abs(CCF_p.max())/10)
    ax[1].set_ylabel('RV (m/s)')
    ax[1].set_xlabel('t (d)')
    ax[3].set_ylabel('f')
    ax[3].set_xlabel('t (d)')
    ax[4].set_ylabel('BIS diff.S')
    ax[4].set_xlabel('RV (m/s)')
    fig.tight_layout()




def retrieve_observable(ss, t, pathdata, path_grid):
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
    '''
    Generate an animated GIF
    '''

    CCF_p =CCF[:]-np.mean(CCF,axis=0)

    lags = np.arange(len(t))

    # Define the colors
    colors  = ["dodgerblue","r"]  # Red -> Black -> Blue

    # Create a colormap from the list of colors
    colormap = mcolors.LinearSegmentedColormap.from_list("BlueRed", colors)

    # Normalize the continuous variable to map it to the range [0, 1] for the colormap
    norm = Normalize(vmin=np.min(rv_sim), vmax=np.max(rv_sim))
    scalar_map = ScalarMappable(norm=norm, cmap=colormap)

    fig, ax = plt.subplots(ncols=5,figsize=(50/1.5, 10/1.5))

    # Create an animation
    ani = FuncAnimation(fig, partial(update, ax=ax, fig=fig, lags=lags, scalar_map=scalar_map, t=t, rv_sim=rv_sim, CCF_p=CCF_p, RV=RV, BIS=BIS, lc=lc, path_grid=path_grid), frames=len(lags), repeat=False)

    # Save the animation as a video
    writer = FFMpegWriter(fps=3, metadata=dict(artist='Me'), bitrate=2400)


    
    N_spots = np.shape(ss.spot_map)[0]
    if N_spots == 1:
        ani.save(pathdata+"Sun_demo_q_{}_spot_size_{}_lat_{}.gif".format(ss.facular_area_ratio, ss.spot_map[0][4], ss.spot_map[0][2]), writer=writer)
    else:
        ani.save(pathdata+"Sun_demo_q_{}_N_spots_{}.gif".format(ss.facular_area_ratio, N_spots), writer=writer)


def simulation_routine(Q, spot_size_list, latitude_list, longitude_list, periods_nbr=1,conf_file_path= None,pathdata=None, path_grid=None):
    """
    Run the whole simulation and asve the observables #TODO
    """

    ss = simulation_ini(Q, spot_size_list, latitude_list, longitude_list, conf_file_path, periods_nbr)
    
    t=np.linspace(0,28*periods_nbr,28*periods_nbr)

    ss.compute_forward(observables=['rv','lc'],t=t)

    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    plt.ioff()

    retrieve_observable(ss, t, pathdata, path_grid)