import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors


from functools import partial
import starsim


### Grid + photometry

plt.rcParams['font.size'] = 18

def update(frame, ax, fig, lags, t, lc, path_grid, vec_grids, types, color_dict):
    lag = lags[frame]

    # Plot flux intensity (only update the necessary parts)
    ax[1].clear()  # Clear previous plot to avoid overlapping
    ax[1].plot(t[:lag], lc[:lag], color='gray', alpha=0.2, linewidth=1)
    ax[1].plot(t, lc, color='#9B2226', marker='.', markersize=12, linewidth=0)


    ax[1].set_xlim(t.min(), t.max())
    ax[1].set_ylabel('relative flux intensity')
    ax[1].set_xlabel('t [day]')
    ax[1].ticklabel_format(axis='both', style='sci')

    # Scatter plot for the grid (precompute color and data)
    ax[0].clear()  # Clear previous scatter plot to avoid overlapping
    ax[0].scatter(vec_grids[lag][:, 1], vec_grids[lag][:, 2], 
                  color=[ color_dict[np.argmax(i)] for i in types[lag]], s=50, alpha=0.8)

    ax[0].locator_params(axis='y', nbins=5)
    ax[0].tick_params(grid_alpha=0.7)
    ax[1].tick_params(grid_alpha=0.7)

    fig.tight_layout()

def load_data(path_grid, t, lags):
    # Preload vec_grid and typ for each frame to avoid repeated loading
    vec_grids = []
    types = []
    for lag in lags:
        vec_grid = np.load(path_grid + 'vec_gridt{:.4f}.npy'.format(t[lag]))
        typ = np.load(path_grid + 'typt{:.4f}.npy'.format(t[lag]))
        vec_grids.append(vec_grid)
        types.append(typ)
    return vec_grids, types

# Initialize parameters
lc = ss.results['lc']
lags = np.arange(len(t))
color_dict = {0: '#CA6702', 1: '#9B2226', 2: '#EE9B00', 3: '#0A9396'}

# Preload data
vec_grids, types = load_data(path_grid, t, lags)

# Set up the figure and axis
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20 / 1.5 + 2, 10 / 1.5))

# Create the animation
ani = FuncAnimation(fig, partial(update, ax=ax, fig=fig, lags=lags, t=t, lc=lc, 
                                 path_grid=path_grid, vec_grids=vec_grids, types=types, 
                                 color_dict=color_dict), 
                    frames=len(lags), repeat=False)
output = '/home/sophie-stucki/Documents/starsim_simulations/test_distinct_q_dav/grid_Q_{}_r_{}_lat_{}_long_{}.gif'.format(Q_list, spot_size_list, latitude_list, longitude_list)

# Save the animation
writer = FFMpegWriter(fps=3, metadata=dict(artist='Me'), bitrate=2400)
ani.save(output, writer=writer)




### Grid + photometry + RV + bis

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