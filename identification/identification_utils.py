"""
Identification of features utils

Created on Wednesday 12.02.2025
by Sophie Stucki (stucki@ieec.cat)

Take SDO images of the LOS magnetogram and the limb-darkening-corrected continuum intensity and return the following features:
- spot
- faculae
- network
- plage

follwing the methodology of A. Sen 2023

"""

import os
from astropy.convolution import convolve_fft, Gaussian2DKernel

import scipy.ndimage as nd
from scipy.spatial.distance import euclidean

# from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from astropy.coordinates import solar_system_ephemeris
from astropy.modeling.functional_models import Disk2D

import sunpy.coordinates  as coord # NOQA
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.coordinates import frames

from sunkit_image.coalignment import mapsequence_coalign_by_match_template as mc_coalign

import numpy as np

import cmasher as cmr


def load_data(start_time='2015-01-01T00:00:00', end_time='2015-01-01T00:11:00', jsoc_email="stucki@ieec.cat", Ic_serie='hmi.Ic_noLimbDark_720s', M_serie='hmi.M_45s', path=None):
    """
    load the data from jsoc, fits are download in a sunpy file

    Params:
            - start_time: start of the observations
            - end_time: end of the observations
            - jsoc_email: mail to have access to the data
            - Ic_serie: jsoc name of the continuum data
            - M_serie: jsoc name of the magnetogram data
            - path: directory to retrieve the files into

    Return:
            - cont_sequence: (sunpy) map sequence of continuum data
            - los_sequence:  (sunpy) map sequence of magnetogram data
    """

    #continuum images
    cont_res = Fido.search(a.Time(start_time, end_time),
                    a.jsoc.Series(Ic_serie),a.jsoc.Notify(jsoc_email))  

    cont_files = Fido.fetch(cont_res, path=path)
    cont_files.sort()

    #los magnetorgam data
    los_res = Fido.search(a.Time(start_time, end_time),
                    a.jsoc.Series(M_serie),a.jsoc.Notify(jsoc_email))  

    los_files = Fido.fetch(los_res, path=path)
    los_files.sort()

    cont_sequence = sunpy.map.Map(cont_files, sequence=True,allow_errors=True)
    los_sequence = sunpy.map.Map(los_files, sequence=True,allow_errors=True)

    return cont_sequence, los_sequence


def downsample_array(arr, window_size):
    """
    downsample a numpa array

    Params:
            - arr: numpy array
            - window_size: new size

    Return:
            - downsampled: downsampled numpy array
    """

    # Ensure dimensions are divisible by the window size
    assert arr.shape[0] % window_size == 0 and arr.shape[1] % window_size == 0, "Array dimensions must be divisible by the window size"
    
    # Reshape the array to create blocks of size window_size x window_size
    reshaped = arr.reshape(arr.shape[0] // window_size, window_size, arr.shape[1] // window_size, window_size)
    
    # Take the mean along the last two axes (averaging each block)
    downsampled = reshaped.mean(axis=(1, 3))
    
    return downsampled


def two_graphs_plot(cont_map, los_map):

    """
    Plot of the continuum and the los magnetogram sunpy map

    Params:
            - cont_map: map of the continuum data
            - los_map: map of the los magnetogram data
    
    """

    fig, axes = plt.subplots(1, 2, figsize=(10,28))  # Wider, more balanced layout

    # First subplot
    ax1 = axes[0]
    img1 = ax1.imshow(cont_map.data, cmap=cmr.sunburst, norm=ImageNormalize(vmin=0., vmax=1.7), alpha=0.7, interpolation='nearest')

    # Create smaller colorbar for first plot
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.48)  # Adjust size (5% width) and padding
    cbar1 = fig.colorbar(img1, cax=cax1, orientation='vertical')

    # Second subplot
    ax2 = axes[1]
    norm = colors.LogNorm(vmin=9e0, vmax=1.5e3)  # Define log normalization for second plot
    img2 = ax2.imshow(los_map.data,  cmap=cmr.get_sub_cmap('ocean_r', 0.1,1), norm=norm, alpha=0.7, interpolation='nearest')
    

    # Create smaller colorbar for second plot
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.48)  # Adjust size (5% width) and padding
    cbar2 = fig.colorbar(img2, cax=cax2, orientation='vertical', norm=norm, format="{x:.0e}")


    # Iterating over all the axes in the figure and making the Spines Visibility as False for both subplots
    for ax in [ax1, ax2]:
        for pos in ['right', 'top']:
                ax.spines[pos].set_visible(False)



    plt.tight_layout()
    ax1.set_title('Continuum intensity')
    ax2.set_title('LOS magnetogram [Gauss]')

    plt.show()




def coord_grid(map):
    """
    Return the helioprojective coordinates of a given map

    Params:
            - map: sunpy map

    Return:
            - xg,yg: grid of x,y coordinate
    """

    x = np.linspace(map.bottom_left_coord.Tx,map.top_right_coord.Tx,int(map.dimensions[0].value))
    y = np.linspace(map.bottom_left_coord.Ty,map.top_right_coord.Ty,int(map.dimensions[1].value))

    xg,yg = np.meshgrid(x,y)   

    return xg, yg 


def noise_threshold(map, threshold=8):
    """ 
    Set 0 for all pixels < threshold

    Params:
            - map: sunpy map
            - threshold: noise level

    Return: 
            - the map where all pixels < threshold are set to 0
    """

    return sunpy.map.sources.HMIMap( np.where(map.data < threshold, np.full(np.shape(map.data), np.nan), map.data), map.fits_header)


def mu(map, xg, yg):
    """ 
    Compute mu = cos(theta), the angle between the outward normal on the solar surface and the direction of the LOS
    
    Params:
            - map: sunpy map
            - xg, yg: mehsgrid of helioprojective coordinates

    Return: 
            - mu 
    """

    #radius of the sun
    Rs = map.rsun_obs

    return np.where( np.sqrt(xg**2 + yg**2) > Rs, np.full(np.shape(xg), np.nan),np.sqrt(xg**2 + yg**2) / Rs)


def removing_foreshortening_effect(map, xg, yg):
    """
    Remove the foreshortening effect in a map

    Params: 
            - map: sunpy map
            - xg, yg: mehsgrid of helioprojective coordinates

    Return:
            - the map without foreshortening effect
    """

    return sunpy.map.sources.HMIMap(map.data / mu(map, xg, yg), map.fits_header)


def active_area_identification(cont_map, los_map, xg, yg):
    """
    Identification of active pixels and divise them into bright faculae and dark sunspots

    Params:
            - cont_map: map of the continuum data
            - los_map: mag of the magnetic field data
            - xg,yg: mehsgrid of helioprojective coordinates

    Return:
            - identification array where the pixels are set to 1 for dark spots, 0 for bright faculae and nan otherwise

    """
    #threshold between active and quiet photosphere
    B_th = 24 / mu(los_map, xg, yg)

    #array of active pixels
    W_ij = np.where(los_map.data > B_th, 1, np.nan)
    
    #threshold between bright faculae and dark sunspots
    I_th = 0.89 * np.nansum(cont_map.data * W_ij) / np.nansum(W_ij)

    #array of pixels divided into bright and dark intensity
    V_ij = np.where(cont_map.data < I_th, 1, 0)
    V_ij = np.where(np.isnan(los_map.data), np.nan, V_ij) 

    return V_ij * W_ij


def active_area_smoothing(active_area, spot_threshold=0.5, spot_kernel_size=4, feature_threshold=0.5, feature_kernel_size=4):
    """
    Divide the the identification array for dark spots and bright features and smooth them with a gaussian kernel

    Params:
            - active_area: identification array
            - XXX_threshold: cutoff where the identification pixels < threshold are set to 0 
            - XXX_kernel_size: size of the gaussian kernel

    Return:
            - feature_area: identification array of bright features
            - smooth_feature_area: smoothed identification array of bright features
            - spot_area: identification array of darks sunspots
            - smooth_spot_area: smoothed identification array of darks sunspots
    """

    #divide the identification array into dark spots and bright features
    feature_area = np.where(active_area == 0, 1, np.nan)
    spot_area = np.where(active_area == 1, 1, np.nan)

    #smoothing 
    smooth_feature_area = np.where(np.isnan(feature_area),0, feature_area)
    smooth_feature_area =  convolve_fft(smooth_feature_area, Gaussian2DKernel(feature_kernel_size))
    smooth_feature_area = np.where(smooth_feature_area < feature_threshold, np.nan, smooth_feature_area)

    smooth_spot_area = np.where(np.isnan(spot_area),0, spot_area)
    smooth_spot_area =  convolve_fft(smooth_spot_area, Gaussian2DKernel(spot_kernel_size))
    smooth_spot_area = np.where(smooth_spot_area < spot_threshold, np.nan, smooth_spot_area)
    
    return spot_area, smooth_spot_area, feature_area, smooth_feature_area


def micro_hemisphere_to_arcsec2(map, value_in_mh=20):
    """
    Convert area to arcsec^2 from micro-hemisphere

    Params:
            - map: sunpy map to extract the solar radius at its observation time
            - value_in_mh: area value in micro-hemisphere

    Return: 
            - area in arcsec^2

    """
    #solar radius
    Rs = map.rsun_obs

    return value_in_mh * 1e-6 * np.pi * Rs**2


def value_nbr(array):
    """
    Count the number of non Nan value in an array

    Params:
            - array: numpy array
    Return:
            - number of non Nan value in the array 
    """

    return array.size - np.count_nonzero(np.isnan(array))


#TOFOX name
def helioprojective_to_heliospheric(x, y, map):
    """
    Give the angles in x and y-axis according to their helioprojective angles

    Params:
            - x,y: helioprojective angles in x,y-axis
            - map: TODO

    Return: 
            - theta, phi: angle coordinates

    """
    theta = np.where(np.sqrt(x**2 + y**2) <= map.rsun_obs, np.arctan((map.dsun - map.rsun_meters) / map.rsun_meters * np.tan(x)), np.nan)
    phi = np.where(np.sqrt(x**2 + y**2) <= map.rsun_obs, np.arctan((map.dsun - map.rsun_meters) / map.rsun_meters * np.tan(y)), np.nan)

    return theta, phi



def network_identification(feature_area, smooth_feature_area, resolution=1, area_th=60, method='scipy'):
    """
    Identification of plage and netwrok in the bright features using the following approachs:

    - scipy.ndimage
    - sklearn.cluster (soon)

    Params:
            -
            - smooth_active_area:
            - area_th
            - method: method of identification ('scipy' or 'sklearn')

    Return: 
            - identification: identification array where network pixels are set to 0, plage pixels are set to 1 and NaN otherwise
            - plage_nbr: number of plage objects
    
    """


    plage_nbr = 0

    identification = 0 * feature_area

    #using scipy.ndimage
    if method=='scipy':
        #TOTEST smooth or not?
        labeled_array, num = nd.label(np.where(np.isnan(smooth_feature_area), 0, smooth_feature_area ))
        print(num)

        slices = nd.find_objects(labeled_array) 

        for i in range(len(slices)):
            area = value_nbr(smooth_feature_area[slices[i]]) * resolution**2

            if area > area_th:
                identification[slices[i]] = np.where(np.isnan(smooth_feature_area[slices[i]]), np.nan, 1)
                plage_nbr += 1

         
    
    #using sklearn.cluster
    elif method=='sklearn':
            print('WARNING: this method has not been tested')
            points = np.argwhere(smooth_feature_area > 0)

            clustering = DBSCAN(eps=1e3, min_samples=5).fit(points)
            labels = clustering.labels_
            unique_labels = set(labels)
            
            for unique_label in unique_labels:
                if unique_label == -1:  # Skip noise
                    continue

                # Extract points for this cluster
                cluster_points = points[labels == unique_label]

                area = value_nbr(smooth_feature_area[cluster_points])

                if area <= area_th:
                    identification[cluster_points] = np.where(np.isnan(smooth_feature_area[cluster_points]), np.nan, 0)

                else:
                    identification[cluster_points] = np.where(np.isnan(smooth_feature_area[cluster_points]), np.nan, 1)
                    plage_nbr += 1

    else:
        print('This method is not available. Please chose between scipy and sklearn.')

    return identification, plage_nbr
        
            

def spot_nbr(spot_area, smooth_spot_area,  method='scipy'):
    """
    Compute the estimated number of dark sunspots, their location and their area in pixels using the following approachs:

    - scipy.ndimage
    - sklearn.cluster (soon)

    Params:
            - spot_area: identification array of dark spots
            - smooth_spot_area: smoothed identification array of dark spots
            - method: method of identification ('scipy' or 'sklearn')

    Return:
            - num_features: number of spots
            - locs: locations of the spots
            - pxl_area: 
    """

    #using scipy.ndimage
    if method=='scipy':

        pxl_area = []

        labeled_array, num_features = nd.label(np.where(np.isnan(smooth_spot_area), 0, spot_area))

        #TODO take projection into account?
        locs = nd.center_of_mass(smooth_spot_area, labels=labeled_array, index=range(1, num_features+1))
    
        #TOFIX
        if len(locs)>0:
            locs = np.split(np.array(locs),2,1)

            slices = nd.find_objects(labeled_array) 

            for i in range(len(slices)):
                pxl_area.append(value_nbr(smooth_spot_area[slices[i]]))



    #using sklearn.cluster
    elif method=='sklearn':
        print('WARNING: this method has not been tested')
        #TOTEST
        clustering = DBSCAN(eps=5, min_samples=1).fit(smooth_spot_area)
        labels = clustering.labels_
        unique_labels = set(labels)
            
        for unique_label in unique_labels:
            if unique_label == -1:  # Skip noise
                continue
            # Extract points for this cluster
            cluster_points = smooth_spot_area[labels == unique_label]

            # Compute centroid
            centroid = np.mean(cluster_points, axis=0)


    else:
        print('This method is not available. Please chose between scipy and sklearn.')
        


    return num_features, locs, pxl_area


def identifiation_plot(identification, spot_area, los_map, filename=None):
    """
    plot the identification map

    Params:
            - identifacation: faculae map
            - spot_area: spot map
            - los_map: sdo los magnetogram map
            - filename: if not None, filename to save the plot
    
    """

    final = np.full(np.shape(identification), np.nan)

    final = np.where(np.isnan(los_map.data), final, 0)

    final = np.where(identification == 1, 2, final)

    final = np.where(identification == 0, 3, final)

    final = np.where(spot_area > 0, 1, final)

    all_colors = cmr.take_cmap_colors(cmr.torch_r,N=20)
    newcolors = [all_colors[1], all_colors[3], all_colors[8], all_colors[18]]

    cmap = ListedColormap(newcolors)
    cmap.set_bad('white')  # Set NaN values to white


    plt.figure(dpi=300)
    im = plt.imshow(final, cmap=cmap)
    cbar = plt.colorbar(im)
    yticks = [0.6, 1.2, 2, 2.8]


    # add tick labels to colorbar
    cbar.set_ticks(yticks, labels=['Quiet sun', 'Spot', 'Faculae', 'Network'],rotation=90)
    cbar.ax.tick_params(length=0)         # remove tick lines
    

    #Selecting the axis-X making the bottom and top axes False. 
    plt.tick_params(axis='x', which='both', bottom=False, 
                    top=False, labelbottom=False) 
    
    # Selecting the axis-Y making the right and left axes False 
    plt.tick_params(axis='y', which='both', right=False, 
                    left=False, labelleft=False) 
    
    # Iterating over all the axes in the figure 
    # and make the Spines Visibility as False 
    for pos in ['right', 'top', 'bottom', 'left']: 
        plt.gca().spines[pos].set_visible(False) 
    

    if filename != None:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def features_rounding(array_size, pxl_locs, pxl_areas, area_th=1e3):
    
    """
    Replacing the identified features bigger than a area threshold by round objects

    Params:
            - array_size: size of the identification array
            - pxl_locs: localization in pixel units of the feature center
            - pxl_areas: area in pixel of the features
            - area_th (soon)

    Return:
            - rounded_feature_area: identification array with the rounded features
    """
    X = np.arange(0, int(array_size))
    Y = np.arange( int(array_size))

    x,y = np.meshgrid(X,Y)
    
    #initialization
    rounded_feature_area = np.zeros(x.shape)

    #radius
    R = np.sqrt(np.array(pxl_areas) / np.pi)


    for i in range(len(pxl_areas)):

        rounded_feature_area += Disk2D(1, np.squeeze(pxl_locs[0][i]), np.squeeze(pxl_locs[1][i]), R[i])(y, x)

    #TODO projection!!!!!

    return rounded_feature_area