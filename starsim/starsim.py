import tqdm
import numpy as np
import emcee
import gc
from pathlib import Path
from multiprocessing import Pool
import sys
import os
import corner
from configparser import ConfigParser
#import traceback
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m
import collections
from astropy.io import fits
import json
from scipy.interpolate import interp1d
from scipy import optimize
from . import spectra
from . import nbspectra
from . import SA
from numba import jit
import starsim
#initialize numba
nbspectra.dummy()

class StarSim(object): 
    """
    Main Starsim class. Reads the configuration file and store the options into variables.
    """
    def __init__(self,conf_file_path='starsim.conf'):
            self.path = Path(__file__).parent #path of working directory
            self.conf_file_path = self.path / conf_file_path
            self.conf_file = self.__conf_init() 
            #files
            self.filter_name =  str(self.conf_file.get('files','filter_name'))
            self.orders_CRX_filename= str(self.conf_file.get('files','orders_CRX_filename'))
            #general
            self.simulation_mode = str(self.conf_file.get('general','simulation_mode'))
            self.wavelength_lower_limit = float(self.conf_file.get('general','wavelength_lower_limit'))
            self.wavelength_upper_limit = float(self.conf_file.get('general','wavelength_upper_limit'))
            self.n_grid_rings = int(self.conf_file.get('general','n_grid_rings'))
            #star
            self.radius = float(self.conf_file.get('star','radius')) #Radius of the star in solar radii
            self.mass = float(self.conf_file.get('star','mass')) #Mass of the star in solar radii
            self.rotation_period = float(self.conf_file.get('star','rotation_period')) #Rotation period in days
            self.inclination = np.deg2rad(90-float(self.conf_file.get('star','inclination'))) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
            self.temperature_photosphere = float(self.conf_file.get('star','temperature_photosphere'))
            self.spot_T_contrast = float(self.conf_file.get('star','spot_T_contrast'))
            self.facula_T_contrast = float(self.conf_file.get('star','facula_T_contrast'))
            # self.temperature_spot = self.temperature_photosphere - float(self.conf_file.get('star','spot_T_contrast'))
            # self.temperature_facula = self.temperature_photosphere + float(self.conf_file.get('star','facula_T_contrast'))
            self.convective_shift = float(self.conf_file.get('star','convective_shift'))#CB in m/s
            self.logg = float(self.conf_file.get('star','logg'))
            self.facular_area_ratio = [] #float(self.conf_file.get('star','facular_area_ratio'))
            self.differential_rotation = float(self.conf_file.get('star','differential_rotation'))
            #rv
            self.ccf_template = str(self.conf_file.get('rv','ccf_template'))
            self.ccf_mask = str(self.conf_file.get('rv','ccf_mask'))
            self.ccf_weight_lines = int(self.conf_file.get('rv','ccf_weight_lines'))
            self.path_weight_lines = str(self.conf_file.get('rv','path_weight_lines'))
            self.ccf_rv_range= float(self.conf_file.get('rv','ccf_rv_range'))*1000 #in m/s
            self.ccf_rv_step= float(self.conf_file.get('rv','ccf_rv_step'))*1000 #in m/s
            self.kind_interp = 'cubic'#str(self.conf_file.get('rv','ccf_interpolation_spectra'))
            #limb-darkening
            self.use_phoenix_limb_darkening = int(self.conf_file.get('LD','use_phoenix_limb_darkening'))
            self.limb_darkening_law = str(self.conf_file.get('LD','limb_darkening_law'))
            self.limb_darkening_q1= float(self.conf_file.get('LD','limb_darkening_q1'))
            self.limb_darkening_q2= float(self.conf_file.get('LD','limb_darkening_q2'))
            #spots
            self.spots_evo_law = str(self.conf_file.get('spots','spots_evo_law'))
            self.plot_grid_map = int(self.conf_file.get('spots','plot_grid_map'))
            self.reference_time = float(self.conf_file.get('spots','reference_time'))
            #planet
            self.planet_period = float(self.conf_file.get('planet','planet_period')) #in days
            self.planet_transit_t0 = float(self.conf_file.get('planet','planet_transit_t0')) #in days
            self.planet_radius = float(self.conf_file.get('planet','planet_radius')) #in R* units
            self.planet_impact_param = float(self.conf_file.get('planet','planet_impact_param')) #from 0 to 1
            self.planet_spin_orbit_angle = float(self.conf_file.get('planet','planet_spin_orbit_angle'))*np.pi/180 #in deg
            self.simulate_planet=int(self.conf_file.get('planet','simulate_planet'))
            self.planet_semi_amplitude = float(self.conf_file.get('planet','planet_semi_amplitude')) #in m/s
            self.planet_esinw = float(self.conf_file.get('planet','planet_esinw')) 
            self.planet_ecosw = float(self.conf_file.get('planet','planet_ecosw')) 
            # self.inc_pl = np.arccos((self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-np.sqrt(self.planet_esinw**2+self.planet_ecosw**2))) #inclination of the planet orbit

            #optimization
            self.prior_spot_initial_time = json.loads(self.conf_file.get('optimization','prior_spot_initial_time'))
            self.prior_spot_life_time = json.loads(self.conf_file.get('optimization','prior_spot_life_time'))
            self.prior_spot_latitude = json.loads(self.conf_file.get('optimization','prior_spot_colatitude'))
            self.prior_spot_longitude = json.loads(self.conf_file.get('optimization','prior_spot_longitude'))
            self.prior_spot_coeff_1 = json.loads(self.conf_file.get('optimization','prior_spot_coeff_1'))
            self.prior_spot_coeff_2 = json.loads(self.conf_file.get('optimization','prior_spot_coeff_2'))
            self.prior_spot_coeff_3 = json.loads(self.conf_file.get('optimization','prior_spot_coeff_3'))
            self.prior_t_eff_ph = json.loads(self.conf_file.get('optimization','prior_t_eff_ph'))
            self.prior_spot_T_contrast = json.loads(self.conf_file.get('optimization','prior_spot_T_contrast'))
            self.prior_facula_T_contrast = json.loads(self.conf_file.get('optimization','prior_facula_T_contrast'))
            self.prior_q_ratio = json.loads(self.conf_file.get('optimization','prior_q_ratio'))
            self.prior_convective_blueshift = json.loads(self.conf_file.get('optimization','prior_convective_blueshift'))
            self.prior_p_rot = json.loads(self.conf_file.get('optimization','prior_p_rot'))
            self.prior_inclination = json.loads(self.conf_file.get('optimization','prior_inclination'))
            self.prior_Rstar = json.loads(self.conf_file.get('optimization','prior_stellar_radius'))
            self.prior_LD1 = json.loads(self.conf_file.get('optimization','prior_limb_darkening_q1'))
            self.prior_LD2 = json.loads(self.conf_file.get('optimization','prior_limb_darkening_q2'))
            self.prior_Pp = json.loads(self.conf_file.get('optimization','prior_period_planet'))
            self.prior_T0p = json.loads(self.conf_file.get('optimization','prior_time_transit_planet'))
            self.prior_Kp = json.loads(self.conf_file.get('optimization','prior_semi_amplitude_planet'))
            self.prior_esinwp = json.loads(self.conf_file.get('optimization','prior_esinw_planet'))
            self.prior_ecoswp = json.loads(self.conf_file.get('optimization','prior_ecosw_planet'))
            self.prior_Rp = json.loads(self.conf_file.get('optimization','prior_radius_planet'))
            self.prior_bp = json.loads(self.conf_file.get('optimization','prior_impact_parameter_planet'))
            self.prior_alp = json.loads(self.conf_file.get('optimization','prior_spin_orbit_planet'))           

            self.nwalkers = int(self.conf_file.get('optimization','N_walkers'))
            self.steps = int(self.conf_file.get('optimization','N_steps'))
            self.planet_impact_paramurns = int(self.conf_file.get('optimization','N_burns'))
            self.N_cpus = int(self.conf_file.get('optimization','N_cpus'))
            self.N_iters_SA = int(self.conf_file.get('optimization','N_iters_SA'))
            
            #others
            self.return_Imu = 0 # Returns I(mu) in order to get limb darkening law.
            self.convolve_spec_with_filt = False # If True, it convolves self.results['spec'] with the selected filter in order to get the light curve, which is stored in self.results['spec_lc'].
            self.prova = "lol"


            #FUNCTIONS USED TO ADD BISECTORS TO THE PHOTOSPHERE AT DIFF ANGLES
            self.fun_coeff_bisectors_amu = spectra.cifist_coeff_interpolate
            self.fun_coeff_bisector_spots = spectra.dumusque_coeffs
            self.fun_coeff_bisector_faculae = spectra.dumusque_coeffs


            #initialize other variables to store results
            self.data = collections.defaultdict(dict) #dictionary to store input data
            self.instruments = []
            self.observables=[]
            self.results = {} #initialize results attribute. It will be a dictionary containing the results of the method forward (maybe also inverse)
            self.name_params = {'rv': 'RV\n[m/s]', 'contrast': 'CCF$_{{cont}}$', 'fwhm':'CCF$_{{FWHM}}$\n[m/s]', 'bis':'CCF$_{{BIS}}$\n[m/s]', 
            'lc':'Norm. flux', 'ff_sp': 'ff$_{{spot}}$ \n[%]','ff_ph': 'ff$_{{phot}}$ \n[%]','ff_fc': 'ff$_{{fac}}$ \n[%]','ff_pl': 'ff$_{{pl}}$ \n[%]',
            'crx':'CRX \n[m/s/Np]','ccx':'C$_{{Cont}}$X \n[1/Np]','cfx':'C$_{{FWHM}}$X \n[m/s/Np]','cbx':'C$_{{BIS}}$X \n[m/s/Np]'}
            self.rvo = None #initialize
            self.conto = None #initialize
            self.fwhmo = None #initialize
            self.planet_impact_paramiso = None #initialize

            #read and check spotmap
            pathspots = self.path / 'spotmap.dat' #path relatve to working directory 
            self.spot_map=np.loadtxt(pathspots)

            if self.spot_map.ndim == 1:
                self.spot_map = np.array([self.spot_map]) #to avoid future errors
            elif self.spot_map.ndim == 0:
                sys.exit('The spot map file spotmap.dat is empty')
            self.facular_area_ratio = [self.spot_map[i][0] for i in range(len(self.spot_map))]

            #select mode
            if self.simulation_mode == 'grid':
                pass
            elif self.simulation_mode == 'fast':
                pass
            else: 
                sys.exit('simulation_mode in configuration file is not valid. Valid modes are "fast" or "grid".')

            #mode to select the template used to compute the ccf. Model are Phoenix models, mask are custom masks. 
            if self.ccf_template == 'model': #use phoenix models
                pass
            elif self.ccf_template == 'mask': #use maske
                pathmask = self.path / 'masks' / self.ccf_mask
                try:
                    d = np.loadtxt(pathmask,unpack=True)
                    if len(d) == 2:
                        self.wvm = d[0]
                        self.fm = d[1]
                    elif len(d) == 3:
                        self.wvm = spectra.air2vacuum((d[0]+d[1])/2) #HARPS mask ar in air, not vacuum
                        self.fm = d[2]
                    else:
                        sys.exit('Mask format not valid. Must have two (wv and weight) or three columns (wv1 wv2 weight).')

                except:
                    sys.exit('Mask file not found. Save it inside the masks folder.')


                if self.ccf_weight_lines:
                    pathweight = self.path / 'masks' / self.path_weight_lines
                    try:
                        order, order_weight, order_wvi, order_wvf = np.loadtxt(pathweight,unpack=True)
                        self.wvm, self.fm = nbspectra.weight_mask(order_wvi,order_wvf,order_weight,self.wvm,self.fm)
                    except:
                        sys.exit('File containing the weights of each order cold not be found/read. Make sure the file is inside the masks folder, and that it have 4 columns: order num, weight, initial wavelength, final wavelength')

                #Finally, set the wavlength range around the available lines
                self.wavelength_upper_limit=self.wvm.max()+1
                self.wavelength_lower_limit=self.wvm.min()-1
                print(self.wavelength_lower_limit)
                print(self.wavelength_upper_limit)


            else:
                sys.exit('ccf_template in configuration file is not valid. Valid modes are "model" or "mask".')

    @property
    def temperature_spot(self):
        return self.temperature_photosphere - self.spot_T_contrast

    @property
    def temperature_facula(self):
        return self.temperature_photosphere + self.facula_T_contrast

    @property
    def vsini(self):
        return 1000*2*np.pi*(self.radius*696342)*np.cos(self.inclination)/(self.rotation_period*86400) #vsini in m/s

    @property
    def planet_semi_major_axis(self):
        return 4.2097*self.planet_period**(2/3)*self.mass**(1/3)/self.radius #semi major axis in stellar radius units 


    def __conf_init(self):
        """creates an instance of class ConfigParser, and read the .conf file. Returns the object created
        ConfigParser, and read the .conf file. Returns the object created
        """
        conf_file_Object = ConfigParser(inline_comment_prefixes='#')
        if not conf_file_Object.read([self.conf_file_path]):
            print("The configuration file in" + str(self.conf_file_path) + " could not be read, please check that the format and/or path is are correct")
            sys.exit()
        else:
            return conf_file_Object


    def set_stellar_parameters(self,p):
        """Set the stellar parameters that have been optimized.
        """
        self.temperature_photosphere = p[0]
        self.spot_T_contrast = p[1]
        self.facula_T_contrast = p[2]
        self.facular_area_ratio = p[3]
        self.convective_shift = p[4]
        self.rotation_period = p[5]
        self.inclination = np.deg2rad(90-p[6]) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
        self.radius = p[7] #in Rsun
        self.limb_darkening_q1 = p[8]
        self.limb_darkening_q2 = p[9]
        self.planet_period = p[10]
        self.planet_transit_t0 = p[11]
        self.planet_semi_amplitude = p[12]
        self.planet_esinw = p[13]
        self.planet_ecosw = p[14]
        self.planet_radius = p[15]
        self.planet_impact_param = p[16]
        self.planet_spin_orbit_angle = p[17]*np.pi/180 #deg2rad    


    def compute_forward(self,observables=['lc'],t=None,inversion=False, w='eq_2'):


        if inversion==False:
            self.wavelength_lower_limit = float(self.conf_file.get('general','wavelength_lower_limit')) #Repeat this just in case CRX has modified the values
            self.wavelength_upper_limit = float(self.conf_file.get('general','wavelength_upper_limit'))
            print(self.wavelength_lower_limit)


        if t is None:
            sys.exit('Please provide a valid time in compute_forward(observables,t=time)')

        self.obs_times = t


        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(self.n_grid_rings)

        vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian
        theta, phi = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination)), np.arctan2(ys,xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))#coordinates in the star reference 


        #Main core of the method. Dependingon the observables you want, use lowres or high res spectra
        if 'lc' in observables: #use LR templates. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
   
            #Interpolate PHOENIX intensity models, only spot and photosphere
            acd, wvp_lc, flnp_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
            acd, wvp_lc, flns_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)

            #Read filter and interpolate it in order to convolve it with the spectra
            f_filt = spectra.interpolate_filter(self)

             
            
            if self.simulation_mode == 'grid':
                brigh_grid_ph, flx_ph = spectra.compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,f_filt,wvp_lc,'ph') #returns spectrum of grid in ring N, its brightness, and the total flux
                brigh_grid_sp, flx_sp = spectra.compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flns_lc,f_filt,wvp_lc,'sp') #returns spectrum of grid in ring N, its brightness, and the total flux
                brigh_grid_fc, flx_fc = brigh_grid_sp, flx_sp #if there are no faculae
                if np.sum(self.facular_area_ratio)>0:
                    brigh_grid_fc, flx_fc = spectra.compute_immaculate_facula_lc(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,f_filt,wvp_lc) #returns spectrum of grid in ring N, its brightness, and the total flux
                t,FLUX,ff_ph,ff_sp,ff_fc,ff_pl=spectra.generate_rotating_photosphere_lc(self,Ngrid_in_ring,pare,amu,brigh_grid_ph,brigh_grid_sp,brigh_grid_fc,flx_ph,vec_grid,inversion,plot_map=self.plot_grid_map)
                

            #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. NOT FACULAE AND NOT PLANETS YET.
            elif self.simulation_mode == 'fast': #in fast mode only immaculate photosphere is computed
                t,FLUX,ff_ph,ff_sp,ff_fc,ff_pl=nbspectra.generate_rotating_photosphere_fast_lc(self.obs_times,Ngrid_in_ring,acd,amu,pare,flnp_lc,flns_lc,f_filt(wvp_lc),self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)


            self.results['time']=t
            self.results['lc']=FLUX
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc

        
        if 'spec' in observables:#Oscar: based on the procedure for the 'lc' simulation, simulate the spectra as a function of time. At the moment using LR templates as the 'lc' simulation.
            
            #Interpolate PHOENIX intensity models, only spot and photosphere
            acd, wvp_lc, flnp_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
            acd, wvp_lc, flns_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)

             
            
            if self.simulation_mode == 'grid':
                spec_rings_ph, spec_ph = spectra.compute_immaculate_spec(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,wvp_lc,'ph') #returns spectrum of grid in ring N, its brightness, and the total flux
                spec_rings_sp, spec_sp = spectra.compute_immaculate_spec(self,Ngrid_in_ring,acd,amu,pare,flns_lc,wvp_lc,'sp') #returns spectrum of grid in ring N, its brightness, and the total flux
                spec_rings_fc, spec_fc = spec_rings_ph, spec_ph #if there are no faculae
                if np.sum(self.facular_area_ratio)>0:
                    spec_rings_fc, spec_fc = spectra.compute_immaculate_spec(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,wvp_lc,'fc')

                t,SPEC,ff_ph,ff_sp,ff_fc,ff_pl=spectra.generate_rotating_photosphere_spec(self,Ngrid_in_ring,pare,amu,spec_rings_ph,spec_rings_sp,spec_rings_fc,spec_ph,vec_grid,inversion,plot_map=self.plot_grid_map)
            
            
            elif self.simulation_mode == 'fast':
                sys.exit("Simulation of 'spec' is unavailable for the 'fast' mode, please use the 'grid' mode.")
            
            
            else:
                sys.exit("Invalid simulation mode. Only 'grid' mode is available for simulation of 'spec'.")
            
            
            
            self.results['time']=t
            self.results['spec']=SPEC
            self.results['spec_wv']=wvp_lc
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc
            
            
            if self.convolve_spec_with_filt: # I added this parameter to ask if you want to convolve self.results['spec'] with the selected filter in order to get the light curve.
                self.results['spec_lc']=spectra.convolve_spec_with_specified_filter(self)
                
                # #Read filter and interpolate it in order to convolve it with the spectra
                # f_filt = spectra.interpolate_filter(self)
                # spec_lc=np.zeros(len(t)) #brightness for each time step
                # spec_conv_filt=np.zeros([len(t),len(wvp_lc)]) #spectra for each time step convolved with filter
                
                # for k in range(len(t)):
                #     spec_conv_filt[k,:]=SPEC[k,:]*f_filt(wvp_lc) #convolve with filter.
                #     spec_lc[k]=np.sum(spec_conv_filt[k,:])
                
                # self.results['spec_lc']=spec_lc




        if 'rv' in observables or 'bis' in observables or 'fwhm' in observables or 'contrast' in observables: #use HR templates. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
            rvel=self.vsini*np.sin(theta)*np.sin(phi)#*np.cos(self.inclination) #radial velocities of each grid

            wv_rv, flnp_rv, flp_rv =spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
            wv_rv, flns_rv, fls_rv =spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)
            if np.sum(self.facular_area_ratio)>0:
                wv_rv, flnf_rv, flf_rv =spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
            spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized
            

            #Interpolate also Phoenix intensity models to the Phoenix wavelength. 
            # if self.use_phoenix_limb_darkening: 
            acd, wv_rv_LR, flpk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
            acd, wv_rv_LR, flsk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)
            #acd, wv_rv_LR, flfc_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_facula,self.logg)
            #Compute the CCF of the spectrum of each element againts the reference template (photosphere)
            rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
            #CCF with phoenix model  
            if self.ccf_template == 'model':
                ccf_ph = nbspectra.cross_correlation_nb(rv,wv_rv,flnp_rv,wv_rv,spec_ref)
                ccf_sp = nbspectra.cross_correlation_nb(rv,wv_rv,flns_rv,wv_rv,spec_ref)
                if np.sum(self.facular_area_ratio)>0:            
                    ccf_fc = nbspectra.cross_correlation_nb(rv,wv_rv,flnf_rv,wv_rv,spec_ref)
                else:
                    ccf_fc=ccf_ph*0.0

            elif self.ccf_template == 'mask':
                if wv_rv.max()<(self.wvm.max()+1) or wv_rv.min()>(self.wvm.min()-1):
                    sys.exit('Selected wavelength must cover all the mask wavelength range, including 1A overhead covering RV shifts. Units in Angstroms.')

                ccf_ph = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                ccf_sp = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flns_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                if np.sum(self.facular_area_ratio)>0:
                    ccf_fc = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnf_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                else:
                    ccf_fc=ccf_ph*0.0

            

            #Compute the bisector of the three reference CCF and return a cubic spline f fiting it, such that rv=f(ccf).
            fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=False,kind_interp=self.kind_interp)
            rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
            fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=False,kind_interp=self.kind_interp)
            rv_sp = rv - fun_bis_sp(ccf_sp)
            rv_fc = rv_ph
            if np.sum(self.facular_area_ratio)>0:            
                fun_raw_xbisc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)        
                rv_fc = rv - fun_raw_xbisc(ccf_fc)

            if self.simulation_mode == 'grid':

                ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                ccf_sp_g = spectra.compute_immaculate_spot_rv(self,Ngrid_in_ring,acd,amu,pare,flsk_rv,rv_sp,rv,ccf_sp,flxph,rvel)
                ccf_fc_g = ccf_ph_g #to avoid errors, not used
                if np.sum(self.facular_area_ratio)>0:
                    # print('Computing facula. Limb brightening is hard coded. Luke Johnson 2021 maybe is better millor.')
                    ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_fc,rv,ccf_fc,flxph,rvel, wv_rv_LR)
                    #ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flfc_rv,rv_fc,rv,ccf_fc,flxph,rvel)


                RV0, C0, F0, B0,_,_ =spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere

                #integrate the ccfs with doppler shifts at each time stamp
                t,CCF,ff_ph,ff_sp,ff_fc,ff_pl, vec_pos=spectra.generate_rotating_photosphere_rv(self,Ngrid_in_ring,pare,amu,rv,ccf_ph_tot,ccf_ph_g,ccf_sp_g,ccf_fc_g,vec_grid,inversion,plot_map=self.plot_grid_map) 
                

            #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. 
            elif self.simulation_mode == 'fast': #in fast mode only immaculate photosphere is computed
                ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #CCF of immaculate rotating pphotosphere
                RV0, C0, F0, B0,_,_ =spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere
                
                t,CCF,ff_ph,ff_sp,ff_fc,ff_pl=nbspectra.generate_rotating_photosphere_fast_rv(self.obs_times,Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,flxph,flpk_rv,flsk_rv,self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.vsini,self.convective_shift,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)


            if self.simulate_planet:
                rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
            else:
                rvkepler = 0.0

            ccf_params=spectra.compute_ccf_params(self,rv,CCF,plot_test=False)
            self.results['time']=self.obs_times
            self.results['rv']=ccf_params[0] - RV0 + rvkepler #subtract rv of immaculate photosphere
            self.results['contrast']=ccf_params[1]/C0
            self.results['fwhm']=ccf_params[2]
            self.results['bis']=ccf_params[3]
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc
            self.results['CCF']=np.vstack((rv,CCF))
            self.results['raw_xbis'] = ccf_params[4]
            self.results['raw_ybis'] = ccf_params[5]
            self.results['pos'] = vec_pos
            self.results['rv0'] = RV0


            




        if 'crx' in observables: #use HR templates in different wavelengths to compute chromatic index. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
            if w == 'eq_2':
                rotation_period_lat = 1/(1/self.rotation_period + (self.differential_rotation/(2.66) * (1.698 * np.sin(np.pi/2 - theta)**2 + 2.346 * np.sin(np.pi/2 - theta)**4))/360) #Add diff rotation
            elif w =='eq_1':
                rotation_period_lat = 1/(1/self.rotation_period + (self.differential_rotation * np.sin(np.pi/2 - theta)**2) / 360)

            vsini = 1000*2*np.pi*(self.radius*696342)*np.cos(self.inclination)/(rotation_period_lat*86400)
            rvel=vsini*np.sin(theta)*np.sin(phi) #radial velocities of each grid. Inclination already in vsini

            pathorders = self.path / 'orders_CRX' / self.orders_CRX_filename
            # print('Reading the file in',pathorders,'containing the wavelengthranges of each echelle order,to compute the CRX')
            try:
                orders, wvmins, wvmaxs = np.loadtxt(pathorders,unpack=True)
            except:
                sys.exit('Please, provide a valid file containing the order number and wavelength range, inside the folder orders_CRX')

            rvso=np.zeros([len(self.obs_times),len(orders)])
            conto=np.zeros([len(self.obs_times),len(orders)])
            fwhmo=np.zeros([len(self.obs_times),len(orders)])
            biso=np.zeros([len(self.obs_times),len(orders)])
            for i in range(len(orders)):
                # print('\nOrder: {:.0f}, wv range: {:.1f}-{:.1f} nm'.format(orders[i],wvmins[i],wvmaxs[i]))

                self.wavelength_lower_limit, self.wavelength_upper_limit = wvmins[i], wvmaxs[i]

                wv_rv, flnp_rv, flp_rv = spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
                wv_rv, flns_rv, fls_rv = spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)
                if np.sum(self.facular_area_ratio)>0:
                    wv_rv, flnf_rv, flf_rv = spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
                spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized


                acd, wv_rv_LR, flpk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
                acd, wv_rv_LR, flsk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)


                #Compute the CCF of the spectrum of each element againts the reference template (photosphere)
                rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
                ccf_ph = nbspectra.cross_correlation_nb(rv,wv_rv,flnp_rv,wv_rv,spec_ref)
                ccf_sp = nbspectra.cross_correlation_nb(rv,wv_rv,flns_rv,wv_rv,spec_ref)
                if np.sum(self.facular_area_ratio)>0:            
                    rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
                    ccf_fc = nbspectra.cross_correlation_nb(rv,wv_rv,flnf_rv,wv_rv,spec_ref)
                else:
                    ccf_fc = ccf_ph*0.0

                #Compute the bisector of the three reference CCF and return a cubic spline f fiting it, such that rv=f(ccf).
                fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=False,kind_interp=self.kind_interp)
                rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
                fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=False,kind_interp=self.kind_interp)
                rv_sp = rv - fun_bis_sp(ccf_sp)
                rv_fc = rv_ph

                if np.sum(self.facular_area_ratio)>0:            
                    fun_raw_xbisc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)        
                    rv_fc = rv - fun_raw_xbisc(ccf_fc)


                if self.simulation_mode == 'grid':
                    #COMPUTE CCFS of each ring of a non-rotating IMMACULATE PHOTOSPHERE, and total flux of the immaculate star
                    # print('Computing photosphere')

                    ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                    ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                    # print('Computing spot')
                    ccf_sp_g = spectra.compute_immaculate_spot_rv(self,Ngrid_in_ring,acd,amu,pare,flsk_rv,rv_sp,rv,ccf_sp,flxph,rvel)
                    ccf_fc_g = ccf_ph_g #to avoid errors, not used
                    if np.sum(self.facular_area_ratio)>0:
                        # print('Computing facula. Limb brightening is hard coded. Luke Johnson 2021 maybe is better millor.')
                        ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_fc,rv,ccf_fc,flxph,rvel)
                    
                    RV0, C0, F0, B0,_,_ =spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere
                    
                    #integrate the ccfs with doppler shifts at each time stamp
                    t,CCF,ff_ph,ff_sp,ff_fc,ff_pl, vec_pos=spectra.generate_rotating_photosphere_rv(self,Ngrid_in_ring,pare,amu,rv,ccf_ph_tot,ccf_ph_g,ccf_sp_g,ccf_fc_g,vec_grid,inversion,plot_map=self.plot_grid_map) 


                #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. 
                elif self.simulation_mode == 'fast': #in fast mode only immaculate photosphere is computed
                    ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                    ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #CCF of immaculate rotating pphotosphere
                       
                    RV0, C0, F0, B0,_,_ =spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere
                    
                    t,CCF,ff_ph,ff_sp,ff_fc,ff_pl=nbspectra.generate_rotating_photosphere_fast_rv(self.obs_times,Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,flxph,flpk_rv,flsk_rv,self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.vsini,self.convective_shift,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)


                if self.simulate_planet:
                    rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
                else:
                    rvkepler = 0.0


                ccf_params=spectra.compute_ccf_params(self,rv,CCF,plot_test=False)
                
                rvso[:,i]=ccf_params[0] + rvkepler #do not subtract offsets, could bias crx
                conto[:,i]=ccf_params[1]
                fwhmo[:,i]=ccf_params[2]
                biso[:,i]=ccf_params[3]

            lambdas = (np.log(wvmaxs)+np.log(wvmins))/2 #natural log of the central wavelength
            crx=np.zeros(len(self.obs_times))
            ccx=np.zeros(len(self.obs_times))
            cfx=np.zeros(len(self.obs_times))
            cbx=np.zeros(len(self.obs_times))
            for i in range(len(self.obs_times)): #compute crx for each time
                crx[i]=np.polyfit(lambdas,rvso[i,:],deg=1)[0] #crx is the slope of the rv as a function of the central wavelength
                ccx[i]=np.polyfit(lambdas,conto[i,:],deg=1)[0]
                cfx[i]=np.polyfit(lambdas,fwhmo[i,:],deg=1)[0]
                cbx[i]=np.polyfit(lambdas,biso[i,:],deg=1)[0]

            self.results['time']=self.obs_times
            self.rvo=rvso
            self.conto=conto
            self.fwhm=fwhmo
            self.planet_impact_paramiso=biso
            self.results['ccx']=ccx
            self.results['cfx']=cfx
            self.results['cbx']=cbx
            self.results['crx']=crx
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc

        return 


    
    
    def convolve_given_spec_with_specified_filters(self, t=None, spec=None, wv=None, filter_name_list=[None], parallelise=True, vectorise=True):
        """
        If nothing is given, it will convolve the computed self.results with the filter in the config file.

        Parameters
        ----------
        t : TYPE, optional
            DESCRIPTION. The default is self.results['t'].
        spec : TYPE, optional
            DESCRIPTION. The default is self.results['spec'].
        wv : TYPE, optional
            DESCRIPTION. The default is self.results['spec_wv'].
        filter_name_list : TYPE, optional
            DESCRIPTION. The default is [None].
        parallelise : TYPE, optional
            DESCRIPTION. The default is True.
        vectorise : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        spec_lcs : TYPE
            Spectral light curves.

        """
        # Check if 't' is provided as an argument or exists in self.results
        if t is None and 't' not in self.results:
            raise ValueError("'t' has not been provided. Please compute 'self.results['spec']' first (self.compute_forward(observables=['spec']) or provide the time ('t') as an argument.")
        if spec is None and 'spec' not in self.results:
            raise ValueError("The spectral time series ('spec') has not been computed. Please compute 'self.results['spec']' first (self.compute_forward(observables=['spec']) or provide 'spec' as an argument.")
        # Check if 'wv' is provided as an argument or exists in self.results
        if wv is None and 'wv' not in self.results:
            raise ValueError("'wv' (wavelength) has not been provided. Please compute 'self.results['spec']' first (self.compute_forward(observables=['spec']) or provide the wavelength ('wv') as an argument.")
        
        if parallelise:
            num_cores = multiprocessing.cpu_count()
            
            # Parallel execution for each filter_name
            spec_lcs = Parallel(n_jobs=num_cores)(delayed(spectra.convolve_spec_with_specified_filter)(self, t=t, spec=spec, wv=wv, filter_name=filter_name, vectorise=vectorise) for filter_name in filter_name_list)
            
        else:
            spec_lcs = []
            for filter_name in filter_name_list:
                spec_lcs.append(spectra.convolve_spec_with_specified_filter(self, t=t, spec=spec, wv=wv, filter_name=filter_name, vectorise=vectorise))
        
        return spec_lcs
    

    def compute_spectral_lcs(self, t=None, spec=None, wv=None, temp_filters_wv_range=None, temp_filters_wv_step=None, filter_name_list=None, delete_temp_directories=True, parallelise=True, vectorise=True, normalise_by_max=False, return_filter_paths_from_filters_folder=False):
        """
        

        Parameters
        ----------
        temp_filters_wv_range : [min_wv,max_wv] array, optional
            Wavelength range to cover with spectral lcs [min_wv,max_wv]. The default is None.
        temp_filters_wv_step:
            Wavelength step to generate spectral lc filters between wv-step_lc,wv+step_lc for wv in np.arange(min_wv,max_wv,step_wv)
        filter_name_list : TYPE, optional
            DESCRIPTION. The default is None.
        delete_temp_directories : TYPE, optional
            Whether to delete the temporary directories with generated filter files and configuration files. The default is True.

        Returns
        -------
        None.

        """
        if temp_filters_wv_range is None and filter_name_list is None:
            filter_name_list = [self.filter_name]
        if temp_filters_wv_range is not None and filter_name_list is not None:
            sys.exit('Please either provide a filter name list or a wavelength range to generate temporary filters (not both at the same time).')
        if temp_filters_wv_range is not None and temp_filters_wv_step is None:
            sys.exit('Please provide a wavelength step to generate temporary filters.')
        if temp_filters_wv_range is None and temp_filters_wv_step is not None:
            sys.exit('Please provide a wavelength range to generate temporary filters.')
        
        
        
        if temp_filters_wv_range is not None:
            # Define the directory where filter files will be stored
            filter_directory = str(self.path / 'models/filters/spec_flat_filters_for_each_wv/')
            
            # Define the directory where temporary configuration files will be stored
            conf_directory = str(self.path / 'spec_conf_files_for_each_wv/')
            
            filter_paths_from_filters_folder = spectra.generate_temporary_filters_and_conf_files(self, temp_filters_wv_range, temp_filters_wv_step, filter_directory, conf_directory)
            
            spec_lcs = self.convolve_given_spec_with_specified_filters(t=t, spec=spec, wv=wv, filter_name_list=filter_paths_from_filters_folder, parallelise=parallelise, vectorise=vectorise)
            
            # Delete created directories
            if delete_temp_directories:
                try:
                    shutil.rmtree(filter_directory)
                except Exception as e:
                    # Log the exception or handle it in a specific way
                    print(f"Error deleting {filter_directory}: {e}")
                try:
                    shutil.rmtree(conf_directory)
                except Exception as e:
                    # Log the exception or handle it in a specific way
                    print(f"Error deleting {conf_directory}: {e}")
                
                #shutil.rmtree(filter_directory, ignore_errors=True)
                #shutil.rmtree(conf_directory, ignore_errors=True)
        else:
            spec_lcs = self.convolve_given_spec_with_specified_filters(t=t, spec=spec, wv=wv, filter_name_list=filter_name_list, parallelise=parallelise, vectorise=vectorise)
        
        if return_filter_paths_from_filters_folder:
            if normalise_by_max:
                return np.array(spec_lcs)/np.array(spec_lcs).max(axis=1).reshape(-1,1),filter_paths_from_filters_folder
            else:
                return spec_lcs,filter_paths_from_filters_folder
        else:
            if normalise_by_max:
                return np.array(spec_lcs)/np.array(spec_lcs).max(axis=1).reshape(-1,1)
            else:
                return spec_lcs
        
        
    
    def compute_white_light_curve_from_spec(self, t=None, spec=None, wv=None, vectorise=True, normalize_by_max=False):
        """
        Computes the white light curve by summing across all wavelengths.
    
        Parameters
        ----------
        t : array-like, optional
            Time array. Default is self.results['time'].
        spec : 2D array-like, optional
            Spectral data with time and wavelength dimensions. Default is self.results['spec'].
        wv : array-like, optional
            Wavelength array. Default is self.results['spec_wv'].
        vectorise : bool, optional
            If True, uses vectorized operations. Default is True.
    
        Returns
        -------
        white_light_curve : 1D array
            White light curve as a time series.
        """
        if t is None:
            t = self.results['time']
        if spec is None:
            spec = self.results['spec']
        if wv is None:
            wv = self.results['spec_wv']
    
        if vectorise:
            # Sum across the wavelength dimension to get the white light curve
            white_light_curve = np.sum(spec, axis=1)  # Collapse wavelength axis
        else:
            # Non-vectorized method
            white_light_curve = np.zeros(len(t))
            for k in range(len(t)):
                white_light_curve[k] = np.sum(spec[k, :])
                
        # Normalize by maximum value if requested
        if normalize_by_max:
            max_value = np.max(white_light_curve)
            if max_value > 0:  # Prevent division by zero
                normalized_white_light_curve = white_light_curve / max_value
            else:
                normalized_white_light_curve = white_light_curve  # No normalization if max is zero
        else:
            normalized_white_light_curve = white_light_curve  # Return unnormalized
    
        return normalized_white_light_curve

    
#CAREFUL!!! THIS IS NOT EQUIVALENT TO COMPUTING THE WHITE LC FROM THE WHOLE SPECTRUM AS A FUNCTION OF TIME, SINCE SPECTRAL LCS CORRESPOND TO INDEPENDENTLY NORMALISED INTEGRATED SECTIONS OF THE SPECTRUM!!!
    # def compute_white_light_curve_from_spectral_lcs(spectral_lcs, spectral_lcs_err=None):
    #     """
    #     Computes the normalized total/white light curve by summing the provided spectral light curves,
    #     and optionally the associated errors.
    
    #     Parameters
    #     ----------
    #     spectral_lcs : list of 1D arrays
    #         List containing spectral light curves, each representing a different wavelength window.
    #     spectral_lcs_err : list of 1D arrays, optional
    #         List containing errors corresponding to each spectral light curve.
    
    #     Returns
    #     -------
    #     normalized_total_light_curve : 1D array
    #         The summed and normalized total/white light curve.
    #     normalized_total_light_curve_err : 1D array, optional
    #         The normalized errors for the total/white light curve, if errors were provided.
    #     """
    #     # Ensure there are spectral light curves to process
    #     if not spectral_lcs:
    #         raise ValueError("The input list 'spectral_lcs' is empty.")
    
    #     # Sum the spectral light curves
    #     total_light_curve = np.sum(spectral_lcs, axis=0)
        
    #     # Normalize by the number of light curves
    #     normalized_total_light_curve = total_light_curve / len(spectral_lcs)
    
    #     # If errors are provided, compute the normalized error
    #     if spectral_lcs_err is not None:
    #         if len(spectral_lcs_err) != len(spectral_lcs):
    #             raise ValueError("The lengths of 'spectral_lcs' and 'spectral_lcs_err' must match.")
            
    #         total_light_curve_err = np.sqrt(np.sum(np.array(spectral_lcs_err)**2, axis=0))
    #         normalized_total_light_curve_err = total_light_curve_err / len(spectral_lcs)
    #         return normalized_total_light_curve, normalized_total_light_curve_err
    
    #     return normalized_total_light_curve, None


    
    
    
    # TODO: Parallelise when using 'multinest' sampler, since it is not parallelised on juliet, thus we could fit multiple wavelengths at the same time and save time.
    def compute_transmission_spectrum(self, t, spectral_lcs, spectral_lcs_err=None, wv_range=None, wv_step=None, wvs=None, white_lc=None, white_lc_err=None, transit_time=None, transit_length=None, planet_inc=None, planet_inc_for_length=True, plot_model=False, plot_fit_metrics=True, use_juliet=False, juliet_fit_linear_trend=True, juliet_sampler='multinest', juliet_nthreads=None, juliet_dists=None, juliet_hyperps=None, juliet_out_folder=None, juliet_true_values=None, juliet_return_posteriors=False, juliet_create_tmp_symlink=False, juliet_symlink_dir=None, juliet_use_hyperps_per_wv=False, juliet_output_fits=True, juliet_ld_law='quadratic', use_batman_curve_fit=False, batman_fit_linear_trend=True, batman_plot_linear_fits=False, batman_plot_lc_fits=False, batman_ini_pars=None, batman_compute_white_light_curve=False, batman_spec=None, batman_spec_wv=None, batman_out_folder=None, batman_return_u_fitted_list=False, batman_u_fitted_list_to_use=None, batman_return_a_fitted_list=False, batman_a_fitted_list_to_use=None, batman_fit_params={'rp': True, 't0': False, 'a': False, 'inc': False, 'per': False, 'u': True}, batman_ld_law='quadratic', batman_fit_white_lc=True, batman_fit_period_for_white_lc=False, batman_b_to_fix=None, batman_model_based_3point=False, batman_use_kipping_ldc_parametrisation=True, batman_return_batman_fit_metrics_list=False, batman_output_data=False, use_three_point_approx=False, three_point_out_folder=None):
        """
        

        Parameters
        ----------
        wv_range : TYPE
            DESCRIPTION.
        wv_step : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.
        spectral_lcs : TYPE
            DESCRIPTION.
        transit_time : TYPE, optional
            DESCRIPTION. The default is None.
        transit_length : TYPE, optional
            DESCRIPTION. The default is None.
        planet_inc : TYPE, optional
            Planet inclination IN DEGREES!!!. The default is None.
        planet_inc_for_length : TYPE, optional
            Whether to use the planet inclination to compute transit length (correct expression). The default is True.
        use_juliet : TYPE, optional
            DESCRIPTION. The default is False.
        juliet_dists : TYPE, optional
            Example: dists = ['fixed','uniform','TruncatedNormal','fixed','uniform','uniform','fixed','fixed',\
                              'fixed', 'fixed', 'fixed', 'fixed']. The default is None.
        juliet_hyperps : TYPE, optional
            Hyperparameters of the distributions (mean and standard-deviation for normal distributions, lower and upper limits for uniform and loguniform distributions, and fixed values for fixed "distributions", which assume the parameter is fixed). Example: hyperps = [planet_period_cloutier, [transit_time-0.1,transit_time+0.1], [0.116,0.02,0.08,0.15], 0.19, [0., 1.], [0., 1.], 0.0, 90.,a_over_Rstar, 1.0, 0., 0.1]. The default is None.
        juliet_true_values : TYPE, optional
            True values for the hyperparameters, needed to plot corner plots. The default is None.
        use_batman_curve_fit : TYPE, optional
            DESCRIPTION. The default is False.
        use_three_point_approx : TYPE, optional
            DESCRIPTION. The default is False.
        
        juliet_symlink_dir : 
            If 'None' is given, it will try to use the first two folders of the path.
        juliet_use_hyperps_per_wv : 
            If True, instead of entering a list of values for the hyperparameters, you enter an array of lists of values for the hyperparameters, i.e. each list of values for the hyperparameters coresponds to each wavelength.
        juliet_output_fits : 
            Whether to output a folder from juliet containing fits parameters, posteriors, data, etc. or not. Does not affect whether you output plots or not.
        juliet_ld_law : 
            String with the name of the law to use — currently supported laws are the 'quadratic', the 'logarithmic' and the 'squareroot' laws (both with q1 and q2). Juliet also supports linear law (only q1) but it is not implemented here.
        Returns
        -------
        results_depths : TYPE
            DESCRIPTION.

        """
        def log_to_file(filename, message): #This function is to debug
            with open(filename, 'a') as f:  # 'a' mode for appending
                f.write(f"{message}\n")  # Write message and add newline
        
        if not use_juliet and not use_batman_curve_fit and not use_three_point_approx:
            sys.exit('Please choose at least one option to compute transit depths amongst those available: use juliet (slowest), use batman&curve_fit (faster), or use the three-point approximation (fastest).')
        if wvs is None:
            if wv_range is None or wv_step is None:
                sys.exit("Provide either wvs or both wv_range and wv_step.")
            else:
                min_wv,max_wv = wv_range
                step_wv = wv_step
                wvs = np.arange(min_wv,max_wv,step_wv)
        elif wv_range is not None or wv_step is not None:
            sys.exit("Provide either wvs or both wv_range and wv_step.")
        
        
        if transit_time is None:
            sys.exit('Please enter the transit central time or an estimate.')
        
        if planet_inc is None:
            if(self.planet_esinw==0 and self.planet_ecosw==0):
               ecc=0.
               omega=90. * np.pi / 180.
            else:
               ecc=np.sqrt(self.planet_esinw**2+self.planet_ecosw**2)
               omega=np.arctan2(self.planet_esinw,self.planet_ecosw)

            cosi = (self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-ecc**2) #cosine of planet inclination
            inclination = np.arccos(cosi)*180./np.pi
        else:
            inclination = planet_inc
        
        if transit_length is None:
            #Transit length expression from Joshua N. Winn (2014) (https://arxiv.org/abs/1001.2010)
            
            if planet_inc_for_length:
                theoretical_transit_length = self.planet_period/np.pi * np.arcsin(1./self.planet_semi_major_axis * np.sqrt((1+self.planet_radius)**2 - self.planet_impact_param**2)/np.sin(np.pi/180*inclination))
            else:
                theoretical_transit_length = self.planet_period/np.pi * np.arcsin(1./self.planet_semi_major_axis * np.sqrt((1+self.planet_radius)**2 - self.planet_impact_param**2))
            
            
            print("Theoretical transit length: {}".format(theoretical_transit_length))
            transit_length = theoretical_transit_length
        
        t_ini_transit = transit_time-transit_length/2
        t_end_transit = transit_time+transit_length/2
    
        idx_ini_transit = np.argmin(np.abs(t-t_ini_transit))
        idx_end_transit = np.argmin(np.abs(t-t_end_transit))
        
        
        results_depths = {} #Dictionary where each entry is an array of the computed transit depths using each specified method.
        results_depths['juliet'],results_depths['batman_curve_fit'],results_depths['three_point_approx'] = None,None,None
        if use_juliet:
            start_transit_time = time()
            start_wv = start_transit_time
            fit_metrics = {}
            linear_fit_metrics = {}
            #ldc = {}
            depth = {}
            depth_otherdef = {}
            errors_juliet_fitting = []
            juliet_posteriors = []
            
            if juliet_out_folder is None:
                sys.exit('Please enter an output folder for juliet. Beware multinest does not support folder paths longer than 69 characters.')
            else:
                def ensure_folder_exists(folder_path):
                    # Check if the folder exists
                    if not os.path.exists(folder_path):
                        # If it does not exist, create it
                        os.makedirs(folder_path)
                        print(f"Folder created: {folder_path}")
                    else:
                        print(f"Folder already exists: {folder_path}")
                ensure_folder_exists(juliet_out_folder)
            
            if juliet_create_tmp_symlink:
                def create_symlink(original_path, symlink_dir=None):
                    """
                    Create a symlink in a specified directory.
                    
                    Parameters:
                        original_path (str): The original directory path that may be too long.
                        symlink_dir (str, optional): The directory where the symlink should be created. 
                                                     If None, defaults to the first two folders of original_path + '/tmp'.
                        max_length (int): The maximum allowed length for the path. Default is 69 characters.
                
                    Returns:
                        str: The path to use, either the original or the new symlink.
                
                    Raises:
                        Exception: If the symlink directory cannot be created or used.
                    """
                    if symlink_dir is None:
                        parts = original_path.split(os.sep)
                        if len(parts) >= 3:
                            symlink_dir = os.path.join(os.sep, parts[1], parts[2], 'tmp')
                        else:
                            raise Exception(f"Cannot determine a valid symlink directory from path: {original_path}")
            
                    # Create the symlink directory if it doesn't exist
                    if not os.path.exists(symlink_dir):
                        try:
                            os.makedirs(symlink_dir)
                        except Exception as e:
                            raise Exception(f"Failed to create symlink directory: {symlink_dir}") from e
            
                    # Generate a shorter symlink name based on the basename of the original path
                    symlink_name = os.path.basename(original_path)
                    symlink_path = os.path.join(symlink_dir, symlink_name)
                    
                    if len(symlink_path)>=42: #About the maximum to avoid the whole name being >=69.
                        symlink_name = 's'
                        symlink_path = os.path.join(symlink_dir, symlink_name)
                    
                    # Ensure the symlink doesn't already exist
                    if not os.path.exists(symlink_path):
                        try:
                            os.symlink(original_path, symlink_path)
                            print(f"Created symlink: {symlink_path} -> {original_path}")
                        except Exception as e:
                            raise Exception(f"Failed to create symlink: {symlink_path}") from e
                    else:
                        print(f"Symlink already exists: {symlink_path}")
            
                    return symlink_path
                
                juliet_original_out_folder = juliet_out_folder
                juliet_out_folder = create_symlink(juliet_out_folder,juliet_symlink_dir)
                


            
            
            #Linear fit by hand and then juliet fit:
            for i,wv in enumerate(wvs):
                clear_output(wait=True) #Clear output for each iteration
                print("\n\n\n------Wavelength {}.------\n Time since beginning: {} seconds / last wavelength: {} seconds\n\n".format(wv,time()-start_transit_time,time()-start_wv))
                try:
                    start_wv = time()
                    f = spectral_lcs[i]
                    # Create dictionaries:
                    times, fluxes, fluxes_error= {},{},{}
                    # Save data into those dictionaries:
                    times['sim'], fluxes['sim'] = np.array(t),np.array(f)
                    if spectral_lcs_err is None:
                        fluxes_error['sim'] = np.array([0 for i in range(len(t))])
                    else:
                        fluxes_error['sim'] = np.array(spectral_lcs_err[i])
        
                    priors = {}
                    if juliet_fit_linear_trend:
                        ###Fit linear trend and renormalise data:
                        # Get flux and time values outside of transit:
                        fluxes_out_of_transit = np.concatenate((f[:idx_ini_transit], f[idx_end_transit:]))
                        times_out_of_transit = np.concatenate((t[:idx_ini_transit], t[idx_end_transit:]))
                        
                        
                        ini_slope = (f[-1] - f[0]) / (t[-1] - t[0])
                        ini_y_intercept = f[0] - ini_slope * t[0]
                        
                        
                        def linear_trend(t,*p):
                            """
                            Linear trend y=a*t+b. p[0] is the y-intercept and p[1] the slope
                            """
                            return p[0]+p[1]*t
                        
                        popt,pcov = curve_fit(linear_trend,
                                                times_out_of_transit,
                                                fluxes_out_of_transit,
                                                p0=[ini_y_intercept,ini_slope],
                                                bounds=((0.6,-0.002),(1.4,0.002)),
                                                maxfev=10000)
                        
                        
                        y_intercept,slope = popt
                        
                        # Compute the y values for the straight line using the computed slope and y-intercept
                        ini_straight_line_lc = ini_slope * np.array(t) + ini_y_intercept
                        straight_line_lc = slope * np.array(t) + y_intercept
                        straight_line_lc_out_of_transit = slope * np.array(times_out_of_transit) + y_intercept
                        
                        
                        
                        #Get residuals:
                        residuals_linear_fit = fluxes_out_of_transit - straight_line_lc_out_of_transit
                        
                        #Get fit metrics ([mean(residuals),std(residuals),MSE]):
                        mse_linear_fit = np.sum(residuals_linear_fit**2)/np.shape(residuals_linear_fit)[0]
                        linear_fit_metrics[wv] = [np.mean(residuals_linear_fit),
                                                             np.std(residuals_linear_fit),
                                                             mse_linear_fit]
                        
                        if plot_model:
                            # Plot linear fit and residuals
                            fig1 = plt.figure(1)
                            frame1 = fig1.add_axes((.1,.3,.8,.6))
                            plt.plot(t,ini_straight_line_lc,c='red',alpha=0.5,ls='dashed',label="Initial: {}".format([ini_y_intercept,ini_slope]))
                            plt.plot(t,straight_line_lc,c='purple',alpha=0.5,label="Fitted: {}".format(popt))
                            plt.scatter(t,f,alpha=0.5)
                            plt.scatter(times_out_of_transit,fluxes_out_of_transit,alpha=0.5)
                            
                            plt.tight_layout()
            
            
                            # Plot portion of the lightcurve, axes, etc.:
                            plt.ylabel('Relative flux')
                            plt.tick_params(labelbottom=False)  # Hide x-tick labels from the bottom
                            plt.legend(fontsize=8)
            
                            #ADDING RESIDUALS:
                            frame2=fig1.add_axes((.1,.1,.8,.2))
                            plt.scatter(times_out_of_transit,residuals_linear_fit*1e6,color='pink',s=0.5,label=r'MSE: {:.2g}'.format(mse_linear_fit))
                            plt.axhline(y = 0, linestyle = 'dotted',color='black',linewidth=0.5)
                            plt.xlabel(r'Time (BJD - 2460000)')
                            plt.ylabel(r'residuals (ppm)')
                            #plt.gca().yaxis.set_label_coords(-0.06,0.5)
                            plt.tick_params(axis='both',which='both',direction='in',right=True,top=True)
            
                            plt.legend(loc='best')
                            plt.savefig(juliet_out_folder + '/{}_{}_linear_fit.png'.format(transit_time,wv))
                            plt.show()
            
                        # Renormalise light curve dividing by linear trend:
                        f = f / straight_line_lc
                        fluxes['sim'] = fluxes['sim'] / straight_line_lc
                    
                    
                    
                    ###Define parameters for juliet:
                    # Name of the parameters to be fit:
                    params = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_sim', 'q2_sim', 'ecc_p1', 'omega_p1', 'a_p1', 'mdilution_sim', 'mflux_sim', 'sigma_w_sim'] #sigma_w is in ppm
                    
                    if juliet_dists is None:
                        sys.exit('Please enter the distribution for each parameter to use juliet.')
                    else:
                        dists = juliet_dists
                    
                    if juliet_hyperps is None:
                        sys.exit('Please enter the hyperparameters of the distribution for each parameter to use juliet.')
                    else:
                        if juliet_use_hyperps_per_wv:
                            hyperps = juliet_hyperps[i]
                        else:
                            hyperps = juliet_hyperps
                    
                    # Populate the priors dictionary:
                    for param, dist, hyperp in zip(params, dists, hyperps):
                        priors[param] = {}
                        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp
                    folder = juliet_out_folder+'/{:.2f}_{}_fit'.format(transit_time,wv)
                    if juliet_sampler=='multinest' and len(folder)>=69:
                        sys.exit('Warning: The multinest sampler does not support out_folder paths longer than 69 characters. Please try again with a shorter name or use the argument juliet_create_tmp_symlink=True. Current name: {}'.format(folder))
                    # Load dataset into juliet, save results to a folder:
                    dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
                                          yerr_lc = fluxes_error, ld_laws=juliet_ld_law, out_folder = folder)
        
                    # Fit and absorb results into a juliet.fit object:
                    results = dataset.fit(n_live_points = 300, sampler = juliet_sampler, nthreads = multiprocessing.cpu_count() if juliet_nthreads is None else juliet_nthreads)
        
                    
                    
                    #Get residuals:
                    residuals = dataset.data_lc['sim'] - results.lc.evaluate('sim')
                    
                    #Get fit metrics ([mean(residuals),std(residuals),MSE]):
                    mse = np.sum(residuals**2)/np.shape(residuals)[0]
                    fit_metrics[wv] = [np.mean(residuals),
                                       np.std(residuals),
                                       mse]
                    
                    
                    
                    #Save fit parameters, including transit depth and LDC:
                    # Store posterior samples for q1, q2 and p_p1:
                    #q1, q2, p_p1 = results.posteriors['posterior_samples']['q1_sim'],\
                    #               results.posteriors['posterior_samples']['q2_sim'],\
                    #               results.posteriors['posterior_samples']['p_p1']
                    # Store posterior samples for p_p1:
                    p_p1 = results.posteriors['posterior_samples']['p_p1']
                    if juliet_return_posteriors:
                        juliet_posteriors.append(results.posteriors['posterior_samples'])
                    # Get median from quantiles:
                    #q1 = juliet.utils.get_quantiles(q1)[0]
                    #q2 = juliet.utils.get_quantiles(q2)[0]
                    p_p1 = juliet.utils.get_quantiles(p_p1)[0] #R_p/R_s
                    # Save transit depth and LDC:
                    #ldc[wv] = [q1,q2]
                    depth[wv] = p_p1**2 #(R_p/R_s)^2
                    
                    depth_otherdef[wv] = (np.mean([f[idx_ini_transit],f[idx_end_transit]])-np.min(f))/np.mean([f[idx_ini_transit],f[idx_end_transit]])
                    
                    
                    
                    
                    
                    if plot_model:
                        # Plot the data:
                        fig1 = plt.figure(1)
                        frame1 = fig1.add_axes((.1,.3,.8,.6))
                        plt.errorbar(dataset.times_lc['sim'], dataset.data_lc['sim'], \
                                     yerr = dataset.errors_lc['sim'], fmt = '.', alpha = 0.1, label="Data")
                        
                        # Plot the model:
                        plt.plot(dataset.times_lc['sim'], results.lc.evaluate('sim'),label="Juliet fit")
                        plt.tight_layout()
            
            
                        # Plot axes, etc.:
                        #plt.xlabel('Time (BJD - 2460000)')
                        plt.ylabel('Relative flux')
                        plt.tick_params(labelbottom=False)  # Hide x-tick labels from the bottom
                        plt.legend()
                        
                        #ADDING RESIDUALS:
                        frame2=fig1.add_axes((.1,.1,.8,.2))
                        plt.scatter(dataset.times_lc['sim'],residuals*1e6,color='pink',s=0.5,label=r'MSE: {:.2g}'.format(mse))
                        plt.axhline(y = 0, linestyle = 'dotted',color='black',linewidth=0.5)
                        plt.xlabel(r'Time')
                        plt.ylabel(r'residuals (ppm)')
                        #plt.gca().yaxis.set_label_coords(-0.06,0.5)
                        plt.tick_params(axis='both',which='both',direction='in',right=True,top=True)
                        
                        plt.legend(loc='best')
                        plt.savefig(juliet_out_folder+'/{}_{}_fit.png'.format(transit_time,wv))
                        plt.show()
                        
                        
                        
                        
                        
                        #Plot corner plots
                        if juliet_true_values is not None:
                            # Collect posterior samples for parameters that are not fixed
                            samples = []
                            labels = []
                            true_values = juliet_true_values
                            truths = []
                            for i in range(len(params)):
                                param_name = params[i]
                                try:
                                    param_samples = results.posteriors['posterior_samples'][param_name]
                                    samples.append(param_samples)
                                    labels.append(param_name)
                                    truths.append(true_values[i])
                                except KeyError:
                                    print(f"No posterior samples for '{param_name}', skipping...")
                
                            # Plot corner plot only if there are samples for at least two parameters
                            if len(samples) >= 2:
                                print('truths:{}'.format(truths))
                                print('len of truths: {}'.format(len(truths)))
                                print('len of samples: {}'.format(len(samples)))
                                fig = corner.corner(np.array(samples).T, labels=labels, truths=truths, #truth_color='red',
                                                    quantiles=[0.16, 0.5, 0.84],show_titles=True,
                                                    title_fmt='.6f',
                                                    title_kwargs={"fontsize": 12},
                                                    label_kwargs={"fontsize": 12})
                                
                                ndim = len(labels)
                                axes = np.array(fig.axes).reshape((ndim, ndim))
                                
                                for xi in range(ndim):
                                    ax = axes[-1, xi]  # Target the axes in the lowest row
                                    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                                    
                                    ax.tick_params(axis='x', which='major', labelsize=9)  # Adjust the font size of x-tick labels as needed
                                    
                                    ax.xaxis.set_label_coords(0.5, -0.5)
                                for yi in range(ndim):
                                    ax = axes[yi, 0]  # Target the axes in the leftmost column
                                    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                                    
                                    ax.tick_params(axis='y', which='major', labelsize=9)  # Adjust the font size of x-tick labels as needed
                                    
                                    ax.yaxis.set_label_coords(-0.5, 0.5)
                                
                                #plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
                                #plt.tick_params(axis='x', which='major', pad=10)  # Increase padding between tick labels and axis
                                plt.savefig(juliet_out_folder+'/{}_{}_cornerplot.png'.format(transit_time,wv))
                                plt.show()
                            else:
                                print("Insufficient parameters with posterior samples to plot corner plot.")
                    
                    if not juliet_output_fits: #If we don't want the output files from juliet, delete them:
                        if os.path.exists(folder):
                            shutil.rmtree(folder)
                        
                        
                except Exception as e:
                    # Handle the exception
                    error_info = {
                        "transit_time": transit_time,
                        "wv": wv,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    errors_juliet_fitting.append(error_info)
                    print(f"Error on transit_time {transit_time} and wv {wv}: {type(e).__name__} - {e}")
            
            
            try:
                if plot_fit_metrics:
                    #Plot all LINEAR fit metrics for this transit_time:
                    #  Mean of residuals
                    plt.plot(wvs,[linear_fit_metrics[wv][0] for wv in wvs])
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("mean(residuals)")
                    plt.savefig(juliet_out_folder + '/linear_fit_transit_time_{}_mean_residuals.png'.format(transit_time))
                    plt.show()
                    #  Std of residuals
                    plt.plot(wvs,[linear_fit_metrics[wv][1] for wv in wvs])
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("std(residuals)")
                    plt.savefig(juliet_out_folder + '/linear_fit_transit_time_{}_std_residuals.png'.format(transit_time))
                    plt.show()
                    #  MSE
                    plt.plot(wvs,[linear_fit_metrics[wv][2] for wv in wvs])
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("MSE")
                    plt.savefig(juliet_out_folder + '/linear_fit_transit_time_{}_MSE.png'.format(transit_time))
                    plt.show()
                    
                    #Plot all fit metrics for this transit_time:
                    #  Mean of residuals
                    plt.plot(wvs,[fit_metrics[wv][0] for wv in wvs])
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("mean(residuals)")
                    plt.savefig(juliet_out_folder + '/fit_transit_time_{}_mean_residuals.png'.format(transit_time))
                    plt.show()
                    #  Std of residuals
                    plt.plot(wvs,[fit_metrics[wv][1] for wv in wvs])
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("std(residuals)")
                    plt.savefig(juliet_out_folder + '/fit_transit_time_{}_std_residuals.png'.format(transit_time))
                    plt.show()
                    #  MSE
                    plt.plot(wvs,[fit_metrics[wv][2] for wv in wvs])
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("MSE")
                    plt.savefig(juliet_out_folder + '/fit_transit_time_{}_MSE.png'.format(transit_time))
                    plt.show()
                    
                if plot_model:
                    #Plot transit depth vs wavelength for this transit_time:
                    plt.plot(wvs,[1000000*depth[wv] for wv in wvs],label="Data")
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("Transit depth (ppm)")
                    plt.legend()
                    plt.savefig(juliet_out_folder + '/transit_depth_vs_lambda_transit_time_{}_ppm.png'.format(transit_time))
                    plt.show()
                    
                    #Plot transit depth otherdef vs wavelength for this transit_time:
                    plt.plot(wvs,[1000000*depth_otherdef[wv] for wv in wvs],label="Data")
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("Transit depth other def (ppm)")
                    plt.legend()
                    plt.savefig(juliet_out_folder + '/transit_depth_otherdef_vs_lambda_transit_time_{}_ppm.png'.format(transit_time))
                    plt.show()
                
                
                
                #### SAVING DATA:
                output_base_filename = juliet_out_folder + "/transit_depth_vs_wavelength_data_transit_time_{}/".format(transit_time)
                os.makedirs(os.path.dirname(output_base_filename), exist_ok=True) #Check if directory exists, if not then create it
                with open(output_base_filename+"transit_depth_vs_lambda.txt","w") as f:
                    results =  np.array([wvs,[depth[wv] for wv in wvs]])
                    np.savetxt(f, results, fmt='%.10f', delimiter="\t")
                df = pd.DataFrame(list(fit_metrics.items()), columns=['Key', 'Value'])
                df.to_csv(output_base_filename+"fit_metrics.csv", index=False, sep='\t')
                df = pd.DataFrame(list(linear_fit_metrics.items()), columns=['Key', 'Value'])
                df.to_csv(output_base_filename+"linear_fit_metrics.csv", index=False, sep='\t')
        
        
                
            except Exception as e:
                # Handle the exception
                error_info = {
                    "transit_time": transit_time,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                print(f"Error on transit_time {transit_time}: {type(e).__name__} - {e}")
                #traceback.print_exc()
            
            
            # After the loop, you can inspect the list of errors
            print("\nList of Errors Juliet Fitting:")
            for error in errors_juliet_fitting:
                print(f"Iteration {error['transit_time']},{error['wv']}: {error['error_type']} - {error['error_message']}")

            


            results_depths['juliet'] = (np.array([depth[wv] for wv in wvs]))
            
            if juliet_create_tmp_symlink:
                if juliet_original_out_folder != juliet_out_folder:
                    def remove_symlink(symlink_path):
                        """
                        Remove the symlink without affecting the original directory.
                    
                        Parameters:
                            symlink_path (str): The path of the symlink to remove.
                        """
                        if os.path.islink(symlink_path):
                            os.unlink(symlink_path)
                            print(f"Removed symlink: {symlink_path}")
                        else:
                            print(f"No symlink found at: {symlink_path}")
                    
                    remove_symlink(juliet_out_folder)
                    juliet_out_folder = juliet_original_out_folder
            
        
        if use_batman_curve_fit:
            fit_metrics = {}
            linear_fit_metrics = {}
            if batman_output_data or batman_plot_lc_fits or batman_plot_linear_fits:
                if batman_out_folder is None:
                    sys.exit('Please enter an output folder for batman.')
                else:
                    def ensure_folder_exists(folder_path):
                        # Check if the folder exists
                        if not os.path.exists(folder_path):
                            # If it does not exist, create it
                            os.makedirs(folder_path)
                            print(f"Folder created: {folder_path}")
                        else:
                            print(f"Folder already exists: {folder_path}")
                    ensure_folder_exists(batman_out_folder)
            
            
            if batman_fit_white_lc:
                if white_lc is None:
                    if not batman_compute_white_light_curve:
                        sys.exit("Error: The white light curve (integrated spectrum) is needed to use the batman method if batman_fit_white_lc=True. Either provide it or set batman_compute_white_light_curve=True and provide batman_spec and batman_spec_wv.")
                    else:
                        if batman_spec is None:
                            sys.exit("Error: batman_spec is None. The white light curve (integrated spectrum) is needed to use the batman method if batman_fit_white_lc=True. Either provide it or set batman_compute_white_light_curve=True and provide batman_spec and batman_spec_wv.")
                        elif batman_spec_wv is None:
                            sys.exit("Error: batman_spec_wv is None. The white light curve (integrated spectrum) is needed to use the batman method if batman_fit_white_lc. Either provide it or set batman_compute_white_light_curve=True and provide batman_spec and batman_spec_wv.")
                        else:
                            white_lc,white_lc_err = self.compute_white_light_curve_from_spec(t=t, spec=batman_spec, wv=batman_spec_wv, vectorise=True, normalize_by_max=True)
                #if white_lc_err is None:
                #    white_lc_err = np.array([0 for i in range(len(t))])
                
                ######################################### FIT WHITE LIGHT CURVE #########################################
                
                #Fit linear trend for the white light curve:
                if batman_fit_linear_trend:
                    ###Fit linear trend and renormalise data:
                    # Get flux and time values outside of transit:
                    f = white_lc
                    fluxes_out_of_transit = np.concatenate((f[:idx_ini_transit], f[idx_end_transit:]))
                    times_out_of_transit = np.concatenate((t[:idx_ini_transit], t[idx_end_transit:]))
                    
                    
                    ini_slope = (f[-1] - f[0]) / (t[-1] - t[0])
                    ini_y_intercept = f[0] - ini_slope * t[0]
                    
                    
                    def linear_trend(t,*p):
                        """
                        Linear trend y=a*t+b. p[0] is the y-intercept and p[1] the slope
                        """
                        return p[0]+p[1]*t
                    
                    popt,pcov = curve_fit(linear_trend,
                                            times_out_of_transit,
                                            fluxes_out_of_transit,
                                            p0=[ini_y_intercept,ini_slope],
                                            bounds=((ini_y_intercept-2,ini_slope-0.100),(ini_y_intercept+2,ini_slope+0.100)),#((0.4,-0.020),(1.6,0.020)),#bounds=((0.6,-0.002),(1.4,0.002)),
                                            maxfev=10000)
                    
                    
                    y_intercept,slope = popt
                    
                    # Compute the y values for the straight line using the computed slope and y-intercept
                    ini_straight_line_lc = ini_slope * np.array(t) + ini_y_intercept
                    straight_line_lc = slope * np.array(t) + y_intercept
                    straight_line_lc_out_of_transit = slope * np.array(times_out_of_transit) + y_intercept
                    
                    
                    
                    #Get residuals:
                    residuals_linear_fit = fluxes_out_of_transit - straight_line_lc_out_of_transit
                    
                    #Get fit metrics ([mean(residuals),std(residuals),MSE]):
                    mse_linear_fit = np.sum(residuals_linear_fit**2)/np.shape(residuals_linear_fit)[0]
                    linear_fit_metrics_white_lc = [np.mean(residuals_linear_fit),
                                                         np.std(residuals_linear_fit),
                                                         mse_linear_fit]
                    
                    if batman_plot_linear_fits:
                        # Plot linear fit and residuals
                        fig1 = plt.figure(1)
                        frame1 = fig1.add_axes((.1,.3,.8,.6))
                        plt.plot(t,ini_straight_line_lc,c='red',alpha=0.5,ls='dashed',label="Initial: {}".format([ini_y_intercept,ini_slope]))
                        plt.plot(t,straight_line_lc,c='purple',alpha=0.5,label="Fitted: {}".format(popt))
                        plt.scatter(t,f,alpha=0.5)
                        plt.scatter(times_out_of_transit,fluxes_out_of_transit,alpha=0.5)
                        
                        plt.tight_layout()
        
        
                        # Plot portion of the lightcurve, axes, etc.:
                        plt.ylabel('Relative flux')
                        plt.tick_params(labelbottom=False)  # Hide x-tick labels from the bottom
                        plt.legend(fontsize=8)
        
                        #ADDING RESIDUALS:
                        frame2=fig1.add_axes((.1,.1,.8,.2))
                        plt.scatter(times_out_of_transit,residuals_linear_fit*1e6,color='pink',s=0.5,label=r'MSE: {:.2g}'.format(mse_linear_fit))
                        plt.axhline(y = 0, linestyle = 'dotted',color='black',linewidth=0.5)
                        plt.xlabel(r'Time (BJD - 2460000)')
                        plt.ylabel(r'residuals (ppm)')
                        #plt.gca().yaxis.set_label_coords(-0.06,0.5)
                        plt.tick_params(axis='both',which='both',direction='in',right=True,top=True)
        
                        plt.legend(loc='best')
                        plt.savefig(batman_out_folder + '/{}_white_lc_linear_fit.png'.format(transit_time))
                        plt.show()
        
                    # Renormalise light curve dividing by linear trend:
                    f = f / straight_line_lc
                    white_lc = white_lc / straight_line_lc
                
                
                
                
                
                
                # initialize batman model with known system parameters
                
                #global params
                params = batman.TransitParams()       # object to store transit parameters
                if batman_ini_pars is None: #if no initial parameters are provided, initialise them according to those in the conf file.
                    # limb-darkening coefficients
                    u1, u2 = 0.5, 0.5 #0.088, 0.15
                    u3, u4 = 0.5, 0.5
                    
                    if(self.planet_esinw==0 and self.planet_ecosw==0):
                       ecc=0
                       omega=90  * np.pi / 180.
                    else:
                       ecc=np.sqrt(self.planet_esinw**2+self.planet_ecosw**2)
                       omega=np.arctan2(self.planet_esinw,self.planet_ecosw)
                    
                    params.inc = inclination #orbital inclination (in degrees)
                    params.ecc = ecc                        # eccentricity
                    params.w = omega*180/np.pi              # longitude of periastron (in degrees)
                    params.a = self.planet_semi_major_axis                     # semi-major axis
                    params.per = self.planet_period                 # orbital period
                    params.rp =  self.planet_radius                  # planetary radius 
                    params.t0 = transit_time                      # time of mid-transit
                    params.limb_dark = batman_ld_law #'quadratic'        # limb-darkening law
                    if batman_ld_law == 'linear':
                        params.u = [u1] # limb-darkening coefficients
                        u_bounds = ([0], [1])
                    elif batman_ld_law == 'quadratic':
                        params.u = [u1, u2]                   # limb-darkening coefficients
                        u_bounds = ([0, 0], [1, 1])
                    elif batman_ld_law == 'nonlinear':
                        params.u = [u1, u2, u3, u4]                   # limb-darkening coefficients
                        u_bounds = ([0, 0, 0, 0], [1, 1, 1, 1])
                    elif batman_ld_law == 'three-parameter': #This is equivalent to the 'nonlinear' law from batman but with c1=0, it's the Claret four parameter law with c1=0. This is named "Kipping–Sing limb darkening law" on this paper and is recommended especially for high impact parameters: https://arxiv.org/pdf/2410.1861
                        params.limb_dark = 'nonlinear'
                        params.u = [0, u2, u3, u4]
                        u_bounds = ([0, 0, 0], [1, 1, 1])
                    else:
                        raise ValueError(f"Unsupported limb-darkening law: {batman_ld_law}")
                else: #if initial parameters are provided, use them
                    params.inc,params.ecc,params.w,params.a,params.per,params.rp,params.t0,params.limb_dark,params.u = batman_ini_pars
                    if batman_ld_law == 'linear':
                        u_bounds = ([0], [1])
                    elif batman_ld_law == 'quadratic':
                        u_bounds = ([0, 0], [1, 1])
                    elif batman_ld_law == 'nonlinear':
                        u_bounds = ([0, 0, 0, 0], [1, 1, 1, 1])
                    elif batman_ld_law == 'three-parameter': #This is equivalent to the 'nonlinear' law from batman but with c1=0, it's the Claret four parameter law with c1=0. This is named "Kipping–Sing limb darkening law" on this paper and is recommended especially for high impact parameters: https://arxiv.org/pdf/2410.1861
                        u_bounds = ([0, 0, 0], [1, 1, 1])
                    else:
                        raise ValueError(f"Unsupported limb-darkening law: {batman_ld_law}")
                    
                if batman_fit_period_for_white_lc:
                    lower_bounds = [0, t[0], 0, 0, 0]
                    upper_bounds = [1, t[-1], np.inf, 90, np.inf]
                else:
                    lower_bounds = [0, t[0], 0, 0]
                    upper_bounds = [1, t[-1], np.inf, 90]
                
                lower_bounds += list(u_bounds[0])
                upper_bounds += list(u_bounds[1])
                
                model = None
                if not batman_model_based_3point:
                    # create batman model
                    model = batman.TransitModel(params, t)
    
                if batman_fit_period_for_white_lc:
                    def lc_model(exp_times, rp, t0, a, inc, per, *u):
                    
                        """
                        
                        Function to define the light curve model
                    
                        """
                        
                        params.rp = rp
                        params.t0 = t0
                        params.a = a
                        params.inc = inc
                        params.per = per
                        if batman_ld_law == 'three-parameter':
                            params.u = [0, u[0], u[1], u[2]]  # Fix u1 = 0
                        else:
                            params.u = u
                        
                        if batman_use_kipping_ldc_parametrisation:
                            if batman_ld_law == 'three-parameter':
                                third	= 1./3. #0.333333333333333
                                twopi	= 2*np.pi #6.283185307179586
                                P1	= 4.500841772313891
                                P2	= 17.14213562373095
                                Q1	= 7.996825477806030
                                Q2	= 8.566161603278331
                                
                                alpha_t = u[0]
                                alpha_h = u[1]
                                alpha_r = u[2]
                                c_2 = (alpha_h**third)*( P1 + 0.25*np.sqrt(alpha_r)*( -6.0*np.cos(twopi*alpha_t) + P2*np.sin(twopi*alpha_t) ) )
                                c_3 = (alpha_h**third)*( -Q1 - Q2*np.sqrt(alpha_r)*np.sin(twopi*alpha_t) )
                                c_4 = (alpha_h**third)*( P1 + 0.25*np.sqrt(alpha_r)*( 6.0*np.cos(twopi*alpha_t) + P2*np.sin(twopi*alpha_t) ) )
                                params.u = [0, c_2, c_3, c_4]
                            elif batman_ld_law == 'quadratic':
                                q1 = u[0]
                                q2 = u[1]
                                params.u = [2*np.sqrt(q1)*q2,np.sqrt(q1)*(1-2*q2)]
                            else:
                                raise ValueError("Kipping LDC parametrisation is not yet implemented for LD laws different than quadratic or three-parameter. batman_use_kipping_ldc_parametrisation=True yet batman_ld_law is neither quadratic nor three-parameter.")
                        
                        if batman_model_based_3point:
                            # create batman model
                            model1 = batman.TransitModel(params, exp_times)
                            light_curve = model1.light_curve(params)
                        else:
                            if model is not None:
                                light_curve = model.light_curve(params)  # Proceed only if model is defined
                            else:
                                # Handle the case where model wasn't initialized
                                raise ValueError("Model was not initialized. Check your batman_model_based_3point condition.")
                        
                        return light_curve
                else:
                    def lc_model(exp_times, rp, t0, a, inc, *u):
                    
                        """
                        
                        Function to define the light curve model
                    
                        """
                        
                        params.rp = rp
                        params.t0 = t0
                        params.a = a
                        params.inc = inc
                        if batman_ld_law == 'three-parameter':
                            params.u = [0, u[0], u[1], u[2]]  # Fix u1 = 0
                        else:
                            params.u = u
                        
                        if batman_use_kipping_ldc_parametrisation:
                            if batman_ld_law == 'three-parameter':
                                third	= 1./3. #0.333333333333333
                                twopi	= 2*np.pi #6.283185307179586
                                P1	= 4.500841772313891
                                P2	= 17.14213562373095
                                Q1	= 7.996825477806030
                                Q2	= 8.566161603278331
                                
                                alpha_t = u[0]
                                alpha_h = u[1]
                                alpha_r = u[2]
                                c_2 = (alpha_h**third)*( P1 + 0.25*np.sqrt(alpha_r)*( -6.0*np.cos(twopi*alpha_t) + P2*np.sin(twopi*alpha_t) ) )
                                c_3 = (alpha_h**third)*( -Q1 - Q2*np.sqrt(alpha_r)*np.sin(twopi*alpha_t) )
                                c_4 = (alpha_h**third)*( P1 + 0.25*np.sqrt(alpha_r)*( 6.0*np.cos(twopi*alpha_t) + P2*np.sin(twopi*alpha_t) ) )
                                params.u = [0, c_2, c_3, c_4]
                            elif batman_ld_law == 'quadratic':
                                q1 = u[0]
                                q2 = u[1]
                                params.u = [2*np.sqrt(q1)*q2,np.sqrt(q1)*(1-2*q2)]
                            else:
                                raise ValueError("Kipping LDC parametrisation is not yet implemented for LD laws different than quadratic or three-parameter. batman_use_kipping_ldc_parametrisation=True yet batman_ld_law is neither quadratic nor three-parameter.")
                        
                        if batman_model_based_3point:
                            # create batman model
                            model1 = batman.TransitModel(params, exp_times)
                            light_curve = model1.light_curve(params)
                        else:
                            if model is not None:
                                light_curve = model.light_curve(params)  # Proceed only if model is defined
                            else:
                                # Handle the case where model wasn't initialized
                                raise ValueError("Model was not initialized. Check your batman_model_based_3point condition.")
                    
                        return light_curve
                
                
                if batman_fit_period_for_white_lc:
                    p0 = [params.rp, params.t0, params.a, params.inc, params.per]
                else:
                    p0 = [params.rp, params.t0, params.a, params.inc]
                
                if batman_ld_law =='three-parameter':
                    p0 += params.u[1:]
                else:
                    p0 += params.u
                
                if white_lc_err is None:
                    # perform light curve fitting
                    popt, pcov = optimize.curve_fit(lc_model, 
                                        t, 
                                        white_lc,
                                        p0 = p0,#, params.u[0], params.u[1]], 
                                        bounds=(lower_bounds, upper_bounds),
                                        maxfev = 4000,
                                        ftol = 1e-10,
                                        xtol = 1e-10,
                                        method = 'trf',#potser canviar a trf i posar bounds per rp,u.
                                        absolute_sigma=True)
                else:
                    # perform light curve fitting
                    popt, pcov = optimize.curve_fit(lc_model, 
                                        t, 
                                        white_lc, 
                                        sigma = white_lc_err, 
                                        p0 = p0, #, params.u[0], params.u[1]], 
                                        bounds=(lower_bounds, upper_bounds),
                                        maxfev = 4000,
                                        ftol = 1e-10,
                                        xtol = 1e-10,
                                        method = 'trf',#potser canviar a trf i posar bounds per rp,u.
                                        absolute_sigma=True)
            
                # estimate parameter errors from covariance matrix
                perr = np.sqrt(np.diag(pcov))
                
                # get fitted parameters
                transit_depth = popt[0]**2
                rp_fitted = popt[0]
                t0_fitted = popt[1]
                a_fitted = popt[2]
                inc_fitted = popt[3]
                if batman_fit_period_for_white_lc:
                    per_fitted = popt[4]
                    u_fitted = [0]+list(popt[5:]) if batman_ld_law == 'three-parameter' else popt[5:]
                else:
                    u_fitted = [0]+list(popt[4:]) if batman_ld_law == 'three-parameter' else popt[4:]
                
                #Get residuals: 
                residuals = white_lc - lc_model(t, *popt)
                
                #Get fit metrics ([mean(residuals),std(residuals),MSE]):
                mse = np.sum(residuals**2)/np.shape(residuals)[0]
                fit_metrics_white_lc = [np.mean(residuals),
                                   np.std(residuals),
                                   mse]
                
                if plot_model:
                    # print the fitted parameters
                    print('rp: {:.5f}'.format(popt[0]))
                    print('Transit depth: {:.5f} %'.format(transit_depth*100))
                    print('Time of mid-transit: {:.5f} days'.format(t0_fitted))
                    print('Semi-major axis: {:.5f}'.format(a_fitted))
                    print('Inclination: {:.5f} deg'.format(inc_fitted))
                    #print('LD coeffs: u1={:.5f},u2={:.5f} deg'.format(u_fitted[0],u_fitted[1]))
                    if batman_fit_period_for_white_lc:
                        print('Planet period: {:.5f} days'.format(per_fitted))
                    print('LD coeffs: ' + ', '.join([f'u{i+1}={u_fitted[i]:.5f}' for i in range(len(u_fitted))]))
                    
                    if batman_fit_period_for_white_lc:
                        legend_text = (
                            f"$r_p$: {popt[0]:.5f}\n"
                            f"Transit depth: {transit_depth*1e6:.1f} ppm\n"
                            f"Mid-transit time: {t0_fitted:.5f} days\n"
                            f"Semi-major axis: {a_fitted:.5f}\n"
                            f"Inclination: {inc_fitted:.5f}°\n"
                            f"Planet period: {per_fitted:.5f} days\n"
                            + '\n'.join([f"LD coeffs: $u_{i+1}$={u_fitted[i]:.5f}" for i in range(len(u_fitted))])
                        )
                    else:
                        legend_text = (
                            f"$r_p$: {popt[0]:.5f}\n"
                            f"Transit depth: {transit_depth*1e6:.1f} ppm\n"
                            f"Mid-transit time: {t0_fitted:.5f} days\n"
                            f"Semi-major axis: {a_fitted:.5f}\n"
                            f"Inclination: {inc_fitted:.5f}°\n"
                            + '\n'.join([f"LD coeffs: $u_{i+1}$={u_fitted[i]:.5f}" for i in range(len(u_fitted))])
                        )
                        #f"LD coeffs: $u_1$={u_fitted[0]:.5f}, $u_2$={u_fitted[1]:.5f}"
                    #)
                
                    # plot the white-light curve with the fitted model
                    plt.figure(figsize = (20, 7))
                    plt.scatter(t, white_lc, color = 'indianred', alpha = 0.5)
                    plt.plot(t, lc_model(t, *popt), color = 'black', alpha = 0.5, linewidth = 2, label=legend_text)
                    plt.xlabel('Time (days)')
                    plt.ylabel('Normalized flux')
                    plt.legend()
                    plt.savefig(batman_out_folder + '/{}_white_lc_fit.png'.format(transit_time))
                    plt.show()
                
                    # calculate the residuals
                    res = white_lc - lc_model(t, *popt)
                    
                    # plot residuals
                    plt.figure(figsize = (20, 3))
                    if white_lc_err is not None:
                        plt.errorbar(t, 1e6*res, yerr = 1e6*white_lc_err, fmt = 'o', color = 'indianred', alpha = 0.5)
                    else:
                        plt.scatter(t, 1e6*res, color = 'indianred', alpha = 0.5)
                    plt.axhline(1e6*np.mean(res))
                    plt.axhline(1e6*(np.mean(res) + np.std(res)), linestyle = '--')
                    plt.axhline(1e6*(np.mean(res) - np.std(res)), linestyle = '--')
                    plt.xlabel('Time (days)')
                    plt.ylabel('Residuals (ppm)')
                    plt.savefig(batman_out_folder + '/{}_white_lc_fit_residuals.png'.format(transit_time))
                    plt.show()
                
                    # Print photon noise and residuals standard deviation
                    if white_lc_err is not None:
                        print('Photon noise: {:.1f} ppm'.format(np.mean(white_lc_err)*1e6))
                    print('Residuals std: {:.1f} ppm\n\n\n\n\n'.format(np.std(res)*1e6))
            
            
            ######################################### FIT SPECTRAL LIGHT CURVES #########################################
            # initialize batman model with known system parameters
            
            if batman_u_fitted_list_to_use is not None and batman_fit_params['u']:
                raise ValueError("batman_u_fitted_list_to_use is not None, yet batman_fit_params['u'] is True.")
            if batman_a_fitted_list_to_use is not None and batman_fit_params['a']:
                raise ValueError("batman_a_fitted_list_to_use is not None, yet batman_fit_params['a'] is True.")
            if batman_b_to_fix is not None and batman_fit_params['inc']:
                raise ValueError("batman_b_to_fix is not None, yet batman_fit_params['inc'] is True.")
            
            #global params
            params = batman.TransitParams()       # object to store transit parameters
            if batman_ini_pars is None: #if no initial parameters are provided, initialise them according to those in the conf file and/or those fitted previously
                if batman_fit_white_lc: #initialise according to params fitted on the white_lc and those in the conf file
                    params.inc = inc_fitted #orbital inclination (in degrees)
                    params.ecc = ecc                        # eccentricity
                    params.w = omega*180/np.pi              # longitude of periastron (in degrees)
                    params.a = a_fitted                     # semi-major axis
                    if batman_fit_period_for_white_lc:
                        params.per = per_fitted
                    else:
                        params.per = self.planet_period                 # orbital period
                    params.rp =  rp_fitted                  # planetary radius 
                    params.t0 = t0_fitted                      # time of mid-transit
                    params.limb_dark = batman_ld_law #'quadratic'        # limb-darkening law
                    params.u = u_fitted                   # limb-darkening coefficients
                else: #initialise according to params in the conf file
                    # limb-darkening coefficients
                    u1, u2 = 0.5, 0.5 #0.15, 0.15#0.088, 0.15
                    u3, u4 = 0.5, 0.5
                    
                    if(self.planet_esinw==0 and self.planet_ecosw==0):
                       ecc=0
                       omega=90  * np.pi / 180.
                    else:
                       ecc=np.sqrt(self.planet_esinw**2+self.planet_ecosw**2)
                       omega=np.arctan2(self.planet_esinw,self.planet_ecosw)
                    
                    params.inc = inclination #orbital inclination (in degrees)
                    params.ecc = ecc                        # eccentricity
                    params.w = omega*180/np.pi              # longitude of periastron (in degrees)
                    params.a = self.planet_semi_major_axis                     # semi-major axis
                    params.per = self.planet_period                 # orbital period
                    params.rp =  self.planet_radius                  # planetary radius 
                    params.t0 = transit_time                      # time of mid-transit
                    params.limb_dark = batman_ld_law #'quadratic'        # limb-darkening law
                    if batman_ld_law == 'linear':
                        params.u = [u1] # limb-darkening coefficients
                    elif batman_ld_law == 'quadratic':
                        params.u = [u1, u2]                   # limb-darkening coefficients
                    elif batman_ld_law == 'nonlinear':
                        params.u = [u1, u2, u3, u4]                   # limb-darkening coefficients
                    elif batman_ld_law == 'three-parameter': #This is equivalent to the 'nonlinear' law from batman but with c1=0, it's the Claret four parameter law with c1=0. This is named "Kipping–Sing limb darkening law" on this paper and is recommended especially for high impact parameters: https://arxiv.org/pdf/2410.1861
                        params.u = [0, u2, u3, u4]
                    else:
                        raise ValueError(f"Unsupported limb-darkening law: {batman_ld_law}")
                
            else: #if initial parameters are provided, use them
                params.inc,params.ecc,params.w,params.a,params.per,params.rp,params.t0,params.limb_dark,params.u = batman_ini_pars
                
                if batman_fit_white_lc: #set parameters fitted previously:
                    params.inc = inc_fitted #orbital inclination (in degrees)
                    params.a = a_fitted                     # semi-major axis
                    params.rp =  rp_fitted                  # planetary radius 
                    params.t0 = t0_fitted                      # time of mid-transit
                    params.u = u_fitted
                
            
            
            
            # Set the number of limb-darkening coefficients based on the selected law
            if batman_ld_law == 'linear':
                u_initial = [params.u[0]]
                u_bounds = ([0], [1])
            elif batman_ld_law == 'quadratic':
                u_initial = [params.u[0], params.u[1]]
                u_bounds = ([0, 0], [1, 1])
            elif batman_ld_law == 'nonlinear':
                u_initial = [params.u[0], params.u[1], params.u[2], params.u[3]]
                u_bounds = ([0, 0, 0, 0], [1, 1, 1, 1])
            elif batman_ld_law == 'three-parameter':
                params.limb_dark = 'nonlinear'
                u_initial = [params.u[1], params.u[2], params.u[3]]
                u_bounds = ([0, 0, 0], [1, 1, 1])
            else:
                raise ValueError(f"Unsupported limb-darkening law: {batman_ld_law}")
            
            model = None
            if not batman_model_based_3point:
                # create batman model
                model = batman.TransitModel(params, t)
                #print("Model initialized with batman.TransitModel:", model)
            else:
                model = None
                #print("Model not initialised.", model)
                
            # Set initial values for the parameters
            initial_vals = {
                'rp': params.rp,
                'a': params.a,
                't0': params.t0,
                'inc': params.inc,
                'per': params.per,
                'u': u_initial
            }
    
            # Define bounds for the parameters
            bounds = {
                'rp': (0, 1),
                'a': (0, np.inf),
                't0': (t[0], t[-1]),
                'inc': (0, 90),
                'per': (0,np.inf),
                'u': u_bounds
            }
            
            # Filter the parameters to fit based on `fit_params`
            parameters_to_fit = [param for param, to_fit in batman_fit_params.items() if to_fit]
            initial_guess = [initial_vals[param] for param in parameters_to_fit if param != 'u']
            if 'u' in parameters_to_fit:
                initial_guess += list(u_initial)
            #initial_guess = [initial_vals[param] for param in parameters_to_fit]
            lower_bounds = [bounds[param][0] for param in parameters_to_fit if param != 'u'] # else bounds['u'][0] for param in parameters_to_fit]
            if 'u' in parameters_to_fit:
                lower_bounds += list(bounds['u'][0])
            upper_bounds = [bounds[param][1] for param in parameters_to_fit if param != 'u']
            if 'u' in parameters_to_fit:
                upper_bounds += list(bounds['u'][1])
            
            # # Filter the parameters to fit based on `fit_params`
            # parameters_to_fit = [param for param, to_fit in batman_fit_params.items() if to_fit]
            # initial_guess = [rp_fitted, a_fitted, t0_fitted, *u_initial]
            # #initial_guess = [initial_vals[param] for param in parameters_to_fit]
            # lower_bounds = [bounds[param][0] for param in parameters_to_fit if param != 'u' ]+list(bounds['u'][0]) # else bounds['u'][0] for param in parameters_to_fit]
            # upper_bounds = [bounds[param][1] for param in parameters_to_fit if param != 'u' ]+list(bounds['u'][1]) # else bounds['u'][1] for param in parameters_to_fit]
            # #lower_bounds = [bounds[param][0] if param != 'u' else bounds['u'][0] for param in parameters_to_fit]
            # #upper_bounds = [bounds[param][1] if param != 'u' else bounds['u'][1] for param in parameters_to_fit]


            if batman_model_based_3point:
                model_based_depths = []
            
            
            #fit the light curve model to each spectroscopic light curve separately
            lc_depths = []
            lc_depths_err = []
            
            if plot_model:
                # Create a colormap
                cmap = plt.get_cmap('viridis')
                norm = plt.Normalize(vmin=min(wvs), vmax=max(wvs))
                
            u_fitted_list = []
            a_fitted_list = []
            for j,wv in enumerate(wvs):
                
                #Fit linear trend for each spectral light curve:
                if batman_fit_linear_trend:
                    ###Fit linear trend and renormalise data:
                    # Get flux and time values outside of transit:
                    f = spectral_lcs[j]
                    fluxes_out_of_transit = np.concatenate((f[:idx_ini_transit], f[idx_end_transit:]))
                    times_out_of_transit = np.concatenate((t[:idx_ini_transit], t[idx_end_transit:]))
                    
                    
                    ini_slope = (f[-1] - f[0]) / (t[-1] - t[0])
                    ini_y_intercept = f[0] - ini_slope * t[0]
                    
                    
                    def linear_trend(t,*p):
                        """
                        Linear trend y=a*t+b. p[0] is the y-intercept and p[1] the slope
                        """
                        return p[0]+p[1]*t
                    
                    popt,pcov = curve_fit(linear_trend,
                                            times_out_of_transit,
                                            fluxes_out_of_transit,
                                            p0=[ini_y_intercept,ini_slope],
                                            bounds=((ini_y_intercept-2,ini_slope-0.100),(ini_y_intercept+2,ini_slope+0.100)),#((0.2,-0.100),(1.8,0.100)),#((0.4,-0.020),(1.6,0.020)),#bounds=((0.6,-0.002),(1.4,0.002)),
                                            maxfev=10000)
                    
                    
                    y_intercept,slope = popt
                    
                    # Compute the y values for the straight line using the computed slope and y-intercept
                    ini_straight_line_lc = ini_slope * np.array(t) + ini_y_intercept
                    straight_line_lc = slope * np.array(t) + y_intercept
                    straight_line_lc_out_of_transit = slope * np.array(times_out_of_transit) + y_intercept
                    
                    
                    
                    #Get residuals:
                    residuals_linear_fit = fluxes_out_of_transit - straight_line_lc_out_of_transit
                    
                    #Get fit metrics ([mean(residuals),std(residuals),MSE]):
                    mse_linear_fit = np.sum(residuals_linear_fit**2)/np.shape(residuals_linear_fit)[0]
                    linear_fit_metrics[wv] = [np.mean(residuals_linear_fit),
                                                         np.std(residuals_linear_fit),
                                                         mse_linear_fit]
                    
                    if batman_plot_linear_fits:
                        # Create subplots with specified layout for fit and residuals
                        fig1, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
                        
                        # Access the first axes for the fit plot
                        ax_fit = axs[0]
                        ax_fit.plot(t, ini_straight_line_lc, color='red', alpha=0.5, ls='dashed', 
                                    label="Initial: {}".format([ini_y_intercept, ini_slope]))
                        ax_fit.plot(t, straight_line_lc, color='purple', alpha=0.5, 
                                    label="Fitted: {}".format(popt))
                        ax_fit.scatter(t, f, alpha=0.5)
                        ax_fit.scatter(times_out_of_transit, fluxes_out_of_transit, alpha=0.5)
                        
                        # Plot appearance for the fit plot
                        ax_fit.set_ylabel('Relative flux')
                        ax_fit.tick_params(labelbottom=False)  # Hide x-tick labels for the fit plot
                        ax_fit.legend(fontsize=8)
                        
                        # Access the second axes for residuals plot
                        ax_residuals = axs[1]
                        ax_residuals.scatter(times_out_of_transit, residuals_linear_fit * 1e6, color='pink', s=0.5, 
                                             label=r'MSE: {:.2g}'.format(mse_linear_fit))
                        ax_residuals.axhline(y=0, linestyle='dotted', color='black', linewidth=0.5)
                        ax_residuals.set_xlabel(r'Time (BJD - 2460000)')
                        ax_residuals.set_ylabel(r'Residuals (ppm)')
                        ax_residuals.tick_params(axis='both', which='both', direction='in', right=True, top=True)
                        ax_residuals.legend(loc='best')
                        
                        # Adjust layout and save
                        fig1.tight_layout()
                        fig1.savefig(batman_out_folder + '/{}_{}_linear_fit.png'.format(transit_time, wv))
                        plt.close(fig1)
                        # # Plot linear fit and residuals
                        # fig1 = plt.figure(1)
                        # frame1 = fig1.add_axes((.1,.3,.8,.6))
                        # plt.plot(t,ini_straight_line_lc,c='red',alpha=0.5,ls='dashed',label="Initial: {}".format([ini_y_intercept,ini_slope]))
                        # plt.plot(t,straight_line_lc,c='purple',alpha=0.5,label="Fitted: {}".format(popt))
                        # plt.scatter(t,f,alpha=0.5)
                        # plt.scatter(times_out_of_transit,fluxes_out_of_transit,alpha=0.5)
                        
                        # plt.tight_layout()
        
        
                        # # Plot portion of the lightcurve, axes, etc.:
                        # plt.ylabel('Relative flux')
                        # plt.tick_params(labelbottom=False)  # Hide x-tick labels from the bottom
                        # plt.legend(fontsize=8)
        
                        # #ADDING RESIDUALS:
                        # frame2=fig1.add_axes((.1,.1,.8,.2))
                        # plt.scatter(times_out_of_transit,residuals_linear_fit*1e6,color='pink',s=0.5,label=r'MSE: {:.2g}'.format(mse_linear_fit))
                        # plt.axhline(y = 0, linestyle = 'dotted',color='black',linewidth=0.5)
                        # plt.xlabel(r'Time (BJD - 2460000)')
                        # plt.ylabel(r'residuals (ppm)')
                        # #plt.gca().yaxis.set_label_coords(-0.06,0.5)
                        # plt.tick_params(axis='both',which='both',direction='in',right=True,top=True)
        
                        # plt.legend(loc='best')
                        # plt.savefig(batman_out_folder + '/{}_{}_linear_fit.png'.format(transit_time,wv))
                        # plt.show()
        
                    # Renormalise spectral light curve dividing by linear trend:
                    f = f / straight_line_lc
                    spectral_lcs[j] = spectral_lcs[j] / straight_line_lc
                
                if spectral_lcs_err is None:
                    yerr = np.array([0 for i in range(len(t))])
                else:
                    yerr = spectral_lcs_err[j],
                
                
                

                # # fit model to light curve
                # #print("rp_fitted: {}".format(rp_fitted))
                # #sys.exit("rp_fitted: {}".format(rp_fitted))
                # def speclc_model(exp_times, *fit_vars):
                #     # Update `params` with values to be fitted
                #     param_vals = initial_vals.copy()
                #     u_values = u_initial.copy()
                    
                #     idx = 0
                #     for i, param in enumerate(parameters_to_fit):
                #         if param == 'u':
                #             for k in range(len(u_values)):
                #                 u_values[k] = fit_vars[idx]
                #                 idx += 1
                #         else:
                #             param_vals[param] = fit_vars[idx]
                #             idx += 1
        
                #     # Update `params` with the fitted parameter values
                #     params.rp = param_vals['rp']
                #     params.a = param_vals['a']
                #     params.t0 = param_vals['t0']
                #     params.u = u_values
                    
                #     return model.light_curve(params)
                
                # # Choose a lambda function depending on whether to fix limb darkening
                # if batman_u_fitted_list_to_use is not None:
                #     fixed_u_values = batman_u_fitted_list_to_use[j]
                #     speclc_model = lambda t, *fit_vars: speclc_model(t, *fit_vars, u=fixed_u_values)
                
                # Define the main model function with possible fixed `u` and/or `a` values
                def create_speclc_model(fixed_u=None, fixed_a=None):
                    def speclc_model(exp_times, *fit_vars):
                        # Copy initial values to update with fitted parameters
                        param_vals = initial_vals.copy()
                        u_values = u_initial.copy()
                        
                        idx = 0
                        for param in parameters_to_fit:
                            if param == 'u' and fixed_u is None:
                                # If using three-parameter law, fix u1=0 and fit only u2, u3, u4
                                if batman_ld_law == 'three-parameter':
                                    u_values = [0]+list(u_values)
                                    u_values[0] = 0  # Fix u1=0
                                    for k in range(1, len(u_values)):#+1):
                                        u_values[k] = fit_vars[idx]
                                        idx += 1
                                else:
                                    for k in range(len(u_values)):
                                        u_values[k] = fit_vars[idx]
                                        idx += 1
                            elif param == 'a' and fixed_a is None:
                                param_vals['a'] = fit_vars[idx]
                                idx += 1
                            elif param in param_vals:
                                param_vals[param] = fit_vars[idx]
                                idx += 1
                        
                        
                
                        # Assign values to `params`
                        params.rp = param_vals['rp']
                        params.a = fixed_a if fixed_a is not None else param_vals['a']
                        params.t0 = param_vals['t0']
                        params.inc = param_vals['inc']
                        params.per = param_vals['per']
                        params.u = fixed_u if fixed_u is not None else u_values
                        
                        if batman_use_kipping_ldc_parametrisation:
                            if fixed_u is not None:
                                u_values = fixed_u
                            if batman_ld_law == 'three-parameter':
                                third	= 1./3. #0.333333333333333
                                twopi	= 2*np.pi #6.283185307179586
                                P1	= 4.500841772313891
                                P2	= 17.14213562373095
                                Q1	= 7.996825477806030
                                Q2	= 8.566161603278331
                                
                                alpha_t = u_values[1]
                                alpha_h = u_values[2]
                                alpha_r = u_values[3]
                                c_2 = (alpha_h**third)*( P1 + 0.25*np.sqrt(alpha_r)*( -6.0*np.cos(twopi*alpha_t) + P2*np.sin(twopi*alpha_t) ) )
                                c_3 = (alpha_h**third)*( -Q1 - Q2*np.sqrt(alpha_r)*np.sin(twopi*alpha_t) )
                                c_4 = (alpha_h**third)*( P1 + 0.25*np.sqrt(alpha_r)*( 6.0*np.cos(twopi*alpha_t) + P2*np.sin(twopi*alpha_t) ) )
                                params.u = [0, c_2, c_3, c_4]
                            elif batman_ld_law == 'quadratic':
                                q1 = u_values[0]
                                q2 = u_values[1]
                                params.u = [2*np.sqrt(q1)*q2,np.sqrt(q1)*(1-2*q2)]
                            else:
                                raise ValueError("Kipping LDC parametrisation is not yet implemented for LD laws different than quadratic or three-parameter. batman_use_kipping_ldc_parametrisation=True yet batman_ld_law is neither quadratic nor three-parameter.")
                        
                        if batman_b_to_fix is not None:
                            cosi_to_fix_b = (batman_b_to_fix/params.a)*(1+self.planet_esinw)/(1-params.ecc**2) #cosine of planet inclination
                            params.inc = np.arccos(cosi_to_fix_b)*180./np.pi
                        
                        if batman_model_based_3point:
                            # create batman model
                            model1 = batman.TransitModel(params, exp_times)
                        
                            return model1.light_curve(params)
                        else:
                            if model is not None:
                                return model.light_curve(params)  # Proceed only if model is defined
                            else:
                                # Handle the case where model wasn't initialized
                                raise ValueError("Model was not initialized. Check your batman_model_based_3point condition.")
                    
                    return speclc_model
                
                
                # Conditionally create `speclc_model` based on `batman_u_fitted_list_to_use` and `batman_a_fitted_list_to_use`
                fixed_u_values = batman_u_fitted_list_to_use[j] if batman_u_fitted_list_to_use is not None else None
                fixed_a_value = batman_a_fitted_list_to_use[j] if batman_a_fitted_list_to_use is not None else None
                speclc_model = create_speclc_model(fixed_u=fixed_u_values, fixed_a=fixed_a_value)

                                
                
                
                # Create a dictionary to store parameter indices based on `parameters_to_fit`
                param_index = {}
                idx = 0
                for param in parameters_to_fit:
                    if param == 'u':
                        # For limb-darkening coefficients, store indices for each coefficient
                        param_index['u'] = list(range(idx, idx + len(u_initial)))
                        idx += len(u_initial)
                    else:
                        param_index[param] = idx
                        idx += 1
                
                
                
                
                # Fit the model to the data
                popt, pcov = optimize.curve_fit(speclc_model, 
                                                t, 
                                                spectral_lcs[j],
                                                p0=initial_guess,
                                                bounds=(lower_bounds, upper_bounds),
                                                sigma=yerr if spectral_lcs_err is not None else None,
                                                maxfev=4000,
                                                ftol=1e-10,
                                                xtol=1e-10,
                                                method='trf',
                                                absolute_sigma=True)
                
                
                # Extract fitted parameters based on indices in `param_index`
                fitted_params = {}
                for param, idx in param_index.items():
                    if param == 'u':
                        fitted_params[param] = [popt[i] for i in idx]  # Extract limb-darkening coefficients
                    else:
                        fitted_params[param] = popt[idx]  # Extract single parameter
                
                # Assign fitted parameters to respective variables
                rp_fitted = fitted_params.get('rp', params.rp) #If 'rp' was in parameters to fit, get the fitted value, otherwise default to params.rp
                #a_fitted = fitted_params.get('a', params.a)
                a_fitted = fixed_a_value if fixed_a_value is not None else fitted_params.get('a', params.a)
                t0_fitted = fitted_params.get('t0', params.t0)
                inc_fitted = fitted_params.get('inc', params.inc)
                per_fitted = fitted_params.get('per', params.per)
                u_fitted = fixed_u_values if fixed_u_values is not None else fitted_params.get('u', params.u)
                if batman_ld_law == 'three-parameter' and fixed_u_values is None:
                    u_fitted = [0] + u_fitted
                # if batman_u_fitted_list_to_use is not None:
                #     u_fitted = batman_u_fitted_list_to_use[j]
                # else:
                #     u_fitted = fitted_params.get('u', params.u)  # List of fitted limb-darkening coefficients
                
                # Use `rp_fitted`, `a_fitted`, etc., for any subsequent calculations
                
                
                #log_to_file(batman_out_folder+"/output.log", f"j,wv: {j},{wv}")
                #log_to_file(batman_out_folder+"/output.log", f"popt: {np.array(popt)}")
                #log_to_file(batman_out_folder+"/output.log", f"u_fitted: {np.array(u_fitted)}")
                #log_to_file(batman_out_folder+"/output.log", f"u_initial: {np.array(u_initial)}")
                #log_to_file(batman_out_folder+"/output.log", f"pcov: {np.array(pcov)}")
                
                
                
                
                # estimate error
                perr = np.sqrt(np.diag(pcov))
        
                # save depth and depth error
                if rp_fitted is not None:
                    lc_depths.append(rp_fitted**2)
                    lc_depths_err.append(2 * perr[param_index['rp']] * rp_fitted)
                #lc_depths.append(popt[0]**2)
                #lc_depths_err.append(2*perr[0]*popt[0])
                
                if batman_u_fitted_list_to_use is None:
                    u_fitted_list.append(u_fitted)
                if batman_a_fitted_list_to_use is None:
                    a_fitted_list.append(a_fitted)
                
                #Get residuals:
                if batman_u_fitted_list_to_use is None:
                    residuals = spectral_lcs[j] - speclc_model(t, *popt)
                else:
                    residuals = spectral_lcs[j] - speclc_model(t, *popt)#, fixed_u_values)
                
                #Get fit metrics ([mean(residuals),std(residuals),MSE]):
                mse = np.sum(residuals**2)/np.shape(residuals)[0]
                fit_metrics[wv] = [np.mean(residuals),
                                   np.std(residuals),
                                   mse]
                
                if batman_model_based_3point:
                    model_based_depths.append((np.mean([speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)]) - np.min(speclc_model(np.linspace(np.min(t),np.max(t),10000), *popt))) / np.mean([speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)]))
                    
                    out_of_transit_flux = np.mean([speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)])
                    print("Before transit flux: {}\nAfter transit flux: {}".format(speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)))
                    out_of_transit_flux = np.mean([speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)])
                    print(f"Out-of-transit flux: {out_of_transit_flux}")
                    mid_transit_flux = np.min(speclc_model(np.linspace(np.min(t), np.max(t), 10000), *popt))
                    print(f"Mid-transit flux: {mid_transit_flux}")
                    
                    print("Depth: {}".format(((np.mean([speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)]) - np.min(speclc_model(np.linspace(np.min(t),np.max(t),10000), *popt))) / np.mean([speclc_model(np.array([t[idx_ini_transit-3]]), *popt), speclc_model(np.array([t[idx_end_transit+3]]), *popt)]))))


                    
                
                if batman_plot_lc_fits:
                    
                    legend_text = (
                        f"$r_p$: {rp_fitted:.5f}\n"
                        f"Transit depth: {rp_fitted**2 * 1e6:.1f} ppm\n"
                        f"Mid-transit time: {t0_fitted:.5f} days\n"
                        f"Semi-major axis: {a_fitted:.5f}\n"
                        f"Inclination: {inc_fitted:.5f}°\n"
                        f"Planet period: {per_fitted:.5f} days\n"
                        + '\n'.join([f"LD coeffs: $u_{i+1}$={u_fitted[i]:.5f}" for i in range(len(u_fitted))])
                    )
                    if batman_u_fitted_list_to_use is None:
                        plt.plot(t, speclc_model(t, *popt), color = cmap(norm(wvs[j])), alpha = 0.5, linewidth = 2, label=legend_text)
                    else:
                        plt.plot(t, speclc_model(t, *popt), color = cmap(norm(wvs[j])), alpha = 0.5, linewidth = 2, label=legend_text)# popt[0], batman_u_fitted_list_to_use[j][0], batman_u_fitted_list_to_use[j][1]), color = cmap(norm(wvs[j])), alpha = 0.5, linewidth = 2, label=legend_text)
                    plt.scatter(t, spectral_lcs[j], color = cmap(norm(wvs[j])), alpha=0.1)
                    plt.xlabel('Time (days)')
                    plt.ylabel('Normalized flux')
                    plt.legend()
                    plt.savefig(batman_out_folder + '/batman_fits_wv_{}_transit_time_{}.png'.format(wvs[j],transit_time))
                    plt.show()
                    plt.close()
                
                if j%20==0 and plot_model:
                    print(f"j={j}, wv={wv}")
                    if j==0:
                        fig, ax = plt.subplots()
                    ax.errorbar(t, spectral_lcs[j], yerr=yerr, fmt='o', color=cmap(norm(wvs[j])), alpha=0.01)
                if plot_model:
                    #Plot model:
                    if batman_u_fitted_list_to_use is None:
                        ax.plot(t, speclc_model(t, *popt), color = cmap(norm(wvs[j])), alpha = 0.5, linewidth = 2)
                    else:
                        ax.plot(t, speclc_model(t, *popt), color = cmap(norm(wvs[j])), alpha = 0.5, linewidth = 2)#popt[0], batman_u_fitted_list_to_use[j][0], batman_u_fitted_list_to_use[j][1]), color = cmap(norm(wvs[j])), alpha = 0.5, linewidth = 2)
                    
            
            
            if plot_model:
                # Create a ScalarMappable and associate it with the colormap and normalization
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(wvs)  # Add the data to the ScalarMappable
                
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label('Wavelength (microns)')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Normalized flux')
                fig.savefig(batman_out_folder + '/some_models_batman_fits_transit_time_{}.png'.format(transit_time))
                plt.close(fig)
                
                
            
                #Plot transmission spectrum:
                plt.figure(figsize = (15, 11))
                plt.errorbar(wvs, np.array(lc_depths)*100, yerr = np.array(lc_depths_err)*100, fmt = 'o', capsize = 2, color = 'indianred', markeredgecolor = 'black')
                plt.xlabel('Wavelength (microns)')
                plt.ylabel('Transit depth (%)')
                plt.savefig(batman_out_folder + '/transit_depth_batman_vs_lambda_transit_time_{}_%.png'.format(transit_time))
                plt.show()
                
                
                
            results_depths['batman_curve_fit'] = (np.array(lc_depths))
            if batman_model_based_3point:
                results_depths['batman_model_based_3point'] = (np.array(model_based_depths))
            
            
            batman_fit_metrics_list = [fit_metrics_white_lc,list(linear_fit_metrics.values()),list(fit_metrics.values())]
            
            
            if plot_fit_metrics:
                if batman_fit_white_lc:
                    #Plot all WHITE LC LINEAR fit metrics for this transit_time:
                    #  Mean of residuals
                    plt.plot(range(3),linear_fit_metrics_white_lc)
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("white_lc linear fit metrics")
                    plt.title("mean(res), std(res), MSE")
                    plt.savefig(batman_out_folder + '/white_linear_fit_transit_time_{}.png'.format(transit_time))
                    plt.show()
                
                
                #Plot all LINEAR fit metrics for this transit_time:
                #  Mean of residuals
                plt.plot(wvs,[linear_fit_metrics[wv][0] for wv in wvs])
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("mean(residuals)")
                plt.savefig(batman_out_folder + '/linear_fit_transit_time_{}_mean_residuals.png'.format(transit_time))
                plt.show()
                #  Std of residuals
                plt.plot(wvs,[linear_fit_metrics[wv][1] for wv in wvs])
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("std(residuals)")
                plt.savefig(batman_out_folder + '/linear_fit_transit_time_{}_std_residuals.png'.format(transit_time))
                plt.show()
                #  MSE
                plt.plot(wvs,[linear_fit_metrics[wv][2] for wv in wvs])
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("MSE")
                plt.savefig(batman_out_folder + '/linear_fit_transit_time_{}_MSE.png'.format(transit_time))
                plt.show()
                
                
                
                
                if batman_fit_white_lc:
                    #Plot all WHITE LC fit metrics for this transit_time:
                    #  Mean of residuals
                    plt.plot(range(3),fit_metrics_white_lc)
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("white_lc fit metrics")
                    plt.title("mean(res), std(res), MSE")
                    plt.savefig(batman_out_folder + '/white_fit_transit_time_{}.png'.format(transit_time))
                    plt.show()
                
                
                #Plot all fit metrics for this transit_time:
                #  Mean of residuals
                plt.plot(wvs,[fit_metrics[wv][0] for wv in wvs])
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("mean(residuals)")
                plt.savefig(batman_out_folder + '/fit_transit_time_{}_mean_residuals.png'.format(transit_time))
                plt.show()
                #  Std of residuals
                plt.plot(wvs,[fit_metrics[wv][1] for wv in wvs])
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("std(residuals)")
                plt.savefig(batman_out_folder + '/fit_transit_time_{}_std_residuals.png'.format(transit_time))
                plt.show()
                #  MSE
                plt.plot(wvs,[fit_metrics[wv][2] for wv in wvs])
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("MSE")
                plt.savefig(batman_out_folder + '/fit_transit_time_{}_MSE.png'.format(transit_time))
                plt.show()
                
            if plot_model:
                #Plot transit depth vs wavelength for this transit_time:
                plt.plot(wvs, 1000000*np.array(lc_depths),label="Data")
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("Transit depth (ppm)")
                plt.legend()
                plt.savefig(batman_out_folder + '/transit_depth_vs_lambda_transit_time_{}_ppm.png'.format(transit_time))
                plt.show()
                
                if batman_model_based_3point:
                    # Plot the resulting transit depths across wavelengths
                    plt.plot(wvs, 1e6 * np.array(model_based_depths), label="Model-Based Transit Depth")
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("Transit depth (ppm)")
                    plt.title("Transit Depth Using Model-Based Flux")
                    plt.legend()
                    plt.savefig(batman_out_folder + '/transit_depth_model_based_vs_lambda_transit_time_{}_ppm.png'.format(transit_time))
                    plt.show()
            
            
            if batman_output_data:
                #### SAVING DATA:
                output_base_filename = batman_out_folder + "/transit_depth_vs_wavelength_data_transit_time_{}/".format(transit_time)
                os.makedirs(os.path.dirname(output_base_filename), exist_ok=True) #Check if directory exists, if not then create it
                with open(output_base_filename+"transit_depth_vs_lambda.txt","w") as f:
                    results =  np.array([wvs,np.array(lc_depths)])
                    np.savetxt(f, results, fmt='%.10f', delimiter="\t")
                df = pd.DataFrame(list(fit_metrics.items()), columns=['Key', 'Value'])
                df.to_csv(output_base_filename+"fit_metrics.csv", index=False, sep='\t')
                df = pd.DataFrame(list(linear_fit_metrics.items()), columns=['Key', 'Value'])
                df.to_csv(output_base_filename+"linear_fit_metrics.csv", index=False, sep='\t')
                
                if batman_fit_white_lc:
                    with open(output_base_filename+"white_lc_linear_fit_metrics.txt","w") as f:
                        results =  linear_fit_metrics_white_lc
                        np.savetxt(f, results, fmt='%.10f', delimiter="\t")
                    with open(output_base_filename+"white_lc_fit_metrics.txt","w") as f:
                        results =  fit_metrics_white_lc
                        np.savetxt(f, results, fmt='%.10f', delimiter="\t")
            
            
        
        if use_three_point_approx:
            results_depths['three_point_approx'] = (np.array([(np.mean([spectral_lc[idx_ini_transit],spectral_lc[idx_end_transit]])-np.min(spectral_lc))/np.mean([spectral_lc[idx_ini_transit],spectral_lc[idx_end_transit]]) for spectral_lc in spectral_lcs]))
            if plot_model:
                if three_point_out_folder is None:
                    if batman_out_folder is not None:
                        three_point_out_folder = batman_out_folder
                        print("Warning: three_point_out_folder is None, so 3point results were output to batman_out_folder")
                    elif juliet_out_folder is not None:
                        three_point_out_folder = juliet_out_folder
                        print("Warning: three_point_out_folder and batman_out_folder are None, so 3point results were output to juliet_out_folder")
                    else:
                        sys.exit("ERROR! three_point_out_folder, batman_out_folder and juliet_out_folder are None. Please provide a folder to output plots!")
                        
                plt.plot(wvs,[1000000*results_depth for results_depth in results_depths['three_point_approx']],label="Data")
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("Transit depth (ppm)")
                plt.title("Transit depth using 3-point approx.")
                plt.legend()
                plt.savefig(three_point_out_folder + '/transit_depth_3pointapprox_vs_lambda_transit_time_{}_ppm.png'.format(transit_time))
                plt.show()
        
        
        # Initialize an empty list to hold the outputs
        output = []
        
        # Add juliet_posteriors if requested
        if juliet_return_posteriors:
            output.append(juliet_posteriors)
        
        # Add u_fitted_list if requested
        if batman_return_u_fitted_list:
            output.append(u_fitted_list)
        
        # Add a_fitted_list if requested
        if batman_return_a_fitted_list:
            output.append(a_fitted_list)
        
        #Add batman_fit_params_list if requested:
        if batman_return_batman_fit_metrics_list:
            output.append(batman_fit_metrics_list)
        
        # Add results_depths at the end, since it should always be included
        output.append(results_depths)
        
        # Return the output as a tuple if more than one item is in the output list, otherwise return the single item directly
        return tuple(output) if len(output) > 1 else output[0]

        
        # if juliet_return_posteriors:
        #     return juliet_posteriors,results_depths
        # elif batman_return_u_fitted_list:
        #     return u_fitted_list,results_depths
        # elif batman_return_a_fitted_list:
        #     return a_fitted_list,results_depths
        # return results_depths
    
    
    # def fit_spectral_lcs(self,use_juliet=False):
    #     return 
    
    # def compute_transmission_spectrum(self,use_juliet=False):
    #     self.fit_spectral_lcs(use_juliet)
    #     return
    
    




    #Optimizes ALL the parameters (including spots) using an MCMC. Use only for 1-2 spots.
    def optimize_MCMC(self):
        os.environ["OMP_NUM_THREADS"] = "1"


        print('\nUsing data from the instruments:')
        self.instruments=[]
        self.observables=[]
        typ=[]

        N_obs=0
        for ins in self.data.keys():
            print('-',ins,', with the observables:')
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(0)
                elif obs in ['rv','fwhm','bis','contrast']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(1)
                if obs in ['crx']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(2)
            N_obs+=len(o)
            self.observables.append(o)
            typ.append(ty)



        N_spots = len(self.spot_map) #number of spots in spot_map

        fixed_spot_it=[self.spot_map[i][1] for i in range(N_spots)]
        fixed_spot_lt=[self.spot_map[i][2] for i in range(N_spots)]
        fixed_spot_lat=[self.spot_map[i][3] for i in range(N_spots)]
        fixed_spot_lon=[self.spot_map[i][4] for i in range(N_spots)]
        fixed_spot_c1=[self.spot_map[i][5] for i in range(N_spots)]
        fixed_spot_c2=[self.spot_map[i][6] for i in range(N_spots)]
        fixed_spot_c3=[self.spot_map[i][7] for i in range(N_spots)]
        fixed_T = self.temperature_photosphere
        fixed_sp_T = self.spot_T_contrast
        fixed_fc_T = self.facula_T_contrast
        fixed_Q = self.facular_area_ratio
        fixed_CB = self.convective_shift
        fixed_Prot = self.rotation_period
        fixed_inc = np.rad2deg(np.pi/2-self.inclination) #in deg. 0 is pole-on
        fixed_R = self.radius
        fixed_LD1 = self.limb_darkening_q1
        fixed_LD2 = self.limb_darkening_q2
        fixed_Pp = self.planet_period
        fixed_T0p = self.planet_transit_t0
        fixed_Kp = self.planet_semi_amplitude
        fixed_esinwp = self.planet_esinw
        fixed_ecoswp = self.planet_ecosw
        fixed_Rp =  self.planet_radius
        fixed_bp = self.planet_impact_param
        fixed_alp =  self.planet_spin_orbit_angle    #spin-orbit angle

        self.vparam=np.array([fixed_T,fixed_sp_T,fixed_fc_T,fixed_Q,fixed_CB,fixed_Prot,fixed_inc,fixed_R,fixed_LD1,fixed_LD2,fixed_Pp,fixed_T0p,fixed_Kp,fixed_esinwp,fixed_ecoswp,fixed_Rp,fixed_bp,fixed_alp,*fixed_spot_it,*fixed_spot_lt,*fixed_spot_lat,*fixed_spot_lon,*fixed_spot_c1,*fixed_spot_c2,*fixed_spot_c3])#

        name_spot_it=['spot_{0}_it'.format(i) for i in range(N_spots)]
        name_spot_lt=['spot_{0}_lt'.format(i) for i in range(N_spots)]
        name_spot_lat=['spot_{0}_lat'.format(i) for i in range(N_spots)]
        name_spot_lon=['spot_{0}_lon'.format(i) for i in range(N_spots)]
        name_spot_c1=['spot_{0}_c1'.format(i) for i in range(N_spots)]
        name_spot_c2=['spot_{0}_c2'.format(i) for i in range(N_spots)]
        name_spot_c3=['spot_{0}_c3'.format(i) for i in range(N_spots)]
        name_T ='T$_{{eff}}$'
        name_sp_T ='$\\Delta$ T$_{{sp}}$'
        name_fc_T ='$\\Delta$ T$_{{fc}}$'
        name_Q ='Fac-spot ratio'
        name_CB ='CS'
        name_Prot ='P$_{{rot}}$'
        name_inc ='inc'
        name_R ='R$_*$'
        name_LD1 = 'q$_1$'
        name_LD2 = 'q$_2$'
        name_Pp = 'P$_{{pl}}$'
        name_T0p = 'T$_{{0,pl}}$'
        name_Kp = 'K$_{{pl}}$'
        name_esinwp = 'esinw'
        name_ecoswp = 'ecosw'
        name_Rp =  'R$_{{pl}}$'
        name_bp = 'b'
        name_alp = '$\\lambda$'  

        lparam=np.array([name_T,name_sp_T,name_fc_T,name_Q,name_CB,name_Prot,name_inc,name_R,name_LD1,name_LD2,name_Pp,name_T0p,name_Kp,name_esinwp,name_ecoswp,name_Rp,name_bp,name_alp,*name_spot_it,*name_spot_lt,*name_spot_lat,*name_spot_lon,*name_spot_c1,*name_spot_c2,*name_spot_c3])

        f_spot_it=[self.spot_map[i][8] for i in range(N_spots)]
        f_spot_lt=[self.spot_map[i][9]for i in range(N_spots)]
        f_spot_lat=[self.spot_map[i][10]for i in range(N_spots)]
        f_spot_lon=[self.spot_map[i][11] for i in range(N_spots)]
        f_spot_c1=[self.spot_map[i][12] for i in range(N_spots)]
        f_spot_c2=[self.spot_map[i][13] for i in range(N_spots)]
        f_spot_c3=[self.spot_map[i][14] for i in range(N_spots)]
        f_T = self.prior_t_eff_ph[0]
        f_sp_T = self.prior_spot_T_contrast[0] 
        f_fc_T = self.prior_facula_T_contrast[0] 
        f_Q = self.prior_q_ratio[0]   
        f_CB = self.prior_convective_blueshift[0]   
        f_Prot = self.prior_p_rot[0] 
        f_inc = self.prior_inclination[0]   
        f_R = self.prior_Rstar[0]
        f_LD1 = self.prior_LD1[0]
        f_LD2 = self.prior_LD2[0]
        f_Pp = self.prior_Pp[0]
        f_T0p = self.prior_T0p[0]
        f_Kp = self.prior_Kp[0]
        f_esinwp = self.prior_esinwp[0]
        f_ecoswp = self.prior_ecoswp[0]
        f_Rp = self.prior_Rp[0]
        f_bp = self.prior_bp[0]
        f_alp = self.prior_alp[0]       
        self.fit=np.array([f_T,f_sp_T,f_fc_T,f_Q,f_CB,f_Prot,f_inc,f_R,f_LD1,f_LD2,f_Pp,f_T0p,f_Kp,f_esinwp,f_ecoswp,f_Rp,f_bp,f_alp ,*f_spot_it,*f_spot_lt,*f_spot_lat,*f_spot_lon,*f_spot_c1,*f_spot_c2,*f_spot_c3])       

        bound_spot_it=np.array([[self.prior_spot_initial_time[1],self.prior_spot_initial_time[2]] for i in range(N_spots)])
        bound_spot_lt=np.array([[self.prior_spot_life_time[1],self.prior_spot_life_time[2]] for i in range(N_spots)])
        bound_spot_lat=np.array([[self.prior_spot_latitude[1],self.prior_spot_latitude[2]]for i in range(N_spots)])
        bound_spot_lon=np.array([[self.prior_spot_longitude[1],self.prior_spot_longitude[2]] for i in range(N_spots)])
        bound_spot_c1=np.array([[self.prior_spot_coeff_1[1],self.prior_spot_coeff_1[2]] for i in range(N_spots)])
        bound_spot_c2=np.array([[self.prior_spot_coeff_2[1],self.prior_spot_coeff_2[2]] for i in range(N_spots)])
        bound_spot_c3=np.array([[self.prior_spot_coeff_3[1],self.prior_spot_coeff_3[2]] for i in range(N_spots)])
        bound_T = np.array([self.prior_t_eff_ph[1],self.prior_t_eff_ph[2]])
        bound_sp_T = np.array([self.prior_spot_T_contrast[1],self.prior_spot_T_contrast[2]]) 
        bound_fc_T = np.array([self.prior_facula_T_contrast[1],self.prior_facula_T_contrast[2]]) 
        bound_Q = np.array([self.prior_q_ratio[1],self.prior_q_ratio[2]])   
        bound_CB = np.array([self.prior_convective_blueshift[1],self.prior_convective_blueshift[2]])   
        bound_Prot = np.array([self.prior_p_rot[1],self.prior_p_rot[2]]) 
        bound_inc = np.array([self.prior_inclination[1],self.prior_inclination[2]])   
        bound_R = np.array([self.prior_Rstar[1],self.prior_Rstar[2]])
        bound_LD1 = np.array([self.prior_LD1[1],self.prior_LD1[2]])
        bound_LD2 = np.array([self.prior_LD2[1],self.prior_LD2[2]])
        bound_Pp = np.array([self.prior_Pp[1],self.prior_Pp[2]])
        bound_T0p = np.array([self.prior_T0p[1],self.prior_T0p[2]])
        bound_Kp = np.array([self.prior_Kp[1],self.prior_Kp[2]])
        bound_esinwp = np.array([self.prior_esinwp[1],self.prior_esinwp[2]])
        bound_ecoswp = np.array([self.prior_ecoswp[1],self.prior_ecoswp[2]])
        bound_Rp = np.array([self.prior_Rp[1],self.prior_Rp[2]])
        bound_bp = np.array([self.prior_bp[1],self.prior_bp[2]])
        bound_alp = np.array([self.prior_alp[1],self.prior_alp[2]])

        bounds=np.array([bound_T,bound_sp_T,bound_fc_T,bound_Q,bound_CB,bound_Prot,bound_inc,bound_R,bound_LD1,bound_LD2,bound_Pp,bound_T0p,bound_Kp,bound_esinwp,bound_ecoswp,bound_Rp,bound_bp,bound_alp,*bound_spot_it,*bound_spot_lt,*bound_spot_lat,*bound_spot_lon,*bound_spot_c1,*bound_spot_c2,*bound_spot_c3])#,*np.array(bound_offset),*np.array(bound_jitter)]) 

        prior_spot_it=[spectra.generate_prior(self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_lt=[spectra.generate_prior(self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_lat=[spectra.generate_prior(self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5],self.nwalkers)for i in range(N_spots)]
        prior_spot_lon=[spectra.generate_prior(self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_c1=[spectra.generate_prior(self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_c2=[spectra.generate_prior(self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_c3=[spectra.generate_prior(self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5],self.nwalkers) for i in range(N_spots)]
        prior_T = spectra.generate_prior(self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5],self.nwalkers)
        prior_sp_T = spectra.generate_prior(self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5],self.nwalkers) 
        prior_fc_T = spectra.generate_prior(self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5],self.nwalkers) 
        prior_Q = spectra.generate_prior(self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5],self.nwalkers)   
        prior_CB = spectra.generate_prior(self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5],self.nwalkers)   
        prior_Prot = spectra.generate_prior(self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5],self.nwalkers) 
        prior_inc = spectra.generate_prior(self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5],self.nwalkers)   
        prior_R = spectra.generate_prior(self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5],self.nwalkers)
        prior_LD1 = spectra.generate_prior(self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5],self.nwalkers)
        prior_LD2 = spectra.generate_prior(self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5],self.nwalkers)
        prior_Pp = spectra.generate_prior(self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5],self.nwalkers)
        prior_T0p = spectra.generate_prior(self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5],self.nwalkers)
        prior_Kp = spectra.generate_prior(self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5],self.nwalkers)
        prior_esinwp = spectra.generate_prior(self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5],self.nwalkers)
        prior_ecoswp = spectra.generate_prior(self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5],self.nwalkers)
        prior_Rp = spectra.generate_prior(self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5],self.nwalkers)
        prior_bp = spectra.generate_prior(self.prior_bp[3],self.prior_bp[4],self.prior_bp[5],self.nwalkers)
        prior_alp = spectra.generate_prior(self.prior_alp[3],self.prior_alp[4],self.prior_alp[5],self.nwalkers)

        priors=np.array([prior_T,prior_sp_T,prior_fc_T,prior_Q,prior_CB,prior_Prot,prior_inc,prior_R,prior_LD1,prior_LD2,prior_Pp,prior_T0p,prior_Kp,prior_esinwp,prior_ecoswp,prior_Rp,prior_bp,prior_alp,*prior_spot_it,*prior_spot_lt,*prior_spot_lat,*prior_spot_lon,*prior_spot_c1,*prior_spot_c2,*prior_spot_c3])#,*prior_offset,*prior_jitter]) 


        logprior_spot_it=np.array([[self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5]] for i in range(N_spots)])
        logprior_spot_lt=np.array([[self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5]] for i in range(N_spots)])
        logprior_spot_lat=np.array([[self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5]]for i in range(N_spots)])
        logprior_spot_lon=np.array([[self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5]] for i in range(N_spots)])
        logprior_spot_c1=np.array([[self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5]] for i in range(N_spots)])
        logprior_spot_c2=np.array([[self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5]] for i in range(N_spots)])
        logprior_spot_c3=np.array([[self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5]] for i in range(N_spots)])
        logprior_T=np.array([self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5]])
        logprior_sp_T=np.array([self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5]]) 
        logprior_fc_T=np.array([self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5]]) 
        logprior_Q=np.array([self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5]])   
        logprior_CB=np.array([self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5]])   
        logprior_Prot=np.array([self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5]]) 
        logprior_inc=np.array([self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5]])   
        logprior_R=np.array([self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5]])
        logprior_LD1=np.array([self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5]])
        logprior_LD2=np.array([self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5]])
        logprior_Pp = np.array([self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5]])
        logprior_T0p = np.array([self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5]])
        logprior_Kp = np.array([self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5]])
        logprior_esinwp = np.array([self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5]])
        logprior_ecoswp = np.array([self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5]])
        logprior_Rp = np.array([self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5]])
        logprior_bp = np.array([self.prior_bp[3],self.prior_bp[4],self.prior_bp[5]])
        logprior_alp = np.array([self.prior_alp[3],self.prior_alp[4],self.prior_alp[5]])
        logpriors=np.array([logprior_T,logprior_sp_T,logprior_fc_T,logprior_Q,logprior_CB,logprior_Prot,logprior_inc,logprior_R,logprior_LD1,logprior_LD2,logprior_Pp,logprior_T0p,logprior_Kp,logprior_esinwp,logprior_ecoswp,logprior_Rp,logprior_bp,logprior_alp,*logprior_spot_it,*logprior_spot_lt,*logprior_spot_lat,*logprior_spot_lon,*logprior_spot_c1,*logprior_spot_c2,*logprior_spot_c3])#,*logprior_offset,*logprior_jitter]) 


        vparamfit=np.array([])
        self.lparamfit=np.array([])
        boundfit=[]
        priors_fit=[]
        logpriors_fit=[]

        for i in range(len(self.fit)):
          if self.fit[i]==1:
            vparamfit=np.append(vparamfit,self.vparam[i])
            self.lparamfit=np.append(self.lparamfit,lparam[i])
            priors_fit.append(priors[i])
            logpriors_fit.append(logpriors[i])
            boundfit.append(bounds[i])
        boundfit=np.asarray(boundfit)
        priors_fit=np.asarray(priors_fit)
        logpriors_fit=np.asarray(logpriors_fit)

        
        ndim = len(self.lparamfit)
        p0=priors_fit.T
        
        print('MCMC uncertainties estimation')
        print('Total parameters to optimize:',ndim)

        preburns=self.planet_impact_paramurns
        burns=self.planet_impact_paramurns
        steps=self.steps
        nwalkers=self.nwalkers


        with Pool(self.N_cpus) as pool:
        #EMCEE

            p1=np.zeros([preburns,nwalkers,ndim])
            lp=np.zeros([preburns,nwalkers,1])

            postot=np.zeros([preburns+burns+steps,nwalkers,ndim])
            lptot=np.zeros([preburns+burns+steps,nwalkers])

            sampler = emcee.EnsembleSampler(nwalkers, ndim, spectra.lnposterior,args=(boundfit,logpriors_fit,self.vparam,self.fit,typ,self),pool=pool,moves=[(emcee.moves.DEMove(), 0.2),(emcee.moves.StretchMove(), 0.8)])
            
            print("Running first burn-in...")
            sampler.run_mcmc(p0,preburns,progress=True)
            p1, lp= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)
            postot[0:preburns,:,:], lptot[0:preburns,:]= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)

            print("Running second burn-in...")
            # p2= sampler.get_last_sample()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, spectra.lnposterior,args=(boundfit,logpriors_fit,self.vparam,self.fit,typ,self),pool=pool,moves=[(emcee.moves.DEMove(), 0.6),(emcee.moves.DESnookerMove(), 0.2),(emcee.moves.StretchMove(), 0.2)])
            p2 = p1[np.unravel_index(lp.argmax(), lp.shape)[0:2]] + 1e-1*(np.max(priors_fit,1)-np.min(priors_fit,1)) * np.random.randn(nwalkers, ndim)
            sampler.reset()
            sampler.run_mcmc(p2,burns,progress=True)
            postot[preburns:preburns+burns,:,:], lptot[preburns:preburns+burns,:]= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)              
            p3= sampler.get_last_sample()

            print("Running production...")
            sampler.reset()
            sampler.run_mcmc(p3,steps,progress=True)
            postot[preburns+burns:preburns+burns+steps,:,:], lptot[preburns+burns:preburns+burns+steps:]= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)

        sampler.get_autocorr_time(quiet=True)

        #END OF EMCEE

        self.samples=postot
        self.logs=lptot

        gc.collect()
        sampler.pool.terminate()

        samples_burned=postot[preburns+burns::,:,:].reshape((-1,ndim))
        logs_burned=lptot[preburns+burns::,:].reshape((-2))

        self.planet_impact_paramestparams=samples_burned[np.argmax(logs_burned),:]
        self.planet_impact_paramestmean=[np.median(samples_burned[:,i]) for i in range(len(samples_burned[0]))]
        self.planet_impact_parameststd=[np.std(samples_burned[:,i]) for i in range(len(samples_burned[0]))]
        self.planet_impact_paramestup=[np.quantile(samples_burned[:,i],0.84135)-np.median(samples_burned[:,i]) for i in range(len(samples_burned[0]))]
        self.planet_impact_paramestbot=[np.median(samples_burned[:,i])-np.quantile(samples_burned[:,i],0.15865) for i in range(len(samples_burned[0]))] 

        param_inv=[]
        # print(P)
        ii=0
        for i in range(len(self.fit)):
          if self.fit[i]==0:
            param_inv.append(np.array(self.vparam[i]))
          elif self.fit[i]==1:
            param_inv.append(np.array(samples_burned[:,ii]))
            ii=ii+1
        
        vsini_inv= 2*np.pi*(param_inv[7]*696342)*np.cos(np.deg2rad(90-param_inv[6]))/(param_inv[5]*86400) #in km/s
        if self.limb_darkening_law == 'linear':
            a_LD=param_inv[8]
            b_LD=param_inv[8]*0
        elif self.limb_darkening_law == 'quadratic':
            a_LD=2*np.sqrt(param_inv[8])*param_inv[9]
            b_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9])
        elif self.limb_darkening_law == 'sqrt':
            a_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9]) 
            b_LD=2*np.sqrt(param_inv[8])*param_inv[9]
        elif self.limb_darkening_law == 'log':
            a_LD=param_inv[9]*param_inv[8]**2+1
            b_LD=param_inv[8]**2-1


        s='Results of the inversion process\n'
        print('Results of the inversion process:')
        s+='    -Mean and 1 sigma confidence interval:\n'
        print('\t -Mean and 1 sigma confidence interval:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+{:.5f}-{:.5f}\n'.format(lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865))
            print('\t \t {} = {:.5f}+{:.5f}-{:.5f}'.format(lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865)))
          else:
            s+='        {} = {:.5f} (fixed)\n'.format(lparam[ip],self.vparam[ip])
            print('\t \t',lparam[ip],' = ',self.vparam[ip],'(fixed) ') 
        s+='        $vsini$ = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(vsini_inv),np.quantile(vsini_inv,0.84135)-np.median(vsini_inv),np.median(vsini_inv)-np.quantile(vsini_inv,0.15865))
        s+='        LD_a = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(a_LD),np.quantile(a_LD,0.84135)-np.median(a_LD),np.median(a_LD)-np.quantile(a_LD,0.15865))
        s+='        LD_b = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(b_LD),np.quantile(b_LD,0.84135)-np.median(b_LD),np.median(b_LD)-np.quantile(b_LD,0.15865)) 

        s+='    -Mean and standard deviation:\n'
        print('\t -Mean and standard deviation:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+-{:.5f}\n'.format(lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip]))
            print('\t \t {} = {:.5f}+-{:.5f}'.format(lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip])))
        s+='    -Best solution, with maximum log-likelihood of {:.5f}\n'.format(np.max(self.logs))
        print('\t -Best solution, with maximum log-likelihood of',np.max(self.logs))  
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}\n'.format(lparam[ip],param_inv[ip][np.argmax(logs_burned)])
            print('\t \t {} = {:.5f}'.format(lparam[ip],param_inv[ip][np.argmax(logs_burned)]))


        fig = plt.figure(figsize=(6,10))
        plt.annotate(s, xy=(0.0, 1.0),ha='left',va='top')
        plt.axis('off')
        plt.tight_layout()
        ofilename = self.path / 'plots' / 'results_inversion.png'
        plt.savefig(ofilename,dpi=200)
            



        
    #Optimize the stellar map. The spot map is optimized using Simulated annealing.
    def compute_inverseSA(self,N_inversions):
        N_spots = len(self.spot_map) #number of spots in spot_map
        # self.n_grid_rings = 5 
        self.simulation_mode = 'fast' #must work in fast mode

        print('Computing',N_inversions,'inversions of',N_spots,'spots each.')
        print('\nUsing data from the instruments:')
        self.instruments=[]
        self.observables=[]
        typ=[]

        N_obs=0
        self.N_data=0 #total number of points
        for ins in self.data.keys():

            print('-',ins,', with the observables:')
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(0)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                elif obs in ['rv','fwhm','bis','contrast']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(1)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                if obs in ['crx']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(2)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
            N_obs+=len(o)
            self.observables.append(o)
            typ.append(ty)


        with Pool(processes=self.N_cpus) as pool:
            res=pool.starmap(SA.inversion_parallel,tqdm.tqdm([(self,typ,i) for i in range(N_inversions)],total=N_inversions),chunksize=1)

        best_maps = np.asarray(res,dtype='object')[:,0]
        lnLs = np.asarray(res,dtype='object')[:,1]


        ofilename = self.path / 'results' / 'inversion_stats.npy'
        np.save(ofilename,np.array([lnLs,best_maps],dtype='object'),allow_pickle=True)

        return best_maps, lnLs



    #Optimize the stellar parameters. For each configuration of the MCMC, the spot map is optimized using SA.
    def optimize_inversion_SA(self):
        os.environ["OMP_NUM_THREADS"] = "1"

        N_spots = len(self.spot_map) #number of spots in spot_map
        # self.n_grid_rings = 5 
        self.simulation_mode = 'fast' #must work in fast mode

        print('\nUsing data from the instruments:')
        self.instruments=[]
        self.observables=[]
        typ=[]

        N_obs=0
        self.N_data=0
        for ins in self.data.keys():
            print('-',ins,', with the observables:')
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(0)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                elif obs in ['rv','fwhm','bis','contrast']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(1)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                if obs in ['crx']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(2)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
            N_obs+=len(o)
            self.observables.append(o)
            typ.append(ty)



        fixed_T = self.temperature_photosphere
        fixed_sp_T = self.spot_T_contrast
        fixed_fc_T = self.facula_T_contrast
        fixed_Q = self.facular_area_ratio
        fixed_CB = self.convective_shift
        fixed_Prot = self.rotation_period
        fixed_inc = np.rad2deg(np.pi/2-self.inclination) 
        fixed_R = self.radius
        fixed_LD1 = self.limb_darkening_q1
        fixed_LD2 = self.limb_darkening_q2
        fixed_Pp = self.planet_period
        fixed_T0p = self.planet_transit_t0
        fixed_Kp = self.planet_semi_amplitude
        fixed_esinwp = self.planet_esinw
        fixed_ecoswp = self.planet_ecosw
        fixed_Rp =  self.planet_radius
        fixed_bp = self.planet_impact_param
        fixed_alp =  self.planet_spin_orbit_angle    #spin-orbit angle

        self.vparam=np.array([fixed_T,fixed_sp_T,fixed_fc_T,fixed_Q,fixed_CB,fixed_Prot,fixed_inc,fixed_R,fixed_LD1,fixed_LD2,fixed_Pp,fixed_T0p,fixed_Kp,fixed_esinwp,fixed_ecoswp,fixed_Rp,fixed_bp,fixed_alp])

        name_T ='T$_{{eff}}$'
        name_sp_T ='$\\Delta$ T$_{{sp}}$'
        name_fc_T ='$\\Delta$ T$_{{fc}}$'
        name_Q ='Fac-spot ratio'
        name_CB ='CS'
        name_Prot ='P$_{{rot}}$'
        name_inc ='inc'
        name_R ='R$_*$'
        name_LD1 = 'q$_1$'
        name_LD2 = 'q$_2$'
        name_Pp = 'P$_{{pl}}$'
        name_T0p = 'T$_{{0,pl}}$'
        name_Kp = 'K$_{{pl}}$'
        name_esinwp = 'esinw'
        name_ecoswp = 'ecosw'
        name_Rp =  'R$_{{pl}}$'
        name_bp = 'b'
        name_alp = '$\\lambda$'  

        self.lparam=np.array([name_T,name_sp_T,name_fc_T,name_Q,name_CB,name_Prot,name_inc,name_R,name_LD1,name_LD2,name_Pp,name_T0p,name_Kp,name_esinwp,name_ecoswp,name_Rp,name_bp,name_alp])

        f_T = self.prior_t_eff_ph[0]
        f_sp_T = self.prior_spot_T_contrast[0] 
        f_fc_T = self.prior_facula_T_contrast[0] 
        f_Q = self.prior_q_ratio[0]   
        f_CB = self.prior_convective_blueshift[0]   
        f_Prot = self.prior_p_rot[0] 
        f_inc = self.prior_inclination[0]   
        f_R = self.prior_Rstar[0]
        f_LD1 = self.prior_LD1[0]
        f_LD2 = self.prior_LD2[0]
        f_Pp = self.prior_Pp[0]
        f_T0p = self.prior_T0p[0]
        f_Kp = self.prior_Kp[0]
        f_esinwp = self.prior_esinwp[0]
        f_ecoswp = self.prior_ecoswp[0]
        f_Rp = self.prior_Rp[0]
        f_bp = self.prior_bp[0]
        f_alp = self.prior_alp[0]     

        self.fit=np.array([f_T,f_sp_T,f_fc_T,f_Q,f_CB,f_Prot,f_inc,f_R,f_LD1,f_LD2,f_Pp,f_T0p,f_Kp,f_esinwp,f_ecoswp,f_Rp,f_bp,f_alp])       

        bound_T = np.array([self.prior_t_eff_ph[1],self.prior_t_eff_ph[2]])
        bound_sp_T = np.array([self.prior_spot_T_contrast[1],self.prior_spot_T_contrast[2]]) 
        bound_fc_T = np.array([self.prior_facula_T_contrast[1],self.prior_facula_T_contrast[2]]) 
        bound_Q = np.array([self.prior_q_ratio[1],self.prior_q_ratio[2]])   
        bound_CB = np.array([self.prior_convective_blueshift[1],self.prior_convective_blueshift[2]])   
        bound_Prot = np.array([self.prior_p_rot[1],self.prior_p_rot[2]]) 
        bound_inc = np.array([self.prior_inclination[1],self.prior_inclination[2]])   
        bound_R = np.array([self.prior_Rstar[1],self.prior_Rstar[2]])
        bound_LD1 = np.array([self.prior_LD1[1],self.prior_LD1[2]])
        bound_LD2 = np.array([self.prior_LD2[1],self.prior_LD2[2]])
        bound_Pp = np.array([self.prior_Pp[1],self.prior_Pp[2]])
        bound_T0p = np.array([self.prior_T0p[1],self.prior_T0p[2]])
        bound_Kp = np.array([self.prior_Kp[1],self.prior_Kp[2]])
        bound_esinwp = np.array([self.prior_esinwp[1],self.prior_esinwp[2]])
        bound_ecoswp = np.array([self.prior_ecoswp[1],self.prior_ecoswp[2]])
        bound_Rp = np.array([self.prior_Rp[1],self.prior_Rp[2]])
        bound_bp = np.array([self.prior_bp[1],self.prior_bp[2]])
        bound_alp = np.array([self.prior_alp[1],self.prior_alp[2]])

        bounds=np.array([bound_T,bound_sp_T,bound_fc_T,bound_Q,bound_CB,bound_Prot,bound_inc,bound_R,bound_LD1,bound_LD2,bound_Pp,bound_T0p,bound_Kp,bound_esinwp,bound_ecoswp,bound_Rp,bound_bp,bound_alp]) 

        prior_T = spectra.generate_prior(self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5],self.steps)
        prior_sp_T = spectra.generate_prior(self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5],self.steps) 
        prior_fc_T = spectra.generate_prior(self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5],self.steps) 
        prior_Q = spectra.generate_prior(self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5],self.steps)   
        prior_CB = spectra.generate_prior(self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5],self.steps)   
        prior_Prot = spectra.generate_prior(self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5],self.steps) 
        prior_inc = spectra.generate_prior(self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5],self.steps)   
        prior_R = spectra.generate_prior(self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5],self.steps)
        prior_LD1 = spectra.generate_prior(self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5],self.steps)
        prior_LD2 = spectra.generate_prior(self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5],self.steps)
        prior_Pp = spectra.generate_prior(self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5],self.steps)
        prior_T0p = spectra.generate_prior(self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5],self.steps)
        prior_Kp = spectra.generate_prior(self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5],self.steps)
        prior_esinwp = spectra.generate_prior(self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5],self.steps)
        prior_ecoswp = spectra.generate_prior(self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5],self.steps)
        prior_Rp = spectra.generate_prior(self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5],self.steps)
        prior_bp = spectra.generate_prior(self.prior_bp[3],self.prior_bp[4],self.prior_bp[5],self.steps)
        prior_alp = spectra.generate_prior(self.prior_alp[3],self.prior_alp[4],self.prior_alp[5],self.steps)

        priors=np.array([prior_T,prior_sp_T,prior_fc_T,prior_Q,prior_CB,prior_Prot,prior_inc,prior_R,prior_LD1,prior_LD2,prior_Pp,prior_T0p,prior_Kp,prior_esinwp,prior_ecoswp,prior_Rp,prior_bp,prior_alp]) 


        logprior_T=np.array([self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5]])
        logprior_sp_T=np.array([self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5]]) 
        logprior_fc_T=np.array([self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5]]) 
        logprior_Q=np.array([self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5]])   
        logprior_CB=np.array([self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5]])   
        logprior_Prot=np.array([self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5]]) 
        logprior_inc=np.array([self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5]])   
        logprior_R=np.array([self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5]])
        logprior_LD1=np.array([self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5]])
        logprior_LD2=np.array([self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5]])
        logprior_Pp = np.array([self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5]])
        logprior_T0p = np.array([self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5]])
        logprior_Kp = np.array([self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5]])
        logprior_esinwp = np.array([self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5]])
        logprior_ecoswp = np.array([self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5]])
        logprior_Rp = np.array([self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5]])
        logprior_bp = np.array([self.prior_bp[3],self.prior_bp[4],self.prior_bp[5]])
        logprior_alp = np.array([self.prior_alp[3],self.prior_alp[4],self.prior_alp[5]])

        logpriors=np.array([logprior_T,logprior_sp_T,logprior_fc_T,logprior_Q,logprior_CB,logprior_Prot,logprior_inc,logprior_R,logprior_LD1,logprior_LD2,logprior_Pp,logprior_T0p,logprior_Kp,logprior_esinwp,logprior_ecoswp,logprior_Rp,logprior_bp,logprior_alp]) 


        vparamfit=np.array([])
        self.lparamfit=np.array([])
        boundfit=[]
        priors_fit=[]
        logpriors_fit=[]

        for i in range(len(self.fit)):
          if self.fit[i]==1:
            vparamfit=np.append(vparamfit,self.vparam[i])
            self.lparamfit=np.append(self.lparamfit,self.lparam[i])
            priors_fit.append(priors[i])
            logpriors_fit.append(logpriors[i])
            boundfit.append(bounds[i])
        boundfit=np.asarray(boundfit)
        priors_fit=np.asarray(priors_fit)
        logpriors_fit=np.asarray(logpriors_fit)

        
        ndim = len(self.lparamfit)
        p0=priors_fit.T
        
        print('Searching random grid for best stellar parameters. Optimizing spotmap at each step.')
        print('Total parameters to optimize:',ndim)

        steps=self.steps

        with Pool(processes=self.N_cpus) as pool:
            res=pool.starmap(SA.inversion_parallel_MCMC,tqdm.tqdm([(self,p0,boundfit,logpriors_fit,typ,i) for i in range(steps)], total=steps),chunksize=1)


        p_used = np.asarray(res,dtype='object')[:,0]
        best_maps = np.asarray(res,dtype='object')[:,1]
        lnLs = np.asarray(res,dtype='object')[:,2]
        
        #print("p_used: {}".format(p_used)) 
        #print("best_maps: {}".format(best_maps)) 
        #print("lnLs: {}".format(lnLs)) 
        
        p_used_noNones = [p for p in p_used if p is not None]
        best_maps_noNones = [p for p in best_maps if p is not None]
        lnLs_noNones = [p for p in lnLs if p is not None]
        
        #print("p_used_noNones: {}".format(p_used_noNones)) 
        #print("best_maps_noNones: {}".format(best_maps_noNones)) 
        #print("lnLs_noNones: {}".format(lnLs_noNones)) 

        ofilename = self.path / 'results' / 'optimize_inversion_SA_stats.npy'
        #np.save(ofilename,np.array([lnLs,p_used,best_maps],dtype='object'),allow_pickle=True)
        np.save(ofilename,np.array([lnLs_noNones,p_used_noNones,best_maps_noNones],dtype='object'),allow_pickle=True)














    def load_data(self,filename=None,t=None,y=None,yerr=None,instrument=None,observable=None,wvmin=None,wvmax=None,filter_name=None,offset=None,fix_offset=False,jitter=0.0,fix_jitter=False):
    
        
        if observable not in ['lc','rv','bis','fwhm','contrast','crx']:
            sys.exit('Observable not valid. Use one of the following: lc, rv, bis, fwhm, contrast or crx')

        if wvmin==None and wvmax==None:
            print('Wavelength range of the instrument not specified. Using the values in the file starsim.conf, ',self.wavelength_lower_limit,'and ',self.wavelength_upper_limit)

        if observable=='lc' and filter_name== None:
            print('Filter file name not specified. Using the values in ',self.filter_name,'. Filters can be retrieved from http://svo2.cab.inta-csic.es/svo/theory/fps3/')
            filter_name = self.filter_name

        self.data[instrument][observable]={}
        self.data[instrument]['wvmin']=wvmin
        self.data[instrument]['wvmax']=wvmax
        self.data[instrument]['filter']=filter_name
        self.data[instrument][observable]['offset']=offset
        self.data[instrument][observable]['jitter']=jitter
        self.data[instrument][observable]['fix_offset']=fix_offset
        self.data[instrument][observable]['fix_jitter']=fix_jitter

        if filename != None:
            filename = self.path / filename
            self.data[instrument][observable]['t'], self.data[instrument][observable]['y'], self.data[instrument][observable]['yerr'] = np.loadtxt(filename,unpack=True)              
        else:
            self.data[instrument][observable]['t'], self.data[instrument][observable]['y'], self.data[instrument][observable]['yerr'] = t, y, yerr
            if t is None:
                sys.exit('Please provide a valid filename with the input data')


        if observable in ['lc','fwhm','bis']:
            if offset == 0.0:
                sys.exit("Error in the input offset of the observable:",observable,". It is a multiplicative offset, can't be 0")
            if offset is None:
                offset=1.0
            self.data[instrument][observable]['yerr']=np.sqrt(self.data[instrument][observable]['yerr']**2+jitter**2)/offset
            self.data[instrument][observable]['y']=self.data[instrument][observable]['y']/offset
            self.data[instrument][observable]['offset_type']='multiplicative'
        else:
            if offset is None:
                offset=0.0
            self.data[instrument][observable]['y']=self.data[instrument][observable]['y'] - offset
            self.data[instrument][observable]['yerr']=np.sqrt(self.data[instrument][observable]['yerr']**2+jitter**2)
            self.data[instrument][observable]['offset_type']='linear'



    ###########################################################
    ################ PLOTS ####################################
    ###########################################################
    def plot_forward_results(self):
        '''method for plotting the results from forward method
        '''
        fig, ax = plt.subplots(len(self.results.keys())-2,figsize=(6,8),sharex=True)
        k=0
        for i, name in enumerate(self.results.keys()):
            if name=='time':
                k-=1
            elif name=='CCF':
                k-=1
            else:
                ax[k].plot(self.results['time'],self.results[name],'.')
                ax[k].set_ylabel(self.name_params[name])
                ax[k].minorticks_on()
                ax[k].tick_params(axis='both',which='both',direction='inout')
            k+=1


        ax[-1].set_xlabel('Obs. time [days]')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0)

        ofilename = self.path  / 'plots' / 'forward_results.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

    def plot_MCMCoptimization_chain(self):
        ndim=len(self.lparamfit)
        fig1 = plt.figure(figsize=(15,12))
        xstep=np.arange(len(self.samples[:,0,0]))
        for ip in range(ndim):
          plt.subplot(m.ceil(ndim/4),4,ip+1)
          for iw in range(0,self.nwalkers):
            ystep=self.samples[:,iw,ip]
            plt.plot(xstep,ystep,'-k',alpha=0.07)
          plt.xlabel('MCMC step')
          plt.ylabel(self.lparamfit[ip])

        ofilename = self.path  / 'plots' / 'MCMCoptimization_chains.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

    def plot_MCMCoptimization_likelihoods(self):
        xtot=self.samples.reshape((-1,len(self.lparamfit)))
        ytot=self.logs.reshape((-2))
        ndim=len(self.lparamfit)
        fig1 = plt.figure(figsize=(15,12))
        
        for ip in range(ndim):
          plt.subplot(m.ceil(ndim/4),4,ip+1)
          plt.plot(xtot[:,ip],ytot,'k,')
          plt.axhline(np.max(ytot)-15,color='r',ls=':')
          plt.ylabel('lnL')
          left=np.min(xtot[ytot>(np.max(ytot)-15),ip])
          right=np.max(xtot[ytot>(np.max(ytot)-15),ip])
          plt.ylim([np.max(ytot)-30,np.max(ytot)+2])
          plt.xlim([left-(right-left)*0.2,right+(right-left)*0.2])
          plt.xlabel(self.lparamfit[ip])
        
        ofilename = self.path  / 'plots' / 'MCMCoptimization_likelihoods.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

    def plot_MCMCoptimization_corner(self):
        fig2, axes = plt.subplots(len(self.lparamfit),len(self.lparamfit), figsize=(2.3*len(self.lparamfit),2.3*len(self.lparamfit)))
        corner.corner(self.samples[-self.steps::,:,:].reshape((-1,len(self.lparamfit))),bins=20,plot_contours=False,fig=fig2,max_n_ticks=2,labels=self.lparamfit,label_kwargs={'fontsize':13},quantiles=(0.16,0.5,0.84),show_titles=True)
        
        ofilename = self.path / 'plots' / 'MCMCoptimization_cornerplot.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()



    def plot_MCMCoptimization_results(self,Nsamples=100,t=None,fold=True):

        sample=self.samples[-self.steps::,:,:].reshape((-1,len(self.lparamfit)))
        num_obs=len(sum(self.observables,[]))

        fig2, ax = plt.subplots(num_obs,1,figsize=(8,12))
        if num_obs == 1:
            ax= [ax]

        stack_dic={}
        for k in range(Nsamples):
            sys.stdout.write("\r [{}/{}]".format(k,Nsamples))
            P=sample[np.random.randint(len(sample))]
            #Variable p contains all the parameters available, fixed and optimized. P are the optimized parameters,vparam are the fixed params.
            p=np.zeros(len(self.vparam))
            # print(P)
            ii=0
            for i in range(len(self.fit)):
              if self.fit[i]==0:
                p[i]=self.vparam[i]
              elif self.fit[i]==1:
                p[i]=P[ii]
                ii=ii+1

            self.temperature_photosphere = p[0]
            self.spot_T_contrast = p[1]
            self.facula_T_contrast = p[2]
            self.facular_area_ratio = p[3]
            self.convective_shift = p[4]
            self.rotation_period = p[5]
            self.inclination = np.deg2rad(90-p[6]) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
            self.radius = p[7] #in Rsun
            self.limb_darkening_q1 = p[8]
            self.limb_darkening_q2 = p[9]
            self.planet_period = p[10]
            self.planet_transit_t0 = p[11]
            self.planet_semi_amplitude = p[12]
            self.planet_esinw = p[13]
            self.planet_ecosw = p[14]
            self.planet_radius = p[15]
            self.planet_impact_param = p[16]
            self.planet_spin_orbit_angle = p[17]*np.pi/180 #deg2rad           

            N_spots=len(self.spot_map)
            for i in range(N_spots):
                self.spot_map[i][1]=p[18+i]
                self.spot_map[i][2]=p[18+N_spots+i]
                self.spot_map[i][3]=p[18+2*N_spots+i]
                self.spot_map[i][4]=p[18+3*N_spots+i]
                self.spot_map[i][5]=p[18+4*N_spots+i]
                self.spot_map[i][6]=p[18+5*N_spots+i]
                self.spot_map[i][7]=p[18+6*N_spots+i]


            #np.round(t/step)*step
            #Compute the model for each instrument and observable, and the corresponding lnL
            l=0
            for i in range(len(self.instruments)):
                for j in self.observables[i]:
                    if k==0:
                        stack_dic['{}_{}'.format(i,j)]=[]
                    self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                    self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                    self.filter_name=self.data[self.instruments[i]]['filter']
                    self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)
                    
                    if self.data[self.instruments[i]][j]['offset_type']=='multiplicative': #j=='lc':
                        
                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=1.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=1.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                            
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        stack_dic['{}_{}'.format(i,j)].append(self.results[j]*offset)
                        l+=1
                    
                    
                    else: #linear offset

                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=0.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=0.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        stack_dic['{}_{}'.format(i,j)].append(self.results[j]+offset)
                        l+=1
        
        if fold is True:
            t=t/self.rotation_period%1*self.rotation_period
        idxsrt=np.argsort(t)

        #Plot the data
        l=0
        for i in range(len(self.instruments)):
            for j in self.observables[i]:
                ax[l].fill_between(t[idxsrt],np.mean(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt]-np.std(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt],np.mean(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt]+np.std(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt],color='k',alpha=0.3)
                ax[l].plot(t[idxsrt],np.mean(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt],'k')
                if fold is True:
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t']/self.rotation_period%1*self.rotation_period,self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr'],fmt='bo',ecolor='lightblue')
                else:
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr'],fmt='bo',ecolor='lightblue')
                ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                l+=1

        ofilename = self.path  / 'plots' / 'MCMCoptimization_timeseries_result.png'
        plt.savefig(ofilename,dpi=200)
        plt.close()
        # plt.show(block=True)



    def plot_inversion_results(self,best_maps,lnLs,Npoints=200,plot_bestlnL=True,plot_separately=True,time_units="",custom_labels=None,return_store_results=False):
        # Ensure `custom_labels` is a dictionary, even if not provided
        if custom_labels is None:
            custom_labels = {}

        self.instruments=[]
        self.observables=[]
        typ=[]
        tmax=-3000000
        tmin=3000000

        for ins in self.data.keys():
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    o.append(obs)
                    ty.append(0)
                    if self.data[ins]['lc']['t'].min()<tmin: tmin=self.data[ins]['lc']['t'].min()
                    if self.data[ins]['lc']['t'].max()>tmax: tmax=self.data[ins]['lc']['t'].max()
                elif obs in ['rv','fwhm','bis','contrast']:
                    o.append(obs)
                    ty.append(1)
                    if self.data[ins][obs]['t'].min()<tmin: tmin=self.data[ins][obs]['t'].min()
                    if self.data[ins][obs]['t'].max()>tmax: tmax=self.data[ins][obs]['t'].max()
                if obs in ['crx']:
                    o.append(obs)
                    ty.append(2)
                    if self.data[ins]['crx']['t'].min()<tmin: tmin=self.data[ins]['crx']['t'].min()
                    if self.data[ins]['crx']['t'].max()>tmax: tmax=self.data[ins]['crx']['t'].max()
            self.observables.append(o)

        num_obs=len(sum(self.observables,[]))

        t=np.linspace(tmin,tmax,Npoints)

        fig, ax = plt.subplots(num_obs,1,figsize=(12,12))
        if num_obs == 1:
            ax= [ax]



        store_results = np.zeros([len(best_maps),num_obs,Npoints])


        bestlnL=np.argmax(lnLs)
        for k in range(len(best_maps)):
            self.spot_map[:,1:8] = best_maps[k]
                        
            #Plot the data
            l=0
            for i in range(len(self.instruments)):
                for j in self.observables[i]:
                    self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                    self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                    self.filter_name=self.data[self.instruments[i]]['filter']
                    self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)

                    if self.data[self.instruments[i]][j]['offset_type']=='multiplicative':
                        
                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=1.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=1.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]                           
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        if k==bestlnL:
                            if plot_bestlnL:
                                ax[l].plot(t,self.results[j]*offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)                        
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                            
                            

                        
                        store_results[k,l,:]=self.results[j]*offset
                        # ax[l].plot(t,self.results[j]*offset,c=cmap.to_rgba(lnLs[k]),alpha=0.5)
                        l+=1
                        
                        if k==bestlnL and plot_separately:
                            # Create a new figure for each individual plot
                            fig_sep = plt.figure(figsize=(8, 6))
                            ax_sep = fig_sep.add_subplot(1, 1, 1)  # Create a single subplot in the new figure
                            
                            # Apply manual options for ticks and labels
                            ax_sep.tick_params(axis='both', direction='in', top=True, right=True, labelsize=14)
    
                            if plot_bestlnL:
                                ax_sep.plot(t, self.results[j] * offset, 'r--', zorder=11, label=f'Offset={offset:.5f}, Jitter={jitter:.5f}')
                            
                            ax_sep.errorbar(
                                self.data[self.instruments[i]][j]['t'],
                                self.data[self.instruments[i]][j]['y'],
                                np.sqrt(self.data[self.instruments[i]][j]['yerr']**2 + jitter**2),
                                fmt='bo', ecolor='lightblue', zorder=10,label='Data'
                            )
                            
                            # Add the mean and standard deviation plots
                            mean_values = np.mean(store_results[:, l - 1, :], axis=0)
                            std_values = np.std(store_results[:, l - 1, :], axis=0)
                            ax_sep.plot(t, mean_values, 'k', label='Mean fit')
                            ax_sep.fill_between(t, mean_values - std_values, mean_values + std_values, color='k', alpha=0.2, label=r'$1\sigma$ range')
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.set_xlabel('Time', fontsize=15)
                            # #ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.legend(fontsize=14)
                            
                            # Generate labels
                            default_label = f'{self.instruments[i]}_{j}'
                            ylabel = custom_labels.get(default_label, default_label)  # Use custom label if available
                            xlabel = f'Time ({time_units})' if time_units else 'Time'
                            
                            # Set labels, title, and font sizes manually
                            ax_sep.set_ylabel(ylabel, fontsize=15)
                            ax_sep.set_xlabel(xlabel, fontsize=15)
                            #ax_sep.set_title(f'Inversion Timeseries - {ylabel}', fontsize=15)
                            ax_sep.legend(fontsize=14)
    
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}')
                            # ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}')
                            # ax_sep.legend()
                            
                            # Save and close the individual plot to isolate it
                            if plot_bestlnL:
                                ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png'
                            else:
                                ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}.png'
                            # ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{ylabel}.png' # BE CAREFUL, this is problematic if you have repeated labels!!! e.g. if you have more than one "Norm. flux (I-band)".
                            fig_sep.savefig(ofilename_sep, dpi=200)
                            plt.close(fig_sep)  # Close the separate figure
                    

                    else: #linear offset

                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=0.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=0.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        store_results[k,l,:]=self.results[j]+offset
                        if k==bestlnL:
                            if plot_bestlnL:
                                ax[l].plot(t,self.results[j]+offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        l+=1
                        
                        if k==bestlnL and plot_separately:
                            # Create a new figure for each individual plot
                            fig_sep = plt.figure(figsize=(8, 6))
                            ax_sep = fig_sep.add_subplot(1, 1, 1)  # Create a single subplot in the new figure
                            
                            # Apply manual options for ticks and labels
                            ax_sep.tick_params(axis='both', direction='in', top=True, right=True, labelsize=14)
    
                            if plot_bestlnL:
                                ax_sep.plot(t, self.results[j] + offset, 'r--', zorder=11, label=f'Offset={offset:.5f}, Jitter={jitter:.5f}')
                            
                            ax_sep.errorbar(
                                self.data[self.instruments[i]][j]['t'],
                                self.data[self.instruments[i]][j]['y'],
                                np.sqrt(self.data[self.instruments[i]][j]['yerr']**2 + jitter**2),
                                fmt='bo', ecolor='lightblue', zorder=10,label='Data'
                            )
                            
                            # Add the mean and standard deviation plots
                            mean_values = np.mean(store_results[:, l - 1, :], axis=0)
                            std_values = np.std(store_results[:, l - 1, :], axis=0)
                            ax_sep.plot(t, mean_values, 'k', label='Mean fit')
                            ax_sep.fill_between(t, mean_values - std_values, mean_values + std_values, color='k', alpha=0.2, label=r'$1\sigma$ range')
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.set_xlabel('Time', fontsize=15)
                            # #ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.legend(fontsize=14)
                            
                            # Generate labels
                            default_label = f'{self.instruments[i]}_{j}'
                            ylabel = custom_labels.get(default_label, default_label)  # Use custom label if available
                            xlabel = f'Time ({time_units})' if time_units else 'Time'
                            
                            # Set labels, title, and font sizes manually
                            ax_sep.set_ylabel(ylabel, fontsize=15)
                            ax_sep.set_xlabel(xlabel, fontsize=15)
                            #ax_sep.set_title(f'Inversion Timeseries - {ylabel}', fontsize=15)
                            ax_sep.legend(fontsize=14)
    
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}')
                            # ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}')
                            # ax_sep.legend()
                            
                            # Save and close the individual plot to isolate it
                            if plot_bestlnL:
                                ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png'
                            else:
                                ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}.png'
                            # ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{ylabel}.png' # BE CAREFUL, this is problematic if you have repeated labels!!! e.g. if you have more than one "Norm. flux (I-band)".
                            fig_sep.savefig(ofilename_sep, dpi=200)
                            plt.close(fig_sep)  # Close the separate figure

        for i in range(num_obs):
            ax[i].plot(t,np.mean(store_results[:,i],axis=0),'k')
            ax[i].fill_between(t,np.mean(store_results[:,i],axis=0)-np.std(store_results[:,i],axis=0),np.mean(store_results[:,i],axis=0)+np.std(store_results[:,i],axis=0),color='k',alpha=0.2)
            


        if plot_bestlnL:
            ofilename = self.path  / 'plots' / 'inversion_timeseries_result_plotbestlnL.png'
        else:
            ofilename = self.path  / 'plots' / 'inversion_timeseries_result.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show(block=True)
        plt.close()
        
        if return_store_results:
            return store_results

    
    def plot_spot_map(self,best_maps,tref=None):

        N_div = 100
        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(N_div)
        vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian
        theta, phi = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination)), np.arctan2(ys,xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))#coordinates in the star reference 

        

        if tref is None:
            tref=best_maps[0][0][0]
        elif len(tref)==1:
            tref=[tref]

        for t in tref:
            Surface=np.zeros(len(vec_grid[:,0])) #len Ngrids
            for k in range(len(best_maps)):

                self.spot_map[:,1:8]=best_maps[k]
                spot_pos=spectra.compute_spot_position(self,t) #return colat, longitude and raddii in radians
             
                vec_spot=np.zeros([len(self.spot_map),3])
                xspot = np.cos(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])+np.sin(self.inclination)*np.cos(spot_pos[:,0])
                yspot = np.sin(spot_pos[:,0])*np.sin(spot_pos[:,1])
                zspot = np.cos(spot_pos[:,0])*np.cos(self.inclination)-np.sin(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])
                vec_spot[:,:]=np.array([xspot,yspot,zspot]).T #spot center in cartesian
                
                for s in range(len(best_maps[k])):
                    if spot_pos[s,2]==0:
                        continue

                    for i in range(len(vec_grid[:,0])):
                        dist=m.acos(np.dot(vec_spot[s],vec_grid[i]))
                        if dist < spot_pos[s,2]:
                            Surface[i]+=1

                    

            cm = plt.cm.get_cmap('afmhot_r')
            #make figure
            fig = plt.figure(1,figsize=(6,6))
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.plot(np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),'k') #circumference
            x=np.linspace(-0.999,0.999,1000)
            h=np.sqrt((1-x**2)/(np.tan(self.inclination)**2+1))
            plt.plot(x,h,'k--')
            spotmap = ax.scatter(vec_grid[:,1],vec_grid[:,2], marker='o', c=Surface/len(best_maps), s=5.0, edgecolors='none', cmap=cm,vmax=(Surface.max()+0.1)/len(best_maps),vmin=-0.2*(Surface.max()+0.1)/len(best_maps))
            # cb = plt.colorbar(spotmap,ax=ax, fraction=0.035, pad=0.05, aspect=20)
            ofilename = self.path  / 'plots' / 'inversion_spotmap_t_{:.4f}.png'.format(t)
            plt.savefig(ofilename,dpi=200)
            # plt.show()
            plt.close()

    def plot_active_longitudes(self,best_maps,tini=None,tfin=None,N_obs=100):

        N_div = 500
        

        if tini is None:
            tini=best_maps[0][0][0]
        if tfin is None:
            tfin=best_maps[0][0][0]+1.0

        tref=np.linspace(tini,tfin,N_obs)
        longs=np.linspace(0,2*np.pi,N_div)
        Surface=np.zeros([N_obs,N_div])
        for j in range(N_obs):
            for k in range(len(best_maps)):
                self.spot_map[:,1:8]=best_maps[k]
                spot_pos=spectra.compute_spot_position(self,tref[j]) #return colat, longitude and raddii in radians

        #update longitude adding diff rotation
                for s in range(len(best_maps[k])):
                    ph_s=(spot_pos[s,1]-((tref[j]-self.reference_time)/self.rotation_period%1*360)*np.pi/180)%(2*np.pi) #longitude
                    r_s=spot_pos[s,2] #radius
                    if r_s==0.0:
                        continue

                    for i in range(N_div):
                        dist=np.abs(longs[i]-ph_s) #distance to spot centre
                        
                        if dist < r_s:
                            Surface[j,i]+=1

        X,Y = np.meshgrid(longs*180/np.pi,tref)
        fig = plt.figure(1,figsize=(6,6))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.1)
        ax = fig.add_subplot(111)
        cm = plt.cm.get_cmap('afmhot_r')
        spotmap=ax.contourf(X,Y,Surface/len(best_maps), 25, cmap=cm,vmax=(Surface.max()+0.1)/len(best_maps),vmin=-0.2*(Surface.max()+0.1)/len(best_maps))
        cb = plt.colorbar(spotmap,ax=ax)   
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Time [d]")

        ofilename = self.path / 'plots' / 'active_map.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show()
        plt.close()

    def plot_optimize_inversion_SA_results(self,DeltalnL,plot_bestlnL=True,equal_aspect_ratio_inv_timeseries=True,results_relative_path=None,plot_separately=True,time_units="",custom_labels=None,plot_relative_path=None):
        # Ensure `custom_labels` is a dictionary, even if not provided
        if custom_labels is None:
            custom_labels = {}

        fixed_T = self.temperature_photosphere
        fixed_sp_T = self.spot_T_contrast
        fixed_fc_T = self.facula_T_contrast
        fixed_Q = self.facular_area_ratio
        fixed_CB = self.convective_shift
        fixed_Prot = self.rotation_period
        fixed_inc = np.rad2deg(np.pi/2-self.inclination) 
        fixed_R = self.radius
        fixed_LD1 = self.limb_darkening_q1
        fixed_LD2 = self.limb_darkening_q2
        fixed_Pp = self.planet_period
        fixed_T0p = self.planet_transit_t0
        fixed_Kp = self.planet_semi_amplitude
        fixed_esinwp = self.planet_esinw
        fixed_ecoswp = self.planet_ecosw
        fixed_Rp =  self.planet_radius
        fixed_bp = self.planet_impact_param
        fixed_alp =  self.planet_spin_orbit_angle    #spin-orbit angle

        self.vparam=np.array([fixed_T,fixed_sp_T,fixed_fc_T,fixed_Q,fixed_CB,fixed_Prot,fixed_inc,fixed_R,fixed_LD1,fixed_LD2,fixed_Pp,fixed_T0p,fixed_Kp,fixed_esinwp,fixed_ecoswp,fixed_Rp,fixed_bp,fixed_alp])

        name_T ='T$_{{eff}}$'
        name_sp_T ='$\\Delta$ T$_{{sp}}$'
        name_fc_T ='$\\Delta$ T$_{{fc}}$'
        name_Q ='Fac-spot ratio'
        name_CB ='CS'
        name_Prot ='P$_{{rot}}$'
        name_inc ='inc'
        name_R ='R$_*$'
        name_LD1 = 'q$_1$'
        name_LD2 = 'q$_2$'
        name_Pp = 'P$_{{pl}}$'
        name_T0p = 'T$_{{0,pl}}$'
        name_Kp = 'K$_{{pl}}$'
        name_esinwp = 'esinw'
        name_ecoswp = 'ecosw'
        name_Rp =  'R$_{{pl}}$'
        name_bp = 'b'
        name_alp = '$\\lambda$'  

        self.lparam=np.array([name_T,name_sp_T,name_fc_T,name_Q,name_CB,name_Prot,name_inc,name_R,name_LD1,name_LD2,name_Pp,name_T0p,name_Kp,name_esinwp,name_ecoswp,name_Rp,name_bp,name_alp])

        f_T = self.prior_t_eff_ph[0]
        f_sp_T = self.prior_spot_T_contrast[0] 
        f_fc_T = self.prior_facula_T_contrast[0] 
        f_Q = self.prior_q_ratio[0]   
        f_CB = self.prior_convective_blueshift[0]   
        f_Prot = self.prior_p_rot[0] 
        f_inc = self.prior_inclination[0]   
        f_R = self.prior_Rstar[0]
        f_LD1 = self.prior_LD1[0]
        f_LD2 = self.prior_LD2[0]
        f_Pp = self.prior_Pp[0]
        f_T0p = self.prior_T0p[0]
        f_Kp = self.prior_Kp[0]
        f_esinwp = self.prior_esinwp[0]
        f_ecoswp = self.prior_ecoswp[0]
        f_Rp = self.prior_Rp[0]
        f_bp = self.prior_bp[0]
        f_alp = self.prior_alp[0]       
        self.fit=np.array([f_T,f_sp_T,f_fc_T,f_Q,f_CB,f_Prot,f_inc,f_R,f_LD1,f_LD2,f_Pp,f_T0p,f_Kp,f_esinwp,f_ecoswp,f_Rp,f_bp,f_alp])       

        self.lparamfit=np.array([])
        for i in range(len(self.fit)):
          if self.fit[i]==1:
            self.lparamfit=np.append(self.lparamfit,self.lparam[i])


        #read the results
        if results_relative_path is None:
            filename = self.path / 'results' / 'optimize_inversion_SA_stats.npy'
        else:
            filename = str(self.path) + results_relative_path
        res = np.load(filename,allow_pickle=True)

        lnLs=res[0]
        params=np.vstack(res[1]).T
        best_maps=res[2]
        ndim=np.sum(self.fit)

        p=np.zeros([ndim,len(lnLs)])
        # print(P)
        ii=0
        for i in range(len(self.fit)):
          if self.fit[i]==1:
            p[ii,:]=params[i,:]
            ii+=1


        
        fig1 = plt.figure(figsize=(15,12))
        
        for ip in range(ndim):
          plt.subplot(m.ceil(ndim/4),4,ip+1)
          plt.plot(p[ip],lnLs,'k.')
          plt.axhline(np.max(lnLs)-DeltalnL,color='r',ls=':')
          plt.ylabel('lnL')
          # left=np.min(xtot[ytot>(np.max(ytot)-15),ip])
          # right=np.max(xtot[ytot>(np.max(ytot)-15),ip])
          plt.ylim([np.max(lnLs)-DeltalnL*3,np.max(lnLs)+DeltalnL/10])
          # plt.xlim([left-(right-left)*0.2,right+(right-left)*0.2])
          plt.xlabel(self.lparamfit[ip])
        
        if plot_relative_path is None:
            ofilename = self.path / 'plots' / 'inversion_MCMCSA_likelihoods.png'
        else:
            ofilename = str(self.path) + plot_relative_path + '/inversion_MCMCSA_likelihoods.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

        paramsnew=res[1][lnLs>(np.nanmax(lnLs)-DeltalnL)]
        pcorner=p[:,lnLs>(np.nanmax(lnLs)-DeltalnL)]
        lnLsnew=lnLs[lnLs>(np.nanmax(lnLs)-DeltalnL)]
        best_maps_new=best_maps[lnLs>(np.nanmax(lnLs)-DeltalnL)]


        fig2, axes = plt.subplots(int(ndim),int(ndim), figsize=(2.3*int(ndim),2.3*int(ndim)))
        corner.corner(pcorner.T,bins=10,plot_contours=False,fig=fig2,max_n_ticks=2,labels=self.lparamfit,label_kwargs={'fontsize':13},quantiles=(0.16,0.5,0.84),show_titles=True)
        
        if plot_relative_path is None:
            ofilename = self.path / 'plots' / 'inversion_MCMCSA_cornerplot.png'
        else:
            ofilename = str(self.path) + plot_relative_path + '/inversion_MCMCSA_cornerplot.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()


        param_inv=[]
        # print(P)
        ii=0
        for i in range(len(self.fit)):
          if self.fit[i]==0:
            param_inv.append(np.array(self.vparam[i]))
          elif self.fit[i]==1:
            param_inv.append(np.array(pcorner[ii]))
            ii=ii+1
        
        vsini_inv= 2*np.pi*(param_inv[7]*696342)*np.cos(np.deg2rad(90-param_inv[6]))/(param_inv[5]*86400) #in km/s
        if self.limb_darkening_law == 'linear':
            a_LD=param_inv[8]
            b_LD=param_inv[8]*0
        elif self.limb_darkening_law == 'quadratic':
            a_LD=2*np.sqrt(param_inv[8])*param_inv[9]
            b_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9])
        elif self.limb_darkening_law == 'sqrt':
            a_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9]) 
            b_LD=2*np.sqrt(param_inv[8])*param_inv[9]
        elif self.limb_darkening_law == 'log':
            a_LD=param_inv[9]*param_inv[8]**2+1
            b_LD=param_inv[8]**2-1


        s='Results of the inversion process with DeltalnL<{:.1f} \n'.format(DeltalnL)
        print('Results of the inversion process:')
        s+='    -Mean and 1 sigma confidence interval:\n'
        print('\t -Mean and 1 sigma confidence interval:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+{:.5f}-{:.5f}\n'.format(self.lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865))
            print('\t \t {} = {:.5f}+{:.5f}-{:.5f}'.format(self.lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865)))
          else:
            s+='        {} = {:.5f} (fixed)\n'.format(self.lparam[ip],self.vparam[ip])
            print('\t \t',self.lparam[ip],' = ',self.vparam[ip],'(fixed) ') 
        s+='        $vsini$ = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(vsini_inv),np.quantile(vsini_inv,0.84135)-np.median(vsini_inv),np.median(vsini_inv)-np.quantile(vsini_inv,0.15865))
        s+='        LD_a = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(a_LD),np.quantile(a_LD,0.84135)-np.median(a_LD),np.median(a_LD)-np.quantile(a_LD,0.15865))
        s+='        LD_b = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(b_LD),np.quantile(b_LD,0.84135)-np.median(b_LD),np.median(b_LD)-np.quantile(b_LD,0.15865)) 

        s+='    -Mean and standard deviation:\n'
        print('\t -Mean and standard deviation:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+-{:.5f}\n'.format(self.lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip]))
            print('\t \t {} = {:.5f}+-{:.5f}'.format(self.lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip])))
        s+='    -Best solution, with maximum log-likelihood of {:.5f}\n'.format(np.max(lnLsnew))
        print('\t -Best solution, with maximum log-likelihood of',np.max(lnLsnew))  
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}\n'.format(self.lparam[ip],param_inv[ip][np.argmax(lnLsnew)])
            print('\t \t {} = {:.5f}'.format(self.lparam[ip],param_inv[ip][np.argmax(lnLsnew)]))

        fig = plt.figure(figsize=(6,10))
        plt.annotate(s, xy=(0.0, 1.0),ha='left',va='top')
        plt.axis('off')
        plt.tight_layout()
        if plot_relative_path is None:
            ofilename = self.path / 'plots' / 'inversion_MCMCSA_results.png'
        else:
            ofilename = str(self.path) + plot_relative_path + '/inversion_MCMCSA_results.png'
        plt.savefig(ofilename,dpi=200)
        plt.close()



        self.instruments=[]
        self.observables=[]
        tmax=-3000000
        tmin=3000000
        Npoints=1000

        for ins in self.data.keys():
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    o.append(obs)
                    ty.append(0)
                    if self.data[ins]['lc']['t'].min()<tmin: tmin=self.data[ins]['lc']['t'].min()
                    if self.data[ins]['lc']['t'].max()>tmax: tmax=self.data[ins]['lc']['t'].max()
                elif obs in ['rv','fwhm','bis','contrast']:
                    o.append(obs)
                    ty.append(1)
                    if self.data[ins][obs]['t'].min()<tmin: tmin=self.data[ins][obs]['t'].min()
                    if self.data[ins][obs]['t'].max()>tmax: tmax=self.data[ins][obs]['t'].max()
                if obs in ['crx']:
                    o.append(obs)
                    ty.append(2)
                    if self.data[ins]['crx']['t'].min()<tmin: tmin=self.data[ins]['crx']['t'].min()
                    if self.data[ins]['crx']['t'].max()>tmax: tmax=self.data[ins]['crx']['t'].max()
            self.observables.append(o)

        num_obs=len(sum(self.observables,[]))

        t=np.linspace(tmin,tmax,Npoints)

        
        if equal_aspect_ratio_inv_timeseries:
            fig, ax = plt.subplots(num_obs, 1, figsize=(12, 12 * num_obs))  # 12 units width and height for each subplot
        else:
            fig, ax = plt.subplots(num_obs,1,figsize=(12,12))
        if num_obs == 1:
            ax= [ax]

        store_results = np.zeros([len(best_maps_new),num_obs,Npoints])

        all_figs_sep = []
        all_axs_sep = []
        all_filenames_sep = []
        bestlnL=np.argmax(lnLsnew)
        for k in range(len(best_maps_new)):
            self.spot_map = best_maps_new[k]
            self.temperature_photosphere = paramsnew[k][0]
            self.spot_T_contrast = paramsnew[k][1]
            self.facula_T_contrast = paramsnew[k][2]
            self.facular_area_ratio = paramsnew[k][3]
            self.convective_shift = paramsnew[k][4]
            self.rotation_period = paramsnew[k][5]
            self.inclination = np.deg2rad(90-paramsnew[k][6]) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
            self.radius = paramsnew[k][7] #in Rsun
            self.limb_darkening_q1 = paramsnew[k][8]
            self.limb_darkening_q2 = paramsnew[k][9]
            self.planet_period = paramsnew[k][10]
            self.planet_transit_t0 = paramsnew[k][11]
            self.planet_semi_amplitude = paramsnew[k][12]
            self.planet_esinw = paramsnew[k][13]
            self.planet_ecosw = paramsnew[k][14]
            self.planet_radius = paramsnew[k][15]
            self.planet_impact_param = paramsnew[k][16]
            self.planet_spin_orbit_angle = paramsnew[k][17]*np.pi/180 #deg2rad   

            #Plot the data
            l=0
            for i in range(len(self.instruments)):
                for j in self.observables[i]:
                    self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                    self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                    self.filter_name=self.data[self.instruments[i]]['filter']
                    self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)

                    if self.data[self.instruments[i]][j]['offset_type']=='multiplicative':
                        
                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=1.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=1.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]                           
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        if k==bestlnL:
                            if plot_bestlnL:
                                ax[l].plot(t,self.results[j]*offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)                        
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        store_results[k,l,:]=self.results[j]*offset
                        l+=1
                        
                        if k==bestlnL and plot_separately:
                            # Create a new figure for each individual plot
                            fig_sep = plt.figure(figsize=(8, 6))
                            ax_sep = fig_sep.add_subplot(1, 1, 1)  # Create a single subplot in the new figure
                            
                            # Apply manual options for ticks and labels
                            ax_sep.tick_params(axis='both', direction='in', top=True, right=True, labelsize=14)
    
                            if plot_bestlnL:
                                ax_sep.plot(t, self.results[j] * offset, 'r--', zorder=11, label=f'Offset={offset:.5f}, Jitter={jitter:.5f}')
                            
                            ax_sep.errorbar(
                                self.data[self.instruments[i]][j]['t'],
                                self.data[self.instruments[i]][j]['y'],
                                np.sqrt(self.data[self.instruments[i]][j]['yerr']**2 + jitter**2),
                                fmt='bo', ecolor='lightblue', zorder=10,label='Data'
                            )
                            
                            # # Add the mean and standard deviation plots
                            # mean_values = np.mean(store_results[:, l - 1, :], axis=0)
                            # std_values = np.std(store_results[:, l - 1, :], axis=0)
                            # ax_sep.plot(t, mean_values, 'k', label='Mean fit')
                            # ax_sep.fill_between(t, mean_values - std_values, mean_values + std_values, color='k', alpha=0.2, label=r'$1\sigma$ range')
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.set_xlabel('Time', fontsize=15)
                            # #ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.legend(fontsize=14)
                            
                            # Generate labels
                            default_label = f'{self.instruments[i]}_{j}'
                            ylabel = custom_labels.get(default_label, default_label)  # Use custom label if available
                            xlabel = f'Time ({time_units})' if time_units else 'Time'
                            
                            # Set labels, title, and font sizes manually
                            ax_sep.set_ylabel(ylabel, fontsize=15)
                            ax_sep.set_xlabel(xlabel, fontsize=15)
                            #ax_sep.set_title(f'Inversion Timeseries - {ylabel}', fontsize=15)
                            #ax_sep.legend(fontsize=14)
    
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}')
                            # ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}')
                            # ax_sep.legend()
                            
                            all_figs_sep.append(fig_sep)
                            all_axs_sep.append(ax_sep)
                            if plot_bestlnL:
                                if plot_relative_path is None:
                                    all_filenames_sep.append(self.path / 'plots' / f'/inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png')
                                else:
                                    all_filenames_sep.append(str(self.path) + plot_relative_path + f'/inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png')
                            else:
                                if plot_relative_path is None:
                                    all_filenames_sep.append(self.path / 'plots' / f'/inversion_timeseries_result_{self.instruments[i]}_{j}.png')
                                else:
                                    all_filenames_sep.append(str(self.path) + plot_relative_path + f'/inversion_timeseries_result_{self.instruments[i]}_{j}.png')
                            
                            # # Save and close the individual plot to isolate it
                            # if plot_bestlnL:
                            #     ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png'
                            # else:
                            #     ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}.png'
                            # # ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{ylabel}.png' # BE CAREFUL, this is problematic if you have repeated labels!!! e.g. if you have more than one "Norm. flux (I-band)".
                            # fig_sep.savefig(ofilename_sep, dpi=200)
                            # plt.close(fig_sep)  # Close the separate figure
                    

                    else: #linear offset

                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=0.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=0.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        store_results[k,l,:]=self.results[j]+offset
                        if k==bestlnL:
                            if plot_bestlnL:
                                ax[l].plot(t,self.results[j]+offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        l+=1
                        
                        if k==bestlnL and plot_separately:
                            # Create a new figure for each individual plot
                            fig_sep = plt.figure(figsize=(8, 6))
                            ax_sep = fig_sep.add_subplot(1, 1, 1)  # Create a single subplot in the new figure
                            
                            # Apply manual options for ticks and labels
                            ax_sep.tick_params(axis='both', direction='in', top=True, right=True, labelsize=14)
    
                            if plot_bestlnL:
                                ax_sep.plot(t, self.results[j] + offset, 'r--', zorder=11, label=f'Offset={offset:.5f}, Jitter={jitter:.5f}')
                            
                            ax_sep.errorbar(
                                self.data[self.instruments[i]][j]['t'],
                                self.data[self.instruments[i]][j]['y'],
                                np.sqrt(self.data[self.instruments[i]][j]['yerr']**2 + jitter**2),
                                fmt='bo', ecolor='lightblue', zorder=10,label='Data'
                            )
                            
                            # # Add the mean and standard deviation plots
                            # mean_values = np.mean(store_results[:, l - 1, :], axis=0)
                            # std_values = np.std(store_results[:, l - 1, :], axis=0)
                            # ax_sep.plot(t, mean_values, 'k', label='Mean fit')
                            # ax_sep.fill_between(t, mean_values - std_values, mean_values + std_values, color='k', alpha=0.2, label=r'$1\sigma$ range')
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.set_xlabel('Time', fontsize=15)
                            # #ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}', fontsize=15)
                            # ax_sep.legend(fontsize=14)
                            
                            # Generate labels
                            default_label = f'{self.instruments[i]}_{j}'
                            ylabel = custom_labels.get(default_label, default_label)  # Use custom label if available
                            xlabel = f'Time ({time_units})' if time_units else 'Time'
                            
                            # Set labels, title, and font sizes manually
                            ax_sep.set_ylabel(ylabel, fontsize=15)
                            ax_sep.set_xlabel(xlabel, fontsize=15)
                            #ax_sep.set_title(f'Inversion Timeseries - {ylabel}', fontsize=15)
                            #ax_sep.legend(fontsize=14)
    
    
                            # ax_sep.set_ylabel(f'{self.instruments[i]}_{j}')
                            # ax_sep.set_title(f'Inversion Timeseries - {self.instruments[i]}_{j}')
                            # ax_sep.legend()
                            
                            all_figs_sep.append(fig_sep)
                            all_axs_sep.append(ax_sep)
                            
                            if plot_bestlnL:
                                if plot_relative_path is None:
                                    all_filenames_sep.append(self.path / 'plots' / f'/inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png')
                                else:
                                    all_filenames_sep.append(str(self.path) + plot_relative_path + f'/inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png')
                            else:
                                if plot_relative_path is None:
                                    all_filenames_sep.append(self.path / 'plots' / f'/inversion_timeseries_result_{self.instruments[i]}_{j}.png')
                                else:
                                    all_filenames_sep.append(str(self.path) + plot_relative_path + f'/inversion_timeseries_result_{self.instruments[i]}_{j}.png')
                            # if plot_bestlnL:
                            #     all_filenames_sep.append(self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png'
                            # else:
                            #     all_filenames_sep.append(self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}.png'
                            
                            
                            # # Save and close the individual plot to isolate it
                            # if plot_bestlnL:
                            #     ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}_plotbestlnL.png'
                            # else:
                            #     ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{self.instruments[i]}_{j}.png'
                            # # ofilename_sep = self.path / 'plots' / f'inversion_timeseries_result_{ylabel}.png' # BE CAREFUL, this is problematic if you have repeated labels!!! e.g. if you have more than one "Norm. flux (I-band)".
                            # fig_sep.savefig(ofilename_sep, dpi=200)
                            # plt.close(fig_sep)  # Close the separate figure

        for i in range(num_obs):
            ax[i].plot(t,np.mean(store_results[:,i],axis=0),'k')
            ax[i].fill_between(t,np.mean(store_results[:,i],axis=0)-np.std(store_results[:,i],axis=0),np.mean(store_results[:,i],axis=0)+np.std(store_results[:,i],axis=0),color='k',alpha=0.2)
            
            if plot_separately:
                all_axs_sep[i].plot(t,np.mean(store_results[:,i],axis=0),'k',label='Mean fit')
                all_axs_sep[i].fill_between(t,np.mean(store_results[:,i],axis=0)-np.std(store_results[:,i],axis=0),np.mean(store_results[:,i],axis=0)+np.std(store_results[:,i],axis=0),color='k',alpha=0.2,label=r'$1\sigma$ range')
                all_axs_sep[i].legend(fontsize=14)
                
                
                
                all_figs_sep[i].savefig(all_filenames_sep[i], dpi=200)
                plt.close(all_figs_sep[i])  # Close the separate figure
            
            
        
        
        
            
        if equal_aspect_ratio_inv_timeseries:
            plt.tight_layout()  # Add this line to automatically adjust the spacing between subplots

        #ofilename = self.path  / 'plots' / 'inversion_timeseries_result.png'
        if plot_bestlnL:
            if plot_relative_path is None:
                ofilename = self.path  / 'plots' / 'inversion_timeseries_result_plotbestlnL.png'
            else:
                ofilename = str(self.path) + plot_relative_path + '/inversion_timeseries_result_plotbestlnL.png'
        else:
            if plot_relative_path is None:
                ofilename = self.path  / 'plots' / 'inversion_timeseries_result.png'
            else:
                ofilename = str(self.path) + plot_relative_path + '/inversion_timeseries_result.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show(block=True)
        plt.close()


    def plot_data_and_model(self,spot_map,stellar_params,Npoints=200):


        #set spot map and stellar params
        self.spot_map = spot_map
        self.set_stellar_parameters(stellar_params)


        self.instruments=[]
        self.observables=[]
        typ=[]
        tmax=-3000000
        tmin=3000000

        for ins in self.data.keys():
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    o.append(obs)
                    ty.append(0)
                    if self.data[ins]['lc']['t'].min()<tmin: tmin=self.data[ins]['lc']['t'].min()
                    if self.data[ins]['lc']['t'].max()>tmax: tmax=self.data[ins]['lc']['t'].max()
                elif obs in ['rv','fwhm','bis','contrast']:
                    o.append(obs)
                    ty.append(1)
                    if self.data[ins][obs]['t'].min()<tmin: tmin=self.data[ins][obs]['t'].min()
                    if self.data[ins][obs]['t'].max()>tmax: tmax=self.data[ins][obs]['t'].max()
                if obs in ['crx']:
                    o.append(obs)
                    ty.append(2)
                    if self.data[ins]['crx']['t'].min()<tmin: tmin=self.data[ins]['crx']['t'].min()
                    if self.data[ins]['crx']['t'].max()>tmax: tmax=self.data[ins]['crx']['t'].max()
            self.observables.append(o)

        num_obs=len(sum(self.observables,[]))

        t=np.linspace(tmin,tmax,Npoints)

        fig, ax = plt.subplots(num_obs,1,figsize=(12,12))
        if num_obs == 1:
            ax= [ax]





                    
        #Plot the data
        l=0
        for i in range(len(self.instruments)):
            for j in self.observables[i]:
                self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                self.filter_name=self.data[self.instruments[i]]['filter']
                self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)

                if self.data[self.instruments[i]][j]['offset_type']=='multiplicative':
                    
                    if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                        offset=1.0
                        jitter=0.0

                    elif self.data[self.instruments[i]][j]['fix_offset']:
                        offset=1.0
                        res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        jitter=res.x[0]                           
                    
                    elif self.data[self.instruments[i]][j]['fix_jitter']:
                        jitter=0.0
                        res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                        offset=res.x[0]

                    else:
                        res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        offset=res.x[0]
                        jitter=res.x[1]

                    self.compute_forward(observables=j,t=t,inversion=True)
                    
                    ax[l].plot(t,self.results[j]*offset,'k',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)                        
                    ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                    ax[l].legend()
                    

                else: #linear offset

                    if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                        offset=0.0
                        jitter=0.0

                    elif self.data[self.instruments[i]][j]['fix_offset']:
                        offset=0.0
                        res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        jitter=res.x[0]
                    
                    elif self.data[self.instruments[i]][j]['fix_jitter']:
                        jitter=0.0
                        res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                        offset=res.x[0]

                    else:
                        res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        offset=res.x[0]
                        jitter=res.x[1]

                    self.compute_forward(observables=j,t=t,inversion=True)

                    ax[l].plot(t,self.results[j]+offset,'k',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)
                    ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                    ax[l].legend()

                l+=1




        ofilename = self.path  / 'plots' / 'data_and_model.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show(block=True)
        plt.close()