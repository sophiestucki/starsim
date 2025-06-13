#NUMBA ############################################
import numba as nb
import numpy as np
import math as m

@nb.njit
def dummy():
    return None

@nb.njit(cache=True,error_model='numpy')
def fit_multiplicative_offset_jitter(x0,f,y,dy):
    off=x0[0]
    jit=x0[1]
    newerr=np.sqrt((dy)**2+jit**2)/off
    lnL=-0.5*np.sum(((y/off-f)/(newerr))**2.0+np.log(2.0*np.pi)+np.log(newerr**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_only_multiplicative_offset(x0,f,y,dy):
    off=x0
    lnL=-0.5*np.sum(((y/off-f)/(dy/off))**2.0+np.log(2.0*np.pi)+np.log((dy/off)**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_linear_offset_jitter(x0,f,y,dy):
    off=x0[0]
    jit=x0[1]
    lnL=-0.5*np.sum(((y-off-f)/(np.sqrt(dy**2+jit**2)))**2.0+np.log(2.0*np.pi)+np.log(dy**2+jit**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_only_linear_offset(x0,f,y,dy):
    off=x0
    lnL=-0.5*np.sum(((y-off-f)/(dy))**2.0+np.log(2.0*np.pi)+np.log(dy**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_only_jitter(x0,f,y,dy):
    jit=x0
    lnL=-0.5*np.sum(((y-f)/(np.sqrt(dy**2+jit**2)))**2.0+np.log(2.0*np.pi)+np.log(dy**2+jit**2))
    return -lnL


@nb.njit(cache=True,error_model='numpy')
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
    
@nb.njit(cache=True,error_model='numpy')
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_
 
@nb.njit(cache=True,error_model='numpy')
def fit_poly(x, y, deg,w):
    a = _coeff_mat(x, deg)*w.reshape(-1,1)
    p = _fit_x(a, y*w)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]
#####################################################
############# UTILITIES #############################
#####################################################
@nb.njit(cache=True,error_model='numpy')
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x-mean)**2/(2*stddev**2))

@nb.njit(cache=True,error_model='numpy')
def gaussian2(x, amplitude, mean, stddev,C):
    return C + amplitude * np.exp(-(x-mean)**2/(2*stddev**2))


@nb.njit(cache=True,error_model='numpy')
def normalize_spectra_nb(bins,wavelength,flux):

    x_bin=np.zeros(len(bins)-1)
    y_bin=np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        idxup = wavelength>bins[i]
        idxdown= wavelength<bins[i+1]
        idx=idxup & idxdown
        y_bin[i]=flux[idx].max()
        x_bin[i]=wavelength[idx][np.argmax(flux[idx])]
    #divide by 6th deg polynomial

    return x_bin, y_bin


@nb.njit(cache=True,error_model='numpy')
def interpolation_nb(xp,x,y,left=0,right=0):

    # Create result array
    yp=np.zeros(len(xp))
    minx=x[0]
    maxx=x[-1]
    lastidx=1 

    for i,xi in enumerate(xp):
        if xi<minx: #extrapolate left
            yp[i]=left
        elif xi>maxx: #extrapolate right
            yp[i]=right
        else:
            for j in range(lastidx,len(x)): #per no fer el loop sobre tota la x, ja que esta sorted sempre comenso amb lanterior.
                if x[j]>xi:
                    #Trobo el primer x mes gran que xj. llavors utilitzo x[j] i x[j-1] per interpolar
                    yp[i]=y[j-1]+(xi-x[j-1])*(y[j]-y[j-1])/(x[j]-x[j-1])
                    lastidx=j
                    break

    return yp



@nb.njit(cache=True,error_model='numpy')
def cross_correlation_nb(rv,wv,flx,wv_ref,flx_ref):
    #Compute the CCF against the reference spectrum. Can be optimized.
    ccf=np.zeros(len(rv)) #initialize ccf
    lenf=len(flx_ref)
    for i in range(len(rv)):
        wvshift=wv_ref*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
        # fshift=np.interp(wvshift,wv,flx)
        fshift = interpolation_nb(wvshift,wv,flx,left=0,right=0)
        ccf[i]=np.sum(flx_ref*fshift)/lenf #compute ccf

    return (ccf-np.min(ccf))/np.max((ccf-np.min(ccf)))


@nb.njit(cache=True,error_model='numpy')
def cross_correlation_mask(rv,wv,f,wvm,fm, phoenix_spectra):
    """
    Function to compute CCF against Phoenix-spectra. The steps used
    are specific to Phoenix spectra, do not use other spectra.
    
    """
    ccf = np.zeros(len(rv))
    lenm = len(wvm)
    wvmin=wv[0]

    if phoenix_spectra:

        for i in range(len(rv)):
            wvshift=wvm*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
            #for each mask line
            for j in range(lenm):
                #find wavelengths right and left of the line.
                wvline=wvshift[j]

                if wvline<3000.0:
                    idxlf = int((wvline-wvmin)/0.1)

                elif wvline<4999.986:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((wvline-3000.0)/0.006) 
                    else:
                        idxlf = int((wvline-wvmin)/0.006)

                elif wvline<5000.0:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1
                    else:
                        idxlf = int((4999.986-wvmin)/0.006) + 1

                elif wvline<10000.0:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((wvline-5000.0)/0.01)
                    elif wvmin<4999.986:
                        idxlf = int((4999.986-wvmin)/0.006) + 1 + int((wvline-5000.0)/0.01)
                    else:
                        idxlf = int((wvline-wvmin)/0.01) 

                elif wvline<15000.0:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((10000.0-5000.0)/0.01) + int((wvline-10000.0)/0.02)
                    elif wvmin<4999.986:
                        idxlf = int((4999.986-wvmin)/0.006) + 1 + int((10000-5000.0)/0.01) + int((wvline-10000.0)/0.02)
                    elif wvmin<10000.0:
                        idxlf = int((10000.0-wvmin)/0.01) + int((wvline-10000.0)/0.02)
                    else:
                        idxlf = int((wvline-wvmin)/0.02)

                else:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((10000.0-5000.0)/0.01) + int((15000.0-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                    elif wvmin<4999.986:
                        idxlf = int((4999.986-wvmin)/0.006) + 1 + int((10000-5000.0)/0.01) + int((15000-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                    elif wvmin<10000.0:
                        idxlf = int((10000.0-wvmin)/0.01) + int((15000-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                    elif wvmin<15000.0:
                        idxlf = int((15000-wvmin)/0.02) + int((wvline-15000.0)/0.03)
                    else:
                        idxlf = int((wvline-wvmin)/0.03)

                idxrg = idxlf + 1

                diffwv=wv[idxrg]-wv[idxlf] #pixel size in wavelength
                midpix=(wv[idxrg]+wv[idxlf])/2 #wavelength between the two pixels
                leftmask = wvline - diffwv/2 #left edge of the mask
                rightmask = wvline + diffwv/2 #right edge of the mask
                frac1 = (midpix - leftmask)/diffwv #fraction of the mask ovelapping the left pixel
                frac2 = (rightmask - midpix)/diffwv #fraction of the mask overlapping the right pixel
                midleft = (leftmask + midpix)/2 #central left overlapp
                midright = (rightmask + midpix)/2 #central wv right overlap
                f1 = f[idxlf] + (midleft-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)
                f2 = f[idxlf] + (midright-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)

                ccf[i]=ccf[i] - f1*fm[j]*frac1 - f2*fm[j]*frac2


    else:
        print('ELSE')
        for i in range(len(rv)):
            wvshift=wvm*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
            #for each mask line
            for j in range(lenm):
                wvline=wvshift[j]

                dist = wvline - wvmin
                k = 0

                while dist >= 0.0:
                    k += 1
                    dist = wvline - wv[k]

                idxlf = int(k-1)

                idxrg = idxlf + 1

                diffwv=wv[idxrg]-wv[idxlf] #pixel size in wavelength
                midpix=(wv[idxrg]+wv[idxlf])/2 #wavelength between the two pixels
                leftmask = wvline - diffwv/2 #left edge of the mask
                rightmask = wvline + diffwv/2 #right edge of the mask
                frac1 = (midpix - leftmask)/diffwv #fraction of the mask ovelapping the left pixel
                frac2 = (rightmask - midpix)/diffwv #fraction of the mask overlapping the right pixel
                midleft = (leftmask + midpix)/2 #central left overlapp
                midright = (rightmask + midpix)/2 #central wv right overlap
                f1 = f[idxlf] + (midleft-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)
                f2 = f[idxlf] + (midright-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)

                ccf[i]=ccf[i] - f1*fm[j]*frac1 - f2*fm[j]*frac2

    return (ccf-np.min(ccf))/np.max((ccf-np.min(ccf)))

@nb.njit(cache=True,error_model='numpy')
def weight_mask(wvi,wvf,o_weight,wvm,fm):
    j=0
    maxj=len(wvi)

    for i in range(len(wvm)):

        if wvm[i]<wvi[j]:
            fm[i]=0.0
        elif wvm[i]>=wvi[j] and wvm[i]<=wvf[j]:
            fm[i]=fm[i]*o_weight[j]
        elif wvm[i]>wvf[j]:
            j+=1
            if j>=maxj:
                fm[i]=0.0
                break
            else:
                i-=1

    return wvm, fm

@nb.njit(cache=True,error_model='numpy')
def polar2colatitude_nb(r,a,i):
    '''Enters the polars coordinates and the inclination i (with respect to the north pole, i=0 makes transits, 90-(inclination defined in exoplanets))
    Returns the colatitude in the star (90-latitude)
    '''
    a=a*m.pi/180.
    i=-i #negative to make the rotation toward the observer.
    theta=m.acos(r*m.sin(a)*m.cos(i)-m.sin(i)*m.sqrt(1-r*r))
    return theta

@nb.njit(cache=True,error_model='numpy')
def polar2longitude_nb(r,a,i):
    '''Enters the polars coordinates and the inclination i (with respect to the north pole, i=0 makes transits, 90-(inclination defined in exoplanets))
    Returns the longitude in the star (from -90 to 90)
    '''
    a=a*m.pi/180.
    i=-i #negative to make the rotation toward the observer.
    h=m.sqrt((1.-(r*m.cos(a))**2.)/(m.tan(i)**2.+1.)) #heigh of the terminator (long=pi/2)
    if r*np.sin(a)>h:
        phi=m.asin(-r*m.cos(a)/m.sqrt(1.-(r*m.sin(a)*m.cos(i)-m.sin(i)*m.sqrt(1.-r*r))**2.))+m.pi #to correct for mirroring of longitudes in the terminator

    else:
        phi=m.asin(r*m.cos(a)/m.sqrt(1.-(r*m.sin(a)*m.cos(i)-m.sin(i)*m.sqrt(1.-r*r))**2.))

    return phi

@nb.njit(cache=True,error_model='numpy')
def speed_bisector_nb(rv,ccf,integrated_bis):
    ''' Fit the bisector of the CCF with a 5th deg polynomial
    '''
    idxmax=ccf.argmax()
    maxccf=ccf[idxmax]
    maxrv=rv[idxmax]

    xnew = rv
    ynew = ccf


    cutleft=0
    cutright=len(ynew)-1
    # if not integrated_bis: #cut the CCF at the minimum of the wings only for reference CCF, if not there are errors.
    for i in range(len(ynew)):
        if xnew[i]>maxrv:
            if ynew[i]>ynew[i-1] and ynew[i]<0.8*maxccf:
                cutright=i
                break

    for i in range(len(ynew)):
        if xnew[-1-i]<maxrv:
            if ynew[-1-i]>ynew[-i] and ynew[i]<0.8*maxccf:
                cutleft=len(ynew)-i
                break

    xnew=xnew[cutleft:cutright]
    ynew=ynew[cutleft:cutright]
    
    minright=np.min(ynew[xnew>maxrv])
    minleft=np.min(ynew[xnew<maxrv])
    minccf=np.max(np.array([minright,minleft]))
    
    if integrated_bis:
        ybis=np.linspace(minccf+0.1*(maxccf-minccf),0.99*maxccf,50) #from 5% to maximum
    else:
        ybis=np.linspace(minccf+0.1*(maxccf-minccf),0.999*maxccf,50) #from 5% to maximum
    xbis=np.zeros(len(ybis))


    for i in range(len(ybis)):
        for j in range(len(ynew)-1):
            if ynew[j]<ybis[i] and ynew[j+1]>ybis[i] and xnew[j]<maxrv:
                rv1=xnew[j]+(xnew[j+1]-xnew[j])*(ybis[i]-ynew[j])/(ynew[j+1]-ynew[j])
            if ynew[j]>ybis[i] and ynew[j+1]<ybis[i] and xnew[j+1]>maxrv:
                rv2=xnew[j]+(xnew[j+1]-xnew[j])*(ybis[i]-ynew[j])/(ynew[j+1]-ynew[j])
        xbis[i]=(rv1+rv2)/2.0 #bisector
    # xbis[-1]=maxrv #at the top should be max RV

    return cutleft,cutright,xbis,ybis


@nb.njit(cache=True,error_model='numpy')
def limb_darkening_law(LD_law,LD1,LD2,amu):

    if LD_law == 'linear':
        mu=1-LD1*(1-amu)

    elif LD_law == 'quadratic':
        a=2*np.sqrt(LD1)*LD2
        b=np.sqrt(LD1)*(1-2*LD2)
        mu=1-a*(1-amu)-b*(1-amu)**2

    elif LD_law == 'sqrt':
        a=np.sqrt(LD1)*(1-2*LD2) 
        b=2*np.sqrt(LD1)*LD2
        mu=1-a*(1-amu)-b*(1-np.sqrt(amu))

    elif LD_law == 'log':
        a=LD2*LD1**2+1
        b=LD1**2-1
        mu=1-a*(1-amu)-b*amu*(1-np.log(amu))

    else:
        print('LD law not valid.')

    return mu

@nb.njit(cache=True,error_model='numpy')
def compute_spot_position(t,spot_map,ref_time,Prot,diff_rot,Revo):
    pos=np.zeros((len(spot_map),4))

    for i in range(len(spot_map)):
        tini = spot_map[i][1] #time of spot apparence
        dur = spot_map[i][2] #duration of the spot
        tfin = tini + dur #final time of spot
        colat = spot_map[i][3] #colatitude
        lat = 90 - colat #latitude
        longi = spot_map[i][4] #longitude
        Rcoef = spot_map[i][5:8] #coefficients for the evolution od the radius. Depends on the desired law.

        pht = longi + (t-ref_time)/Prot%1*360
        #update longitude adding diff rotation
        phsr= pht + (t-ref_time)*diff_rot*(1.698*m.sin(np.deg2rad(lat))**2+2.346*m.sin(np.deg2rad(lat))**4)


        if Revo == 'constant':
            if t>=tini and t<=tfin: 
                rad=Rcoef[0] 
            else:
                rad=0.0
        elif Revo == 'linear':
            if t>=tini and t<=tfin:
                rad=Rcoef[0]+(t-tini)*(Rcoef[1]-Rcoef[0])/dur
            else:
                rad=0.0
        elif Revo == 'quadratic':
            if t>=tini and t<=tfin:
                rad=-4*Rcoef[0]*(t-tini)*(t-tini-dur)/dur**2
            else:
                rad=0.0
        
        else:
            print('Spot evolution law not implemented yet. Only constant and linear are implemented.')
        

        pos[i]=np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad)])
        #return position and radii of spots at t in radians.

    return pos


@nb.njit(cache=True,error_model='numpy')
def compute_planet_pos(t,esinw,ecosw,T0p,Pp,rad_pl,b,a,alp):
    
    if(esinw==0 and ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=m.sqrt(esinw**2+ecosw**2)
       omega=m.atan2(esinw,ecosw)

    t_peri = Ttrans_2_Tperi(T0p,Pp, ecc, omega)
    sinf,cosf=true_anomaly(t,Pp,ecc,t_peri)


    cosftrueomega=cosf*m.cos(omega+m.pi/2)-sinf*m.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
    sinftrueomega=cosf*m.sin(omega+m.pi/2)+sinf*m.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

    if cosftrueomega>0.0: return np.array([1+rad_pl*2, 0.0, rad_pl]) #avoid secondary transits

    cosi = (b/a)*(1+esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

    rpl=a*(1-ecc**2)/(1+ecc*cosf)
    xpl=rpl*(-m.cos(alp)*sinftrueomega-m.sin(alp)*cosftrueomega*cosi)
    ypl=rpl*(m.sin(alp)*sinftrueomega-m.cos(alp)*cosftrueomega*cosi)

    rhopl=m.sqrt(ypl**2+xpl**2)
    thpl=m.atan2(ypl,xpl)
    pos=np.array([rhopl, thpl, rad_pl],dtype=np.float64) #rho, theta, and radii (in Rstar) of the planet
    return pos


@nb.njit(cache=True,error_model='numpy')
def Ttrans_2_Tperi(T0, P, e, w):

    f = m.pi/2 - w
    E = 2 * m.atan(m.tan(f/2.) * m.sqrt((1.-e)/(1.+e)))  # eccentric anomaly
    Tp = T0 - P/(2*np.pi) * (E - e*m.sin(E))      # time of periastron

    return Tp

@nb.njit(cache=True,error_model='numpy')
def true_anomaly(x,period,ecc,tperi):
    fmean=2.0*m.pi*(x-tperi)/period
    #Solve by Newton's method x(n+1)=x(n)-f(x(n))/f'(x(n))
    fecc=fmean
    diff=1.0
    while(diff>1.0E-6):
        fecc_0=fecc
        fecc=fecc_0-(fecc_0-ecc*m.sin(fecc_0)-fmean)/(1.0-ecc*m.cos(fecc_0))
        diff=m.fabs(fecc-fecc_0)
    sinf=m.sqrt(1.0-ecc*ecc)*m.sin(fecc)/(1.0-ecc*m.cos(fecc))
    cosf=(m.cos(fecc)-ecc)/(1.0-ecc*m.cos(fecc))
    return sinf, cosf
########################################################################################
########################################################################################
#                              SPECTROSCOPY FUNCTIONS  FOR SPHERICAL GRID              #
########################################################################################
########################################################################################



#with this the x and y width of each grid is the same, thus the area of the grids is similar in all the sphere, avoiding an over/under sampling of the poles/center
@nb.njit(cache=True,error_model='numpy')
def generate_grid_coordinates_nb(N):

    Nt=2*N-1 #N is number of concentric rings. Nt is counting them two times minus the center one.
    width=180.0/(2*N-1) #width of one grid element.

    centres=np.append(0,np.linspace(width,90-width/2,N-1)) #colatitudes of the concentric grids. The pole of the grid faces the observer.
    anglesout=np.linspace(0,360-width,2*Nt) #longitudes of the grid edges of the most external grid. This grids fix the area of the grids in other rings.
    
    radi=np.sin(np.pi*centres/180) #projected polar radius of the ring.
    amu=np.cos(np.pi*centres/180) #amus

    ts=[0.0] #central grid radius
    alphas=[0.0] #central grid angle

    area=[2.0*np.pi*(1.0-np.cos(width*np.pi/360.0))] #area of spherical cap (only for the central element)
    parea=[np.pi*np.sin(width*np.pi/360.0)**2]

    Ngrid_in_ring=[1]
    
    for i in range(1,len(amu)): #for each ring except firs
        Nang=int(round(len(anglesout)*(radi[i]))) #Number of longitudes to have grids of same width
        w=360/Nang #width i angles
        Ngrid_in_ring.append(Nang)

        angles=np.linspace(0,360-w,Nang)
        area.append(radi[i]*width*w*np.pi*np.pi/(180*180)) #area of each grid
        parea.append(amu[i]*area[-1]) #PROJ. AREA OF THE GRID

        for j in range(Nang):
            ts.append(centres[i]) #latitude
            alphas.append(angles[j]) #longitude


    alphas=np.array(alphas) #longitude of grid (pole faces observer)
    ts=np.array(ts) #colatitude of grid
    Ngrids=len(ts)  #number of grids

    rs = np.sin(np.pi*ts/180) #projected polar radius of grid

    xs = np.cos(np.pi*ts/180) #grid elements in cartesian coordinates. Note that pole faces the observer.
    ys = rs*np.sin(np.pi*alphas/180)
    zs = -rs*np.cos(np.pi*alphas/180)

    return Ngrids,Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea



@nb.njit(cache=True,error_model='numpy')
def loop_compute_immaculate_nb(N,Ngrid_in_ring,ccf_tot,rvel,rv,rvs_ring,ccf_ring):
    #CCF of each pixel, adding doppler and interpolating
    iteration=0
    #Compute the position of the grid projected on the sphere and its radial velocity.
    for i in range(0,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #loop for each grid in the ring
            ccf_tot[iteration,:]=ccf_tot[iteration,:]+interpolation_nb(rv,rvs_ring[i,:] + rvel[iteration],ccf_ring[i,:],ccf_ring[i,0],ccf_ring[i,-1]) 
            iteration=iteration+1

    return ccf_tot



@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,ccf_ph,ccf_sp,ccf_fc,ccf_ph_tot,vis, active_region_types):
    #define things
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    ccf_tot = ccf_ph_tot


    vis_spots_idx=[]
    for i in range(len(vis)-1):
        if vis[i]==1.0:
            vis_spots_idx.append(i)
    ###################### CENTRAL GRID ###############################
    #Central grid is different since it is a circle. initialize values.
    ####################################################################
    dsp=0.0 #fraction covered by each spot
    dfc=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    iteration = 0
    
    for l in vis_spots_idx: #for each spot

        if spot_pos[l][2]==0.0:
            continue

        dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #compute the distance to the grid

        if active_region_types[l] == 0:
            if dist>(width/2+spot_pos[l][2]):
                dsp=0.0
            else:
                if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                    if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                        dsp=1.0
                    else:  #grid partially covered
                        dsp=-(dist-spot_pos[l][2]-width/2)/width

                else: #the grid can completely cover the spot, two cases:
                    if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                        dsp=(2*spot_pos[l][2]/width)**2                 
                    else: #grid partially covered
                        dsp=-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2


            asp+=dsp

        elif active_region_types[l] == 1:
        #FACULA
            # if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
            #     continue 

            if dist>(width/2+spot_pos[l][2]):
                dfc=0.0
            else:
                if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                    if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                        dfc=1.0 
                    else:  #grid partially covered
                        dfc=-(dist-spot_pos[l][2]-width/2)/width 

                else: #if the grid can completely cover the spot, two cases:
                    if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                        dfc=(2*spot_pos[l][2]/width)**2              
                    else: #grid partially covered
                        dfc =-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2 
                dfc -= dsp
            afc+=dfc

    #PLANET
    if simulate_planet:
        if vis[-1]==1.0:
            dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
            
            width2=2*m.sin(width/2)
            if dist>width2/2+planet_pos[2]: apl=0.0
            elif dist<planet_pos[2]-width2/2: apl=1.0
            else: apl=-(dist-planet_pos[2]-width2/2)/width2
    
    if afc < 0:
        afc=0.0
    
    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>1.0:
        apl=1.0
        asp=0.0
        afc=0.0

    if afc + asp > 1.0:
        afc = 1.0 - asp

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)

    aph=1-asp-afc-apl         

    #add the corresponding ccf to the total CCF
    ccf_tot = ccf_tot  - (1-aph)*ccf_ph[iteration] + asp*ccf_sp[iteration] + afc*ccf_fc[iteration]     

    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]


    ############### OTHER GRIDS #######################
    # NOW DO THE SAME FOR THE REST OF GRIDS
    ###################################################
    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            iteration+=1

            dsp=0.0 #fraction covered by each spot
            dfc=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            for l in vis_spots_idx:
                
                if spot_pos[l][2]==0.0: #if radius=0, there is no spot, jump to next spot with continue
                    continue 

                dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #distance between spot centre and grid,multiplying two unit vectors


                #SPOT
                if active_region_types[l] == 0:
                    if dist>(width/2+spot_pos[l][2]): #grid not covered 
                        dsp=0.0
                    
                    else:
                        if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                            if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                                dsp=1.0
                            else:  #grid partially covered
                                dsp=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))

                        elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                            if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                                dsp=(np.pi/4)*(2*spot_pos[l][2]/width)**2                 
                            else: #grid partially covered
                                dsp=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                        else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                            A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                            A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                            Ar=4*(A1+A2)/width**2
                            dsp=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])

                    asp+=dsp
                #FACULA
                elif active_region_types[l] == 1:
                    # if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
                    #     continue 
                    
                    if dist>(width/2+spot_pos[l][2]): #grid not covered by faculae
                        dfc=0.0
                    
                    else:
                        if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                            if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                                dfc=1.0 #subtract spot
                            else:  #grid partially covered
                                dfc=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))
                        elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                            if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                                dfc=(np.pi/4)*(2*spot_pos[l][2]/width)**2              
                            else: #grid partially covered
                                dfc=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                        else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                            A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                            A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                            Ar=4*(A1+A2)/width**2
                            dfc=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])
                        dfc -= dsp
                    afc+=dfc


            #PLANET
            if simulate_planet:
                if vis[-1]==1.0:
                    dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
                    
                    width2=amu[i]*width
                    if dist>width2/2+planet_pos[2]: apl=0.0
                    elif dist<planet_pos[2]-width2/2: apl=1.0
                    else: apl=-(dist-planet_pos[2]-width2/2)/width2

            if afc < 0:
                afc=0.0

            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0

            if afc + asp > 1.0:
                afc = 1.0 - asp

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl          
  
            #add the corresponding ccf to the total CCF
            ccf_tot = ccf_tot  - (1-aph)*ccf_ph[iteration] + asp*ccf_sp[iteration] + afc*ccf_fc[iteration] 

            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])


    return ccf_tot,typ, Aph, Asp, Afc, Apl


@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_nb_sdo(N,Ngrid_in_ring,pare,rs,ccf_ph,ccf_sp,ccf_fc,ccf_ph_tot,sdo_input):
    #define things
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    ccf_tot = ccf_ph_tot

    [array_sp, array_fc] = sdo_input
    n_pxls = len(array_sp)
    typ_cell,_, _ = projection_pxl_to_ss_grid(Ngrid_in_ring, rs, n_pxls)

    aph=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    # Central cell
    k = 0

    idx = typ_cell == k
    n_tot = np.nansum(idx.ravel())

    if n_tot > 0:
        asp = np.nansum(array_sp.ravel()[idx.ravel()]) / n_tot
        afc = np.nansum(array_fc.ravel()[idx.ravel()]) / n_tot


    if afc<0.0:
        afc=0.0
            
    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>1.0:
        apl=1.0
        asp=0.0
        afc=0.0

    if afc + asp > 1.0:
        afc = 1.0 - asp

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)
    
    aph=1-asp-afc-apl  
    

    ccf_tot = ccf_tot  - (1-aph)*ccf_ph[k] + asp*ccf_sp[k] + afc*ccf_fc[k]
    
    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]

    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            k+=1
            aph=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            idx = typ_cell == k
            n_tot = np.nansum(idx.ravel())

            if n_tot > 0:
                asp = np.nansum(array_sp.ravel()[idx.ravel()]) / n_tot
                afc = np.nansum(array_fc.ravel()[idx.ravel()]) / n_tot

            if afc<0.0:
                afc=0
                
            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0

            if afc + asp > 1.0:
                afc = 1.0 - asp

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)
            
            aph=1-asp-afc-apl


            ccf_tot = ccf_tot  - (1-aph)*ccf_ph[k] + asp*ccf_sp[k] + afc*ccf_fc[k]

            
            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])

    return ccf_tot,typ, Aph, Asp, Afc, Apl

@nb.njit(cache=True,error_model='numpy')
def projection_pxl_to_ss_grid(Ngrid_in_ring, rs, n_pxls):
    N = len(Ngrid_in_ring)
    x = np.linspace(-1.0, 1.0, n_pxls)
    xg = np.empty((n_pxls, n_pxls))
    yg = np.empty((n_pxls, n_pxls))
    rg = np.empty((n_pxls, n_pxls))
    theta_g = np.empty((n_pxls, n_pxls))
    
    for i in range(n_pxls):
        for j in range(n_pxls):
            x_val = x[j]
            y_val = x[i]
            r_val = np.sqrt(x_val**2 + y_val**2)
            if r_val > 1.0:
                xg[i, j] = np.nan
                yg[i, j] = np.nan
                rg[i, j] = np.nan
                theta_g[i, j] = np.nan
            else:
                xg[i, j] = x_val
                yg[i, j] = y_val
                rg[i, j] = r_val
                angle = np.arctan2(x_val, y_val) + np.pi / 2.0
                theta_g[i, j] = angle if angle >= 0 else angle + 2 * np.pi

    # Compute lim_r from rs
    r = np.empty_like(rs)
    for i in range(rs.shape[0]):
        r[i] = rs[i]
    r_sorted = np.sort(r)
    r_unique = np.empty_like(r)
    count = 0
    for i in range(r_sorted.shape[0]):
        val = np.round(r_sorted[i], 6)
        if count == 0 or val != r_unique[count - 1]:
            r_unique[count] = val
            count += 1
    r_unique = r_unique[:count]

    lim_r = np.empty(N)
    for i in range(N - 1):
        lim_r[i] = (r_unique[i + 1] + r_unique[i]) / 2.0
    lim_r[N - 1] = 1.0


    typ_cell = np.zeros((n_pxls, n_pxls))
    
    for i in range(n_pxls):
        for j in range(n_pxls):
            r_val = rg[i, j]
            if not np.isnan(r_val):
                # Find ring
                ring = 0
                while ring < N and lim_r[ring] < r_val:
                    ring += 1
                if ring >= N:
                    ring = N - 1
                offset = 0
                for k in range(ring):
                    offset += Ngrid_in_ring[k]
                n_cells = Ngrid_in_ring[ring]
                angle = theta_g[i, j]
                idx = int(np.round(((angle + np.pi / 2.0) % (2 * np.pi)) / (2 * np.pi / n_cells))) % n_cells
                typ_cell[i, j] = offset + idx
            else:
                typ_cell[i, j] = np.nan

    # Flip horizontally
    typ_cell = typ_cell[:, ::-1]
    return typ_cell, xg, yg



@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_lc_nb_sdo(N,Ngrid_in_ring,pare,rs,bph,bsp,bfc,flxph,sdo_input):
    flux = flxph
    [array_sp, array_fc] = sdo_input
    n_pxls = len(array_sp)
    typ_cell,_, _ = projection_pxl_to_ss_grid(Ngrid_in_ring, rs, n_pxls)
    aph=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    # Central cell
    k = 0

    idx = typ_cell == k
    n_tot = np.nansum(idx.ravel())

    if n_tot > 0:
        asp = np.nansum(array_sp.ravel()[idx.ravel()]) / n_tot
        afc = np.nansum(array_fc.ravel()[idx.ravel()]) / n_tot


    if afc<0.0:
        afc=0.0
            
    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>1.0:
        apl=1.0
        asp=0.0
        afc=0.0

    if afc + asp > 1.0:
        afc = 1.0 - asp

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)
    
    aph=1-asp-afc-apl  
    

    #add the corresponding flux to the total flux
    flux = flux - (1-aph)*bph[0]+asp*bsp[0]+bfc[0]*afc
    
    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]

    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            k+=1
            aph=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            idx = typ_cell == k
            n_tot = np.nansum(idx.ravel())

            if n_tot > 0:
                asp = np.nansum(array_sp.ravel()[idx.ravel()]) / n_tot
                afc = np.nansum(array_fc.ravel()[idx.ravel()]) / n_tot

            if afc<0.0:
                afc=0
                
            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0

            if afc + asp > 1.0:
                afc = 1.0 - asp

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)
            
            aph=1-asp-afc-apl


            flux = flux - (1-aph)*bph[i]+asp*bsp[i]+bfc[i]*afc
            
            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])
                

    return flux ,typ, Aph, Asp, Afc, Apl



@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_lc_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,bph,bsp,bfc,flxph,vis, active_region_types):
 
    #define things
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    flux = flxph


    vis_spots_idx=[]
    for i in range(len(vis)-1):
        if vis[i]==1.0:
            vis_spots_idx.append(i)
    ###################### CENTRAL GRID ###############################
    #Central grid is different since it is a circle. initialize values.
    ####################################################################
    dsp=0.0 #fraction covered by each spot
    dfc=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    iteration = 0
    
    for l in vis_spots_idx: #for each spot

        if spot_pos[l][2]==0.0:
            continue

        dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #compute the distance to the grid

        #SPOT
        if active_region_types[l] == 0:

            if dist>(width/2+spot_pos[l][2]):
                dsp=0.0
            else:
                if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                    if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                        dsp=1.0
                    else:  #grid partially covered
                        dsp=-(dist-spot_pos[l][2]-width/2)/width

                else: #the grid can completely cover the spot, two cases:
                    if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                        dsp=(2*spot_pos[l][2]/width)**2                 
                    else: #grid partially covered
                        dsp=-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2


            asp+=dsp
        #FACULA
        elif active_region_types[l] == 1:


            if dist>(width/2+spot_pos[l][2]):
                dfc=0.0
            else:
                if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                    if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                        dfc=1.0 
                    else:  #grid partially covered
                        dfc=-(dist-spot_pos[l][2]-width/2)/width 

                else: #if the grid can completely cover the spot, two cases:
                    if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                        dfc=(2*spot_pos[l][2]/width)**2               
                    else: #grid partially covered
                        dfc =-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2 
                dfc -= dsp
            afc+=dfc

    #PLANET
    if simulate_planet:
        if vis[-1]==1.0:
            dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
            
            width2=2*m.sin(width/2)
            if dist>width2/2+planet_pos[2]: apl=0.0
            elif dist<planet_pos[2]-width2/2: apl=1.0
            else: apl=-(dist-planet_pos[2]-width2/2)/width2

    if afc<0.0:
        afc=0.0

    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>1.0:
        apl=1.0
        asp=0.0
        afc=0.0

    if afc + asp > 1.0:
        afc = 1.0 - asp

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)

    aph=1-asp-afc-apl  

    #add the corresponding flux to the total flux
    flux = flux - (1-aph)*bph[0]+asp*bsp[0]+bfc[0]*afc
    

    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]

    ############### OTHER GRIDS #######################
    # NOW DO THE SAME FOR THE REST OF GRIDS
    ###################################################
    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            iteration+=1

            dsp=0.0 #fraction covered by each spot
            dfc=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            for l in vis_spots_idx:
                
                if spot_pos[l][2]==0.0: #if radius=0, there is no spot, jump to next spot with continue
                    continue 

                dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #distance between spot centre and grid,multiplying two unit vectors


                #SPOT
                if active_region_types[l] == 0:
                    if dist>(width/2+spot_pos[l][2]): #grid not covered 
                        dsp=0.0
                    
                    else:
                        if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                            if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                                dsp=1.0
                            else:  #grid partially covered
                                dsp=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))

                        elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                            if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                                dsp=(np.pi/4)*(2*spot_pos[l][2]/width)**2                 
                            else: #grid partially covered
                                dsp=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                        else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                            A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                            A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                            Ar=4*(A1+A2)/width**2
                            dsp=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])

                    asp+=dsp

                #FACULA
                elif active_region_types[l] == 1:
                    # if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
                    #     continue 
                    
                    if dist>(width/2+spot_pos[l][2]): #grid not covered by faculae
                        dfc=0.0
                    
                    else:
                        if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                            if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                                dfc=1.0 #subtract spot
                            else:  #grid partially covered
                                dfc=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))

                        elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                            if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                                dfc=(np.pi/4)*(2*spot_pos[l][2]/width)**2               
                            else: #grid partially covered
                                dfc=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                        else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                            A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                            A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                            Ar=4*(A1+A2)/width**2
                            dfc=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])
                        dfc -= dsp
                    afc+=dfc


            #PLANET
            if simulate_planet:
                if vis[-1]==1.0:
                    dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
                    
                    width2=amu[i]*width
                    if dist>width2/2+planet_pos[2]: apl=0.0
                    elif dist<planet_pos[2]-width2/2: apl=1.0
                    else: apl=-(dist-planet_pos[2]-width2/2)/width2

            if afc<0.0:
                afc=0.0
        
            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0

            if afc + asp > 1.0:
                afc = 1.0 - asp

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl 
          
            #add the corresponding ccf to the total CCF
            flux = flux - (1-aph)*bph[i]+asp*bsp[i]+bfc[i]*afc

            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])
            

    return flux ,typ, Aph, Asp, Afc, Apl, typ





###############
#CCFS
###############
@nb.njit(cache=True,error_model='numpy') 
def fun_spot_bisect(ccf):
    rv=-1.51773453*ccf**4 +3.52774949*ccf**3 -3.18794328*ccf**2 +1.22541774*ccf -0.22479665 #Polynomial fit to ccf in Fig 2 of Dumusque 2014, plus 400m/s to match Fig6 in Herrero 2016
    return rv


@nb.njit(cache=True,error_model='numpy') 
def fun_cifist(ccf,amu):
    '''Interpolate the cifist bisectors as a function of the projected angle
    '''
    # amv=np.arange(1,0.0,-0.1) #list of angles defined in cfist
    amv=np.arange(0.0,1.01,0.1) #list of angles defined in cfist

    idx_upp=np.searchsorted(amv,amu*0.999999999,side='right')
    idx_low=idx_upp-1


    cxm=np.zeros((len(amv),7)) #coeff of the bisectors. NxM, N is number of angles, M=7, the degree of the polynomial
    #PARAMS COMPUTED WITH HARPS MASK
    cxm[10,:]=np.array([-3.51974861,11.1702017,-13.22368296,6.67694456,-0.63201573,-0.44695616,-0.36838495]) #1.0
    cxm[9,:]=np.array([-4.05903967,13.21901003,-16.47215949,9.51023171,-2.13104764,-0.05153799,-0.36973749]) #0.9
    cxm[8,:]=np.array([-3.92153131,12.76694663,-15.96958217,9.39599116,-2.34394028,0.12546611,-0.42092905]) #0.8
    cxm[7,:]=np.array([-3.81892968,12.62209118,-16.06973368,9.71487198,-2.61439945,0.25356088,-0.43310756]) #0.7
    cxm[6,:]=np.array([-5.37213406,17.6604689,-22.52477323,13.91461247,-4.13186181,0.60271171,-0.46427559]) #0.6
    cxm[5,:]=np.array([-6.35351933,20.92046705,-26.83933359,16.86220487,-5.28285592,0.90643187,-0.47696283]) #0.5
    cxm[4,:]=np.array([-7.67270144,25.60866105,-33.4381214,21.58855269,-7.1527039,1.35990694,-0.48001707]) #0.4
    cxm[3,:]=np.array([-9.24152009,31.09337903,-41.07410957,27.04196984,-9.32910982,1.89291407,-0.455407]) #0.3
    cxm[2,:]=np.array([-11.62006536,39.30962189,-52.38161244,34.98243089,-12.40650704,2.57940618,-0.37337442]) #0.2
    cxm[1,:]=np.array([-14.14768805,47.9566719,-64.20294114,43.23156971,-15.57423374,3.13318175,-0.14451226]) #0.1
    cxm[0,:]=np.array([-16.67531074,56.60372191,-76.02426984,51.48070853,-18.74196044,3.68695732,0.0843499 ]) #0.0

    #interpolate
    cxu=cxm[idx_low]+(cxm[idx_upp]-cxm[idx_low])*(amu-amv[idx_low])/(amv[idx_upp]-amv[idx_low])

    rv = cxu[0]*ccf**6 + cxu[1]*ccf**5 + cxu[2]*ccf**4 + cxu[3]*ccf**3 + cxu[4]*ccf**2 + cxu[5]*ccf + cxu[6]
    return rv






@nb.njit(cache=True,error_model='numpy') 
def check_spot_overlap(spot_map,Q): #TODO
#False if there is no overlap between spots
    N_spots=len(spot_map)
    for i in range(N_spots):
        for j in range(i+1,N_spots):
            t_ini_0 = spot_map[i][1]
            t_ini = spot_map[j][1]
            t_fin_0 = t_ini_0 + spot_map[i][2]
            t_fin = t_ini + spot_map[j][2]
            r_0 = np.max(spot_map[i][5:7])
            r = np.max(spot_map[j][5:7])
            th_0 = m.pi/2-spot_map[i][3]*m.pi/180 #latitude in radians
            th = m.pi/2-spot_map[j][3]*m.pi/180 #latitude in radians
            ph_0 = spot_map[i][4]*m.pi/180 #longitude in radians
            ph = spot_map[j][4]*m.pi/180 #longitude in radians
            

            dist = m.acos(m.sin(th_0)*m.sin(th) + m.cos(th_0)*m.cos(th)*m.cos(m.fabs(ph_0 - ph)))*180/m.pi #in

            if (dist<m.sqrt(Q[i]+1)*(r_0+r)) and not ((t_ini>t_fin_0) or (t_ini_0>t_fin)): #if they touch and coincide in time
                return True
            
    return False







########################################################################################
########################################################################################
#           SPECTROPHOTOMETRY FUNCTIONS ('spec', OSCAR - under development)            #
########################################################################################
########################################################################################

#TODO

#spec[k,:],typ, filling_ph[k], filling_sp[k], filling_fc[k], filling_pl[k] = nbspectra.loop_generate_rotating_spec_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,spec_rings_ph,spec_rings_sp,spec_rings_fc,spec_ph,vis)
@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_spec_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,spec_rings_ph,spec_rings_sp,spec_rings_fc,spec_ph,vis):
 

    #define things
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    spec = spec_ph


    vis_spots_idx=[]
    for i in range(len(vis)-1):
        if vis[i]==1.0:
            vis_spots_idx.append(i)
    ###################### CENTRAL GRID ###############################
    #Central grid is different since it is a circle. initialize values.
    ####################################################################
    dsp=0.0 #fraction covered by each spot
    dfc=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    iteration = 0
    
    for l in vis_spots_idx: #for each spot

        if spot_pos[l][2]==0.0:
            continue

        dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #compute the distance to the grid

        if dist>(width/2+spot_pos[l][2]):
            dsp=0.0
        else:
            if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                    dsp=1.0
                else:  #grid partially covered
                    dsp=-(dist-spot_pos[l][2]-width/2)/width

            else: #the grid can completely cover the spot, two cases:
                if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                    dsp=(2*spot_pos[l][2]/width)**2                 
                else: #grid partially covered
                    dsp=-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2


        asp+=dsp
        #FACULA
        if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
            continue 

        if dist>(width/2+spot_pos[l][3]):
            dfc=0.0
        else:
            if (width/2)<spot_pos[l][3]: #if the spot can cover completely the grid, two cases:
                if dist<=spot_pos[l][3]-(width/2):  #grid completely covered
                    dfc=1.0 - dsp
                else:  #grid partially covered
                    dfc=-(dist-spot_pos[l][3]-width/2)/width - dsp

            else: #if the grid can completely cover the spot, two cases:
                if dist<=(width/2-spot_pos[l][3]): #all the spot is inside the grid
                    dfc=(2*spot_pos[l][3]/width)**2 - dsp                
                else: #grid partially covered
                    dfc =-2*spot_pos[l][3]*(dist-width/2-spot_pos[l][3])/width**2 - dsp

        afc+=dfc


    #PLANET
    if simulate_planet:
        if vis[-1]==1.0:
            dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
            
            width2=2*m.sin(width/2)
            if dist>width2/2+planet_pos[2]: apl=0.0
            elif dist<planet_pos[2]-width2/2: apl=1.0
            else: apl=-(dist-planet_pos[2]-width2/2)/width2

    
    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>1.0:
        apl=1.0
        asp=0.0
        afc=0.0

    if afc + asp > 1.0:
        afc = 1.0 - asp

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)

    aph=1-asp-afc-apl           

    #add the corresponding spectrum flux to the total spectrum
    spec = spec - (1-aph)*spec_rings_ph[0,:]+asp*spec_rings_sp[0,:]+spec_rings_fc[0,:]*afc
    #spec = spec - (1-aph)*bph[i]+asp*bsp[i]+bfc[i]*afc


    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]


    ############### OTHER GRIDS #######################
    # NOW DO THE SAME FOR THE REST OF GRIDS
    ###################################################
    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            iteration+=1

            dsp=0.0 #fraction covered by each spot
            dfc=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            for l in vis_spots_idx:
                
                if spot_pos[l][2]==0.0: #if radius=0, there is no spot, jump to next spot with continue
                    continue 

                dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #distance between spot centre and grid,multiplying two unit vectors


                #SPOT
                if dist>(width/2+spot_pos[l][2]): #grid not covered 
                    dsp=0.0
                
                else:
                    if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                        if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                            dsp=1.0
                        else:  #grid partially covered
                            dsp=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))

                    elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                        if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                            dsp=(np.pi/4)*(2*spot_pos[l][2]/width)**2                 
                        else: #grid partially covered
                            dsp=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                    else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                        A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                        A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                        Ar=4*(A1+A2)/width**2
                        dsp=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])

                asp+=dsp
                #FACULA
                if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
                    continue 
                
                if dist>(width/2+spot_pos[l][3]): #grid not covered by faculae
                    dfc=0.0
                
                else:
                    if (width/m.sqrt(2))<spot_pos[l][3]: #if the spot can cover completely the grid, two cases:
                        if dist<=(m.sqrt(spot_pos[l][3]**2-(width/2)**2)-width/2):  #grid completely covered
                            dfc=1.0-dsp #subtract spot
                        else:  #grid partially covered
                            dfc=-(dist-spot_pos[l][3]-width/2)/(width+spot_pos[l][3]-m.sqrt(spot_pos[l][3]**2-(width/2)**2))-dsp

                    elif (width/2)>spot_pos[l][3]: #if the grid can completely cover the spot, two cases:
                        if dist<=(width/2-spot_pos[l][3]): #all the spot is inside the grid
                            dfc=(np.pi/4)*(2*spot_pos[l][3]/width)**2-dsp               
                        else: #grid partially covered
                            dfc=(np.pi/4)*((2*spot_pos[l][3]/width)**2-(2*spot_pos[l][3]/width**2)*(dist-width/2+spot_pos[l][3]))-dsp

                    else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                        A1=(width/2)*m.sqrt(spot_pos[l][3]**2-(width/2)**2)
                        A2=(spot_pos[l][3]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][3]**2-(width/2)**2)/spot_pos[l][3]))
                        Ar=4*(A1+A2)/width**2
                        dfc=-Ar*(dist-width/2-spot_pos[l][3])/(width/2+spot_pos[l][3])-dsp

                afc+=dfc


            #PLANET
            if simulate_planet:
                if vis[-1]==1.0:
                    dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
                    
                    width2=amu[i]*width
                    if dist>width2/2+planet_pos[2]: apl=0.0
                    elif dist<planet_pos[2]-width2/2: apl=1.0
                    else: apl=-(dist-planet_pos[2]-width2/2)/width2


            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0

            if afc + asp > 1.0:
                afc = 1.0 - asp

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl 
          
            #add the corresponding spectrum flux to the total spectrum
            spec = spec - (1-aph)*spec_rings_ph[i,:]+asp*spec_rings_sp[i,:]+spec_rings_fc[i,:]*afc

            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])
            

    return spec ,typ, Aph, Asp, Afc, Apl




