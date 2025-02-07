#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:59:23 2018

@author: mbexkes3
"""
from numpy import zeros, pi, exp, log, reshape, sqrt,array
from scipy.optimize import brentq
from scipy.special import erfc

import constants as c
#import namelist as n
#import variables as v

#from functions import surface_tension
def surface_tension(T):
    """surface tension of water - pruppacher and klett p130"""
    TC = T - 273.15
    TC = max(TC, -40)
    surface_tension = (75.93 + 0.115*TC + 6.818e-2*TC**2 +
                       6.511e-3*TC**3 + 2.933e-4*TC**4 +
                       6.283e-6*TC**5 + 5.285e-8*TC**6)
    if TC > 0:
        surface_tension = 76.1 - 0.155*TC
        
    surface_tension = surface_tension*c.JOULES_IN_AN_ERG # convert to J/cm2
    surface_tension = surface_tension*1e4 # convert to J/m2
    return surface_tension

def setup_grid(rhobin, kappabin, rkm, Dlow, nbins, nmodes, RH, sig, NAER, D_AER, T, rhoa, k):
    #set - up some variables
    
    D = zeros(nbins+1)
    NBIN = zeros([nmodes, nbins])
    MBIN = zeros([nmodes, nbins])
    MWAT = zeros([nmodes, nbins])


    def FIND_ERFCINV(x):
        return DUMMY_VARS-erfc(x)

    def kk02(MWAT2):
        """ Kappa Koehler theory, Petters and Kriedenwies (2007)
            
        """
        mass_bin_centre = MBIN[N_SEL, N_SEL2]
       
        rhobin3 = rhoa[N_SEL]
        kappabin3 = k[N_SEL]
                
        RHOAT = MWAT2/c.rhow+(mass_bin_centre/rhobin3)
        RHOAT = (MWAT2+(mass_bin_centre))/RHOAT

        Dw = ((MWAT2 + (mass_bin_centre))*6/(pi*RHOAT))**(1/3)
        Dd = ((mass_bin_centre*6)/(rhobin3*pi))**(1/3)
        #KAPPA = (mass_bin_centre/rhobin3*kappabin3)/(mass_bin_centre/rhobin3)
        KAPPA = kappabin3
        
        sigma = surface_tension(T)
        RH_EQ = (((Dw**3-Dd**3)/(Dw**3-Dd**3*(1-KAPPA))*
                     exp((4*sigma*c.mw)/(c.R*T*c.rhow*Dw)))-RH_ACT)
    
        return RH_EQ
    
    for I1 in range(nmodes):
        DUMMY = NAER[I1]/nbins
        D[0] = Dlow
        NBIN[I1, 0] = NAER[I1]*0.5*erfc(-(log(D[0]/D_AER[I1])/sqrt(2)/sig[I1])) #N(D < Dlow) cumulative number concentration. Note erfc(-t) = 2 - erfc(t) 
        
        for J1 in range(1, nbins): #第0个之前已经赋值了
            D[J1] = (DUMMY+NAER[I1]*0.5*
                     erfc(-(log(D[J1-1]/D_AER[I1])/sqrt(2)/sig[I1])))
            DUMMY_VARS = 2/NAER[I1]*D[J1]
            DUMMY_VARS = min(DUMMY_VARS, 2-1e-10)
            DUMMY2 = brentq(FIND_ERFCINV, -1e10, 1e10, xtol=1e-30)        #scipy.special中的erfcinv(x) 中的x为0<x<2, 所以不适用，此处尚待讨论
            D[J1] = exp(-DUMMY2*sig[I1]*sqrt(2)+log(D_AER[I1]))
            
            NBIN[I1, J1] = (NAER[I1]*(0.5*erfc(-log(D[J1]/D_AER[I1])/sqrt(2.0)/sig[I1])-
                0.5*erfc(-log(D[J1-1]/D_AER[I1])/sqrt(2.0)/sig[I1])))   #Is this necessary? Just equals DUMMY = NAER[I1]/nbins
        
            MBIN[I1,:] = pi/6*D[:-1]**3*rhoa[I1]   #D = zeros(nbins+1) This is mass per particle? not 
   
    for I1 in range(nmodes):
        for J1 in range(nbins):
            N_SEL = I1 #0
            N_SEL2 = J1 #0~70
          
            MULT = -1
         
            MULT = 1
            RH_ACT = min(RH, 0.999)
            MWAT[I1, J1] = brentq(kk02, -1e-30, 1e10, xtol=1e-30, maxiter=500)
            
    NBIN = reshape(NBIN, nmodes*nbins)
    MWAT = reshape(MWAT, nmodes*nbins)

    MBIN = reshape(MBIN, nmodes*nbins)
    return NBIN, MWAT, MBIN, D     

