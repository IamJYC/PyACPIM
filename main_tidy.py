#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:51:12 2017

@author: mbexkes3
"""
import numpy as np
from scipy.misc import derivative

import matplotlib.pyplot as plt

from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

import constants as c
import namelist_torun_test as n
#import namelist as n
import functions as f
import variables as v

import importlib # NOTE: need to be in the correct directory for this to work

import time
def run():
    start_time = time.time()

    importlib.reload(f)
    importlib.reload(v)
    importlib.reload(n)
    # get variables
    [nbins, nmodes, simulation_type, PRESS1, PRESS2, w, rhobin, molwbin,nubin, Kappa,
    Temp1, Temp2, Y, ERROR_FLAG, Y_AER, CDP_CONC_liq, CDP_CONC_ice, CDP_CONC_total,
    YICE_AER, alpha_crit, ncomps, output, output_ice, dummy2, ACT_DROPS, setup_output, 
    Heterogeneous_freezing_criteria, Mass_frac, Mass_frac_aer] = v.run()   ###np.shape(Y[INDSV1:INDSV2]) is (700,)
    dt = 1                                                                 ###np.shape(Y_AER) is (2800,70)
    #### INDEXES ############
    IND1 = nbins*nmodes # index for mass
    IND2 = IND1*2 # index for number
    IND3 = IND1*3 # capacitance
    INDSV1 = IND3
    INDSV2 = INDSV1 + IND1*n.n_sv

    INDSV1_ICE = IND1*3
    INDSV2_ICE = INDSV1_ICE+IND1*n.n_sv

    IPRESS = -3
    ITEMP = -2
    IRH = -1

    IRH_SV = [INDSV2+i for i in range(n.n_sv)] # list of indexs for each semi-vol component's RH
    IRH_SV_ICE = [x+IND1 for x in IRH_SV]

    IPRESS_ICE = -3
    ITEMP_ICE = -2
    IRH_ICE = -1
    
    MBIN2 = np.zeros([len(Mass_frac),IND1])   #(20, 70)
    RHOBIN2 = np.zeros([len(Mass_frac),IND1])
    NUBIN2 = np.zeros([len(Mass_frac),IND1])
    MOLWBIN2 = np.zeros([len(Mass_frac),IND1])
    KAPPABIN2 = np.zeros([len(Mass_frac),IND1])
    MBIN22 = np.zeros([len(Mass_frac),IND1])
    for i,key in enumerate(Mass_frac.keys()): 
         for j in range(nmodes):   #j = 0 if nmodes = 1
             MBIN2[i,IND1*j:IND1*(j+1)] = Mass_frac[key][j]*Y_AER[0,IND1*j:IND1*(1+j)]  # Mass of aerosol bin for one mode shape: (20 (10 cores + 10 SVs), 70)
             RHOBIN2[i,IND1*j:IND1*(j+1)] = c.aerosol_dict[key][1]  
             NUBIN2[i,IND1*j:IND1*(1+j)] = c.aerosol_dict[key][2]
             MOLWBIN2[i,IND1*j:IND1*(1+j)] = c.aerosol_dict[key][0]
             KAPPABIN2[i,IND1*j:IND1*(1+j)] = c.aerosol_dict[key][3]

    ########################
    def run_sim(Y,time,Y_AER1, YICE):
            
        def dy_dt_func(t,Y):

            dy_dt = np.zeros(len(Y))
            
            # add condensed semi-vol mass into bins
            MBIN2[-1*n.n_sv:,:] = np.reshape(Y[INDSV1:INDSV2],[n.n_sv,nbins*nmodes]) #(700,) ---> (10, 70), MBIN2[10:19,:] (which is for 10 SVs), t=1时输入的Y为output[0,:], Core的质量不用变所以就用MBIN2的就行，不用重新代入Y
            # calculate saturation vapour pressure over liquid
            svp1 = f.svp_liq(Y[ITEMP])
            # saturation ratio
            SL = svp1*Y[IRH]/(Y[IPRESS]-svp1)
            SL = (SL*Y[IPRESS]/(1 + SL))/svp1

            # water vapour mixing ratio
            WV = c.eps*Y[IRH]*svp1/(Y[IPRESS] - svp1)
            WL = np.sum(Y[IND1:IND2]*Y[:IND1])   # LIQUID MIXING RATIO
            WI = np.sum(YICE[IND1:IND2]*YICE[:IND1]) # ice mixing ratio
            RM=c.RA+WV*c.RV
            
            CPM=c.CP+WV*c.CPV+WL*c.CPW+WI*c.CPI
            
            if simulation_type.lower() == 'chamber':
                # CHAMBER MODEL - pressure change
                dy_dt[IPRESS] = -100*PRESS1*PRESS2*np.exp(-PRESS2*(time+t))
            elif simulation_type.lower() == 'parcel':
                # adiabatic parcel
                 dy_dt[IPRESS] = -Y[IPRESS]/RM/Y[ITEMP]*c.g*w #! HYDROSTATIC EQUATION
            else:
                print('simulation type unknown')
                return
            
    # ----------------------------change in vapour content: -----------------------
            # 1. equilibruim size of particles
            if n.SV_flag:
                
                dummy = np.sum(np.reshape(Y[INDSV1:INDSV2],[n.n_sv,nbins*nmodes]),axis=0)
                Kappa_sv = np.sum((MBIN2[:,:]/MOLWBIN2[:,:])*KAPPABIN2[:,:],axis=0)/np.sum(MBIN2[:,:]/MOLWBIN2[:,:],axis=0)
                KK01 = f.kk01(Y[0:IND1], Y[ITEMP], Y_AER1+dummy, 
                          rhobin, Kappa_sv)  
                
            else:
                KK01 = f.kk01(Y[0:IND1], Y[ITEMP], Y_AER1, 
                          rhobin, Kappa)
            Dw = KK01[2]    # wet diameter
            RHOAT = KK01[1] # density of particles inc water and aerosol mass
            RH_EQ = KK01[0] # equilibrium diameter---shouldn't be saturation ratio or RH??
            # 2. growth rate of particles, Jacobson p455
            # rate of change of radius
            growth_rate = f.DROPGROWTHRATE(Y[ITEMP],Y[IPRESS],SL,RH_EQ,RHOAT,Dw)
            growth_rate[np.isnan(growth_rate)] = 0# get rid of nans
            growth_rate = np.where(Y[IND1:IND2] < 1e-9, 0.0, growth_rate)
            # 3. Mass of water condensing
            # change in mass of water per particle
            dy_dt[:IND1] = (np.pi*RHOAT*Dw**2)* growth_rate 
            
            # 4. Change in vapour content
            # change in water vapour mixing ratio
            dwv_dt = -1*sum(Y[IND1:IND2]*dy_dt[:IND1]) # change to np.sum for speed  
    # -----------------------------------------------------------------------------

            if simulation_type.lower() == 'chamber':
                # CHAMBER MODEL - temperature change
                dy_dt[ITEMP] = -Temp1*Temp2*np.exp(-Temp2*(time+t))
            elif simulation_type.lower() == 'parcel':
                # adiabatic parcel
                dy_dt[ITEMP] = RM/Y[IPRESS]*dy_dt[IPRESS]*Y[ITEMP]/CPM # TEMPERATURE CHANGE: EXPANSION
                dy_dt[ITEMP] = dy_dt[ITEMP] - c.LV/CPM*dwv_dt        
            else:
                print('simulation type unknown')
                return
    # --------------------------------RH change------------------------------------
            dy_dt[IRH] = svp1*dwv_dt*(Y[IPRESS]-svp1)
            dy_dt[IRH] = dy_dt[IRH] + svp1*WV*dy_dt[IPRESS]
            dy_dt[IRH] = (dy_dt[IRH] - 
                         WV*Y[IPRESS]*derivative(f.svp_liq,Y[ITEMP],dx=1.0)
                         *dy_dt[ITEMP])
            dy_dt[IRH] = dy_dt[IRH]/(c.eps*svp1**2)
    # -----------------------------------------------------------------------------
            
    # ------------------------------ SEMI-VOLATILES -------------------------------
            if n.SV_flag: 
                SV_mass = np.reshape(Y[INDSV1:INDSV2],[n.n_sv,n.nmodes*n.nbins])
                SV_mass = np.where(SV_mass == 0.0,1e-30,SV_mass)
                MBIN2[n.n_sv*-1:,:] = SV_mass
                RH_EQ_SV = f.K01SV(Y[:IND1],Y[ITEMP],MBIN2, 
                                  n.n_sv, RHOBIN2, NUBIN2, MOLWBIN2)
                RH_EQ = RH_EQ_SV[0]
                RHOAT = RH_EQ_SV[1]
                DW = RH_EQ_SV[2] 
                
                SVP_ORG = f.SVP_GASES(n.semi_vols,Y[ITEMP], n.n_sv) #Saturation Vapor Pressure of semi-vols
                
                RH_ORG = [x*Y[IPRESS]/c.RA/Y[ITEMP] for x in Y[IRH_SV]] # mixing ratio*density of air (Ra is the individual gas constant for air, and P of air = ρRaT
                RH_ORG = [(x/c.aerosol_dict[key][0])*c.R*Y[ITEMP] for x, key in zip(RH_ORG,n.semi_vols[:n.n_sv])] # just for n_sv keys in dictionary ###  (ρair*mixing ratio)R/Mi*T = (ρair*mixing ratio)*Ri*T = ρiRiT = Pi (partial pressure of semi-vols)
                RH_ORG = [RH_ORG[x]/SVP_ORG[x] for x in range(n.n_sv)] #Relative humidity of semi-vols
                
                dy_dt[INDSV1:INDSV2] = f.SVGROWTHRATE(Y[ITEMP], Y[IPRESS], SVP_ORG,
                                                      RH_ORG, RH_EQ,DW, n.n_sv, n.nbins, n.nmodes,MOLWBIN2)

                dy_dt[INDSV1:INDSV2] = np.where(dy_dt[INDSV1:INDSV2] < 0, 0, dy_dt[INDSV1:INDSV2]) # ??? so no evaporation of semi_vols?
                dy_dt[IRH_SV] = -np.sum(np.reshape(dy_dt[INDSV1:INDSV2],[n.n_sv,IND1])*Y[IND1:IND2],axis=1)

            return dy_dt
        
    #--------------------- SET-UP solver ------------------------------------------
        
        y0 = Y
        t0 = 0.0
        
        #define assimulo problem
        exp_mod = Explicit_Problem(dy_dt_func,y0,t0)
        
        # define an explicit solver
        exp_sim = CVode(exp_mod)
        exp_sim.iter = 'Newton'
        exp_sim.discr = 'BDF'
        #set parameters
        tol_list = np.zeros_like(Y)
        tol_list[0:IND1] = 1e-40 # this is now different to ACPIM (1e-25)
        tol_list[IND1:IND2] = 10 # number
        
        tol_list[IND2:IND3] = 1e-30 # capacitance
        tol_list[IND3:INDSV2] = 1e-26 #condendensed semi-vol mass
        tol_list[IRH_SV] = 1e-26 # RH of each semi-vol compound
        
        tol_list[IPRESS] = 10
        tol_list[ITEMP] =1e-4
        tol_list[IRH] = 1e-8 # set tolerance for each dydt function
        exp_sim.atol = tol_list
        exp_sim.rtol = 1.0e-8
        exp_sim.inith = 0 # initial time step-size
        exp_sim.usejac = False
        exp_sim.maxncf = 100 # max number of convergence failures allowed by solver
        exp_sim.verbosity = 40
        t_output,y_output = exp_sim.simulate(1)
     
        
            
        return y_output[-1,:], t_output[:]


    def run_sim_ice(Y,YLIQ):
         
        def dy_dt_func(t,Y):
            
            dy_dt = np.zeros(len(Y))
            
            svp = f.svp_liq(Y[ITEMP])
            svp_ice = f.svp_ice(Y[ITEMP])
            
            # vapour mixing ratio
            WV = c.eps*Y[IRH_ICE]*svp/(Y[IPRESS_ICE] - svp)
            # liquid mixing ratio
            WL = sum(YLIQ[IND1:IND2]*YLIQ[0:IND1])
            # ice mixing ratio
            WI = sum(Y[IND1:IND2]*Y[0:IND1])

            Cpm = c.CP + WV*c.CPV + WL*c.CPW + WI*c.CPI
            
            # RH with respect to ice
            RH_ICE = WV/(c.eps*svp_ice/
                         (Y[IPRESS_ICE]-svp_ice)) 
    # ------------------------- growth rate of ice --------------------------
            RH_EQ = 1e0 # from ACPIM, FPARCELCOLD - MICROPHYSICS.f90
            
            CAP = f.CAPACITANCE01(Y[0:IND1],np.exp(Y[IND2:IND3])) 

            growth_rate = f.ICEGROWTHRATE(Y[ITEMP_ICE],Y[IPRESS_ICE],RH_ICE,RH_EQ,
                                          Y[0:IND1],np.exp(Y[IND2:IND3]),CAP)
            growth_rate[np.isnan(growth_rate)] = 0 # get rid of nans
            growth_rate = np.where(Y[IND1:IND2] < 1e-6, 0.0, growth_rate)
            
            # Mass of water condensing 
            dy_dt[:IND1] = growth_rate 

    #---------------------------aspect ratio---------------------------------------
            DELTA_RHO = c.eps*svp/(Y[IPRESS_ICE] - svp)
            DELTA_RHOI = c.eps*svp_ice/(Y[IPRESS_ICE] - svp_ice)
            DELTA_RHO = Y[IRH_ICE]*DELTA_RHO-DELTA_RHOI
            DELTA_RHO = DELTA_RHO*Y[IPRESS_ICE]/Y[ITEMP_ICE]/c.RA
            
            RHO_DEP = f.DEP_DENSITY(DELTA_RHO,Y[ITEMP_ICE])
            
            # this is the rate of change of LOG of the aspect ratio
            dy_dt[IND2:IND3] = (dy_dt[0:IND1]*(
                                           (f.INHERENTGROWTH(Y[ITEMP_ICE])-1)/
                                           (f.INHERENTGROWTH(Y[ITEMP_ICE])+2))/
                                           (Y[0:IND1]*c.rhoi*RHO_DEP))
    #------------------------------------------------------------------------------        
            # Change in vapour content
            dwv_dt = -1*sum(Y[IND1:IND2]*dy_dt[0:IND1])

            # change in water vapour mixing ratio
            DRI = -1*dwv_dt
            
            dy_dt[ITEMP_ICE] = 0.0#+c.LS/Cpm*DRI
            
           # if n.Simulation_type.lower() == 'parcel':
           #     dy_dt[ITEMP_ICE]=dy_dt[ITEMP_ICE] + c.LS/Cpm*DRI
    #---------------------------RH change------------------------------------------
            
            dy_dt[IRH_ICE] = (Y[IPRESS_ICE]-svp)*svp*dwv_dt
            
            dy_dt[IRH_ICE] = (dy_dt[IRH_ICE] - 
                              WV*Y[IPRESS_ICE]*derivative(f.svp_liq,Y[ITEMP_ICE],dx=1.0)*
                              dy_dt[ITEMP_ICE])
            dy_dt[IRH_ICE] = dy_dt[IRH_ICE]/(c.eps*svp**2)
    #------------------------------------------------------------------------------        
            return dy_dt
        
        #--------------------- SET-UP solver --------------------------------------
        
        y0 = Y
        t0 = 0.0
          
        #define assimulo problem
        exp_mod = Explicit_Problem(dy_dt_func,y0,t0)
        
        # define an explicit solver
        exp_sim = CVode(exp_mod)
        exp_sim.iter = 'Newton'
        exp_sim.discr = 'BDF'
        # set tolerance for each dydt function
        tol_list = np.zeros_like(Y)
        tol_list[0:IND1] = 1e-30 # mass
        tol_list[IND1:IND2] = 10 # number
        
        tol_list[IND2:IND3] = 1e-30 # aspect ratio
       # tol_list[IND3:IRH_SV_ICE] = 1e-26
        #tol_list[IRH_SV_ICE] = 1e-26
        
        tol_list[IPRESS_ICE] = 10
        tol_list[ITEMP_ICE] =1e-4
        tol_list[IRH_ICE] = 1e-8 
        
        exp_sim.atol = tol_list
        exp_sim.rtol = 1.0e-8
        exp_sim.inith = 1.0e-2 # initial time step-size
        exp_sim.usejac = False
        exp_sim.maxncf = 100 # max number of convergence failures allowed by solver
        exp_sim.verbosity = 40
       
        t_output,y_output = exp_sim.simulate(1)
        
        return y_output[-1,:]

    #    from variables import output, output_ice, dummy2, ACT_DROPS, setup_output

    model_status_file = open('status_file.txt','w')

    for idx in range(len(output)): #len(output) = 2800
        # don't run if there is an error
        if ERROR_FLAG:
             break
        # initialize output with starting values from namelist 
        if idx == 0: ###Note the followings are for t=0

            if n.SV_flag:
                for i, key in enumerate(Mass_frac.keys()):
                    for j in range(nmodes):
                        MBIN22[i, IND1 * j:IND1 * (j + 1)] = Mass_frac[key][j] * Y_AER[0, IND1 * j:IND1 * (1 + j)]
                        Y[INDSV1:INDSV2] = np.reshape(MBIN22[-1 * n.n_sv:, :], [nbins*nmodes*n.n_sv])
                #Y[INDSV1:INDSV2] = np.tile(Y_AER[0,:]*sum(n.SV_MF),n.n_sv) # this is correct (repeat aerosol mass at t=0 for 10 times).  ###links Y and Y_AER (aerosol mass in each bin), Y[: IND1] is water mass in each bin
            else:
                Y[INDSV1:INDSV2] = 0.0 #idx = 0

            Y[IRH_SV] = n.SV_MR#2.58e-10 #0.5e-9 # this is the mixing ratio of semi_vols, length number semi vols
            
            output[idx,:] = Y ### links output and Y (Note that idx=0 here)
            
            output_file = setup_output()
            ACT_DIAM = np.zeros([len(output),IND1])
            
        else:
            # solve warm cloud ODEs
            output[idx,:], test_time = run_sim(output[idx-1,:],(idx-1), Y_AER[idx-1,:], output_ice[idx-1,:])
            Y_AER[idx,:] = Y_AER[idx-1,:]
            # -------------------- calculate the number of activated drops ----------------------
            # 1. find mass for activation
            if n.SV_flag:
                ACT_MASS = f.find_act_mass(Y_AER[idx,:]+np.sum(np.reshape(output[idx,INDSV1:INDSV2],[n.n_sv,nbins*nmodes]),axis=0), output[idx,ITEMP], nbins, nmodes, rhobin, Kappa)
                ACT_DIAM[idx,:] = 2*((ACT_MASS*3)/(4*np.pi*c.rhow))**(1/3)
                
            else:
                ACT_MASS = f.find_act_mass(Y_AER[idx,:], output[idx,ITEMP], nbins, nmodes, rhobin, Kappa)
            # 2. find the number of aerosol with water mass greater than mass for activation 
            ACT_DROPS[idx,:] = np.where(output[idx,:IND1] > ACT_MASS, 
                                        output[idx,IND1:IND2], 0.0)
    # ------calculate the number of particles in CDP size bins --------------------
            CDP_CONC_liq[idx,:], CDP_CONC_ice[idx,:], CDP_CONC_total[idx,:] = f.calc_CDP_concentration(
                        output[idx,:IND2], Y_AER[idx,:], 
                        output_ice[idx,:IND2], YICE_AER[idx,:], output[idx,ITEMP],
                        IND1, IND2, rhobin)
              
    # ---------------------------------------------------------------------------------------      
            # calculate ice if below zero degrees
            if output[idx,ITEMP] < 273.15:
                # 1. find the number of particles that freeze
                icenuc = f.icenucleation(
                        output[idx,0:IND1],Y_AER[idx-1,:], output[idx,IND1:IND2], output[idx,ITEMP], 
                        output[idx,IPRESS],nbins, nmodes, rhobin, 
                        Kappa, ncomps, dt,YICE_AER[idx-1,:], 
                        output_ice[idx-1,:], IND1, IND2, ACT_MASS, output[idx,IRH], alpha_crit,
                        Heterogeneous_freezing_criteria, Mass_frac)
                
                output_ice[idx,:IND1]     = icenuc[0] # ice mass
                YICE_AER[idx,:]           = icenuc[1] # aerosol mass in ice 
                output_ice[idx,IND1:IND2] = icenuc[2] # ice number
                output[idx,IND1:IND2]     = icenuc[3] # liquid number
                Y_AER[idx,:] = Y_AER[idx-1,:]
                

                # initialise run_sim_ice with T,P,RH from run_sim (liquid)
                output_ice[idx,ITEMP_ICE]  = output[idx,ITEMP]
                output_ice[idx,IPRESS_ICE] = output[idx,IPRESS]
                output_ice[idx,IRH_ICE]    = output[idx,IRH]
                
                # check things dont go negative
                output_ice[idx,:] = np.where(output_ice[idx,:] < 0, 
                                              1e-22,
                                              output_ice[idx,:])
         
                dummy2[:IND2] = output[idx,:IND2] # liquid mass and drop/aerosol number
                dummy2[-3:] = output[idx,-3:] # pressure, temperature and RH
                
                # 2. grow ice by solving cold ODEs
                output_ice[idx,:] = run_sim_ice(output_ice[idx,:],dummy2)
                
                # check things dont go negative
                output_ice[idx,:] = np.where(output_ice[idx,:] < 0,
                                             0.0,
                                             output_ice[idx,:])
                
                # change RH and Temp due to ice formation and growth
                output[idx,IRH]   = output_ice[idx,IRH_ICE]
                output[idx,ITEMP] = output_ice[idx,ITEMP_ICE]
                
                # melting 
            if output[idx,ITEMP] > 273.15:
                output[idx,:IND1] = output_ice[idx,:IND1] + output[idx,:IND1]
                output_ice[idx,IND1:IND2] = 0.0
                Y_AER[idx,:] = Y_AER[idx,:] + YICE_AER[idx,:]
                YICE_AER[idx,:] = 0.0
            if np.sum(output_ice[idx,IND1:IND2],axis=0) > 1:
                CDP_CONC_liq[idx,:], CDP_CONC_ice[idx,:], CDP_CONC_total[idx,:] = f.calc_CDP_concentration(
                        output[idx,:IND2], Y_AER[idx,:], 
                        output_ice[idx,:IND2], YICE_AER[idx,:], output[idx,ITEMP],
                        IND1, IND2, rhobin)
            elif np.sum(output[idx,IND1:IND2],axis=0) > 1:
                CDP_CONC_liq[idx,:], dummy, CDP_CONC_total[idx,:] = f.calc_CDP_concentration(
                        output[idx,:IND2], Y_AER[idx,:], 
                        output_ice[idx,:IND2], YICE_AER[idx,:], output[idx,ITEMP],
                        IND1, IND2, rhobin)
            
            # write to netcdf output file
            output_file['ice_total'][idx] = sum(output_ice[idx,IND1:IND2])
            output_file['drop_total'][idx] = sum(ACT_DROPS[idx,:])
            output_file['liq_total'][idx] = sum(output[idx,IND1:IND2])
            output_file['temperature'][idx] = output[idx,ITEMP]
            output_file['pressure'][idx] = output[idx,IPRESS]
            output_file['RH'][idx] = output[idx,IRH]
            output_file['CDP_CONC_total'][idx] = CDP_CONC_total[idx,:]
            #output_file['altitude'][idx] = output[idx,w*idx]
            if n.SV_flag:
                output_file['RH_SV'][idx,:] = output[idx,IRH_SV]
            
            for mode in range(nmodes): # =0 for 1 mode
                start  = IND1+mode*nbins # = IND1 (70)
                end = start+nbins # IND1 + 70 (140)
                output_file['ice_number'][idx,mode,:] = output_ice[idx,start:end]   #70~140
                output_file['liq_number'][idx,mode,:] = output[idx,start:end]
                start2 = mode*nbins #0
                end2 = start2+nbins #0+70 
                output_file['ice_mass'][idx,mode,:] = output_ice[idx,start2:end2]   #0~70
                output_file['liq_mass'][idx,mode,:] = output[idx,start2:end2]
                output_file['activated_drops'][idx,mode,:] = ACT_DROPS[idx,start2:end2] 
            if n.SV_flag:
             #   for mode in range(nmodes):
              #      start3 = INDSV1+mode*nbins
                #    end3 = start3+nbins
                output_file['semivol_mass'][idx,:,:,:] = np.reshape(output[idx,INDSV1:INDSV2],
                                                                      [n.n_sv,nmodes,nbins]) # shouldnt this be start3 and end3? (23/09/19 - ES)
            
           # print(sum(output_ice[idx,IND1:IND2]))
            model_status_file.write(str(idx)+'\n')
            print('time is = ',idx)
          
    print("--- %s seconds ---" % (time.time() - start_time))
    model_status_file.close()
    try:
        output_file.close()
    except:
        pass
    # plotting output
  #  LWC = np.sum(output[:,:IND1]*output[:,IND1:IND2],axis=1) 
  #  plt.plot(LWC*1e3,label='py')
  #  plt.plot(FSSP.Times, FSSP.LWC_exp)
  #  plt.ylim([0,2]),plt.legend()
  #  plt.xlim([0,600])
  #  plt.show()

  #  plt.plot(np.sum(ACT_DROPS/1e6,axis=1))
  #  plt.plot(FSSP.Times, FSSP.Conc_exp)
  #  plt.show()

#    if not ERROR_FLAG:            
#         fig = plt.figure()
#            
#         ax1 = fig.add_subplot(221)
#         ax1.plot(np.sum(ACT_DROPS, axis=1)/1e6,label='py', c='r')
#         ax1.plot(nc['TIMES'],nc['NDROP'][:,0,0,0]/1e6, label = 'ACPIM')
#         ax1.set_title('Activated Drops')
#         ax1.set_ylabel('# cm$^{-3}$')
#        # ax1.legend()
#         # 
#         ax2 = fig.add_subplot(222)
#         ax2.plot(np.sum(output_ice[:,IND1:IND2], axis=1)/1e6)
#       #  ax2.plot(nc['TIMES'],nc['NICE'][:,0,0,0]/1e6)
#         # ax2.plot(obs)
#         ax2.set_title('Ice Number Concentration')
#         ax1.set_ylabel('# cm$^{-3}$')
#         # 
#         ax3 = fig.add_subplot(223)
#         ax3.plot(output[:,IRH])
#         ax3.plot(nc['TIMES'],nc['RH'][:,0,0,0])
#         ax3.set_title('RH')
#         ax3.set_xlabel('Time (s)')
#         # 
#         ax4 = fig.add_subplot(224)
#         ax4.plot(output[:,ITEMP]-273.15)
#         ax4.plot(nc['TIMES'],nc['TEMP'][:,0,0,0]-273.15)
#         ax4.set_title('Temperature')
#         ax4.set_xlabel('Time (s)')
#         # 
#         plt.tight_layout()
#         # change output figure name to input from GUI
#         #plt.savefig(g.output_fig_filename)
#         plt.show()
#           
#        # =============================================================================
#
#         CDP_bins = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
#               16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0,
#               38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0]


#if __name__ == "__main__":
    #run(np.array([N_aer1, N_aer2]), np.array([D_aer1, D_aer2]), np.array([sig_aer1,sig_aer2]), T, w, RH, runtime)
#    output, output_ice,ACT_DROPS = run(np.array([1.237e9, 6.16e8]), np.array([3.38e-7, 4.35e-7]), np.array([0.587, 0.869]), 248.84, 3.55, 1.0167,300)

#plt.pcolor(range(len(output)),CDP_bins,v.CDP_CONC_liq.T/1e6, vmin=0.1,vmax=100, norm=matplotlib.colors.LogNorm())
#plt.colorbar()
  
# =============================================================================
############################### NOTES #########################################
# 29/09/2017 ------------------------------------------------------------------
# BUG - model not running for T -8 -> 1C and RH = 0.9 in NAMELIST
# does not get past first time step
# higher RH does run
# setting growthrate to 1e-11 makes model run
# calculation of growthrate looks fine as does dy_dt[RH]
# Solver passing in NANs for temperature at some points?
# is it because growthrate goes negative?
# 25/10/2017 ------------------------------------------------------------------
# FIX - calculation of aspect ratio (instead of just setting to 1)
# model now runs for T -7 RH = 0.9 
# BUG - losing water mass somewhere when there is ice  
# BUG - Mass required for activation too high - 10/11/2017 ------------------
# 24/11/2017 ------------------------------------------------------------------
# FIX - Mass required for activation corrected, things activate, similar to
#       ACPIM results. Changed RHOAT in KK02 to RHOW and bracketed root in brent
# BUG - Is RHOAT calculated correctly? It shouldnt be too different than RHOW
# 27/11/2017 ------------------------------------------------------------------
# BUG - Freezing around an order of magnitude more ice than ACPIM?
# 14/02/2018 ------------------------------------------------------------------
# FIX - Ice number now agrees with ACPIM, changes to f.icenucleation and 
#        f.activesites made.
# BUG - won't run for small diameters and low kappa values eg 70nm and 0.0061
# FIX - changed tolerence on mass from 1e-25 to 1e-30
# 06/04/2018 ------------------------------------------------------------------
# BUG - too much ice and too high RH at low temperatures (< -25C)
# FIX - moved calculation of capacitance to inside dy_dt_func
# 12/04/18 --------------------------------------------------------------------
# Added internally mixed aerosol capabilities.
# Added new module variables where variables are set up

# to speed up get rid of ifs and try to import less variables
    # do a test surface ten function without if and past in varibles form constants
    # instead of importing from module. 
    # try np.multiply for multipling matrixs instead of for i in mode: for j in bin
# try numba in last instance for speed up if thats important 
# try golden optimization method in scipy
#     