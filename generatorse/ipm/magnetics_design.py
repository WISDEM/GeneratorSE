# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:45:08 2022

@author: lsethura
"""

"""
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071
"""

import numpy as np
import openmdao.api as om
from ipm.femm_fea import FEMM_Geometry
from ipm.structural import PMSG_Outer_Rotor_Structural


class PMSG_active(om.ExplicitComponent):

    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('D_a',0.0, units ='m', desc='Stator diameter ')
        self.add_input('g',0.0,units='m',desc='Air gap length')
        self.add_input('l_s',0.0, units ='m', desc='Stator core length ')
        self.add_output('l_eff_stator',0.0, units ='m', desc='Effective Stator core length ')
        self.add_input('b_s',0.0, units ='m', desc='Stator core length ')
        self.add_input('b_t',0.0, units ='m', desc='Stator core length ')
        self.add_input('h_s',0.0, units ='m', desc='Stator core length ')
        self.add_input('h_t',0.0, units ='m', desc='tooth height ')
        self.add_input('N_nom', 0.0, units = 'rpm', desc='rated speed')
        self.add_input('h_m',0.0, units ='m',desc='magnet height')
        self.add_input('h_ys',0.0, units ='m', desc='Yoke height')
        self.add_input('h_yr',0.0, units ='m', desc='rotor yoke height')
        self.add_input('J_s',0.0, units ='A/(mm*mm)', desc='Stator winding current density')
        self.add_input('N_c',0.0, desc='Number of turns per coil')
        self.add_output('N_s',0.0, desc='Number of turns per coil')
        self.add_input('b',0.0, desc='Slot pole combination')
        self.add_input('c',0.0, desc='Slot pole combination')
        self.add_input('p',0.0, desc='Pole pairs ')
        self.add_output('p1',0.0, desc='Pole pairs ')
        self.add_output("r_g", 0.0, units="m", desc="air gap radius ")
        self.add_output("k_w", 0.0, desc="winding factor ")

        
        # Material properties
        self.add_input('rho_Copper',0.0,units='kg/(m**3)', desc='Copper density')
        self.add_input('resistivity_Cu',0.0,units ='ohm*m', desc='Copper resistivity ')
        
        # PMSG_structrual inputs

        self.add_input('h_sr',0.0,units='m',desc='Structural Mass')
        
        # Magnetic loading
        
        self.add_input("M_Fes", 0.0, units="kg",desc="mstator iron mass ")
        self.add_input("M_Fer", 0.0, units="kg",desc="rotor iron mass ")
        
        self.add_input('B_symax',0.0,desc='Peak Stator Yoke flux density B_ymax')
        self.add_input('B_rymax',0.0, desc='Peak Rotor yoke flux density')
        self.add_input('B_g', 0.0, desc='Peak air gap flux density ')
        self.add_input('tau_p',0.0, units ='m',desc='Pole pitch')
        self.add_input('tau_s',0.0, units ='m',desc='Pole pitch')
        
        
        #Stator design
        self.add_output('A_Cuscalc',0.0, units ='mm**2', desc='Conductor cross-section')

       
        
        # Electrical performance

        self.add_output('f',0.0, units ='Hz', desc='Generator output frequency')
        self.add_input('I_s', 0.0, units ='A', desc='Generator input phase current')
        self.add_output('R_s',0.0, units ='ohm',desc='Stator resistance')
        self.add_output('J_actual',0.0, units ='A/m**2', desc='Current density')
        
        self.add_discrete_input('m',3, desc=' no of phases')
        self.add_discrete_input('k_sfil',0.65,desc='slot fill factor')

        self.add_discrete_input('mu_0',4*np.pi/1e7,desc='premeability of free space')	
        
        # Objective functions
        self.add_output('K_rad', desc='Aspect ratio')
        self.add_output('Test', 0.0, units='m',desc='GCD ')
        self.add_output('P_Cu',0, units='W',desc='Copper losses')


       
        # Other parameters
        self.add_output('R_out',0.0,units='m', desc='Outer radius')
        self.add_output('S',0.0,desc='Stator slots')

        
        # Mass Outputs
        self.add_output('Copper',0.0, units ='kg', desc='Copper Mass')

      
        self.add_output('I',np.zeros(3),units='kg*m**2',desc='Moment of inertia')
        
        
        self.declare_partials('*','*',method='fd')
        
        
        
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        
        ###################################################### Electromagnetic design#############################################
        
       
        
        
        outputs["r_g"]=inputs['g']+inputs["D_a"]*0.5                    #magnet radius
        
        outputs['K_rad']=inputs['l_s'][0]/(2* outputs['r_g'][0])							# Aspect ratio
        
        b_so			            =  2*inputs['g']                              # Slot opening

        
        outputs['p1']               =  np.ceil(inputs['p'][0]/ 5.) * 5
        
     # pole pitch
        
        #Calculating winding factor
        
        Slot_pole                   = inputs['b']/inputs['c']
        
        outputs['S']                =  Slot_pole*2*outputs['p1'] *discrete_inputs["m"]
        
        outputs['k_w']	  =0.933
    
  
        om_e = 2 * np.pi * inputs['N_nom'] / 60*outputs['p1']
       
        # angular frequency in radians
        outputs['f']                = 2*outputs['p1']*inputs["N_nom"]/120					# outout frequency
        outputs['N_s']          =   outputs['S']*2.0/3*inputs["N_c"]								# Stator turns per phase"]

        l_Cus			        = (2*(inputs['l_s']+np.pi/4*(inputs['tau_s']+inputs['b_t'])))  # length of a turn
        
        
        outputs['A_Cuscalc']	= inputs["I_s"]/inputs["J_s"]
        
        A_slot                  = 2*inputs["N_c"]*outputs['A_Cuscalc']*(10**-6)/discrete_inputs["k_sfil"]
        
 
        
        
        
        
        N_coil                      =   2*outputs['S'][0]
        
  
           
        # Calculating stator resistance
        
      
        
        L_Cus                       = outputs['N_s']*l_Cus
        
        outputs['R_s']              =inputs['resistivity_Cu']* (1 + 20 * 0.00393)*(outputs['N_s'])*l_Cus*inputs["J_s"]*1e6/(inputs["I_s"])
        
      
        # Calculating Electromagnetically active mass
        
                  
        V_Cus 	                    =   discrete_inputs["m"]*L_Cus*(outputs['A_Cuscalc']*(10**-6))     # copper volume
        

                    
        outputs['Copper']		    =   V_Cus*inputs['rho_Copper']
        
        h_ew                         =0.25

       
        
        
        
        # Calculating Losses
        ##1. Copper Losses
        
        K_R                         =   1.0   # Skin effect correction co-efficient
        
        outputs['P_Cu']             =   discrete_inputs["m"]*(inputs['I_s']/2**0.5)**2*outputs['R_s']*K_R
            


class Results(om.ExplicitComponent):
    def setup(self):
        self.add_input("B_g", 0.0, desc="Peak air gap flux density ")
        self.add_input("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("T_rated", 0.0, units="N*m", desc="Rated torque ")
        self.add_input("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_input("P_rated", units="W", desc="Machine rating")
        self.add_input("p1", 0.0, desc="pole pairs")
        self.add_input("P_Cu", units="W", desc="Copper losses")
        self.add_output("P_Ftm", units="W", desc="magnet losses")
        self.add_output("P_Fe", units="W", desc="Iron losses")
        self.add_input("l_s", 0.0, units="m", desc="Stator core length")
        self.add_input("D_a", 0.0, units="m", desc="Armature diameter")
        self.add_input("E_p_target", 0.0, units="V", desc="target terminal voltage")
        self.add_input("h_m", 0.0, units="m", desc="magnet height")
        self.add_input("l_m", 0.0, units="m",desc="magnet length ")
        self.add_input("B_rymax", 0.0, desc="Peak Rotor yoke flux density")
        self.add_input("B_smax", 0.0, desc="Peak flux density in the field coils")
        self.add_input("Iron", 0.0, units="kg",desc="magnet length ")
        self.add_input("M_Fes", 0.0, units="kg",desc="mstator iron mass ")
        self.add_input("M_Fer", 0.0, units="kg",desc="rotor iron mass ")
        self.add_input("Copper", 0.0, units="kg", desc="Copper mass")
        self.add_output("mass_PM", 0.0, units="kg", desc="Iron mass")
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_PM',0.0,units ='kg/(m**3)', desc='Magnet density ')
        
        self.add_discrete_input('P_Fe0h',4.0, desc='specific hysteresis losses W/kg @ 1.5 T')
        self.add_discrete_input('P_Fe0e',1.0,desc='specific eddy losses W/kg @ 1.5 T')
        self.add_input('Structural_mass',0.0,units='kg',desc='Structural Mass')
        self.add_output('Mass',0.0,units='kg',desc='Structural Mass')

        self.add_output("E_p", 0.0, units="V", desc="terminal voltage")

        self.add_output("P_Losses", units="W", desc="Total power losses")
        self.add_output("gen_eff", desc="Generator efficiency")
        self.add_output("torque_ratio", desc="Whether torque meets requirement")
        self.add_output("E_p_ratio", desc="Whether terminal voltage meets requirement")
        self.add_input("k_w", 0.0, desc="winding factor ")
        self.add_output('P_ad',0, units='W',desc='Additional losses')

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs,discrete_inputs,discrete_outputs):
        # Unpack inputs
        D_a = float(inputs["D_a"])
        k_w = float(inputs["k_w"])
        B_g = float(inputs["B_g"])
        N_s = float(inputs["N_s"])
        N_nom = float(inputs["N_nom"])
        P_rated = float(inputs["P_rated"])
        P_Cu = float(inputs["P_Cu"])
        l_s = float(inputs["l_s"])
        l_m = float(inputs["l_m"])
        h_m = float(inputs["h_m"])
        T_rated = float(inputs["T_rated"])
        T_e = float(inputs["T_e"])
        E_p_target = float(inputs["E_p_target"])
        p1 = float(inputs["p1"])

        # Calculating  voltage per phase
        om_m = 2 * np.pi * N_nom / 60
        
        om_e = 2*p1 * N_nom/120
        outputs["E_p"] = E_p = l_s * (D_a * 0.5 * k_w * B_g * om_m * N_s)*np.sqrt(3/2)
        
        pFtm                        =300 # specific magnet loss
            
        outputs['P_Ftm']                       =P_Ftm=pFtm*2*p1 *l_m*l_s
        

        outputs["torque_ratio"] = T_e / T_rated
        outputs["E_p_ratio"] = E_p / E_p_target
       
        outputs['mass_PM']          =   4*p1*l_m*h_m*l_s*inputs['rho_PM']
        
       
         # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys	                    =   inputs["M_Fes"]*(inputs['B_smax']/1.5)**2*(discrete_inputs["P_Fe0h"]*om_e/(60)) # Hysteresis losses in stator
            
        P_Ftys	                    =   inputs["M_Fes"]*(inputs['B_smax']/1.5)**2*(discrete_inputs["P_Fe0e"]*(om_e/(60))**2) # Eddy losses in stator
            
        P_Fesnom                   =   P_Hyys+P_Ftys
            
        P_Hyyr	                    =   inputs["M_Fer"]*(inputs['B_rymax']/1.5)**2*(discrete_inputs["P_Fe0h"]*om_e/(60)) # Hysteresis losses in rotor
            
        P_Ftyr	                    =   inputs["M_Fer"]*(inputs['B_rymax']/1.5)**2*(discrete_inputs["P_Fe0e"]*(om_e/(60))**2) # Eddy losses in rotor
            
        P_Fernom                   =   P_Hyyr+P_Ftyr
        
        outputs['P_Fe']             = P_Fe=P_Fesnom+P_Fernom
        
        outputs['P_ad']       =      P_ad           =0.2*(outputs['P_Fe'])

        outputs["P_Losses"] = P_Losses = P_Cu + P_Fe + P_ad+P_Ftm    
        
      
        outputs["gen_eff"] = (1 - P_Losses / P_rated)
            # additional stray losses due to leakage flux
            

        
        outputs['Mass']             = inputs["Iron"]+inputs['Structural_mass']+outputs['mass_PM']+inputs['Copper'] 

