"""DFIG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.api import Group, Problem, Component,ExecComp,IndepVarComp,ScipyOptimizer,pyOptSparseDriver
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.drivers import *
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
import pandas as pd




class DFIG(Component):
	
	""" Estimates overall mass, dimensions and Efficiency of DFIG generator. """
	
	def __init__(self):
		
		super(DFIG, self).__init__()
		
		# DFIG design inputs
		self.add_param('r_s', val=0.0, units ='m', desc='airgap radius r_s')
		self.add_param('l_s', val=0.0, units ='m', desc='Stator core length l_s')
		self.add_param('h_s', val=0.0, units ='m', desc='Stator slot height')
		self.add_param('h_r',val=0.0, units ='m', desc='Rotor slot height')
		
		self.add_param('machine_rating',val=0.0, units ='W', desc='Machine rating')
		self.add_param('n_nom',val=0.0, units ='rpm', desc='rated speed')
		
		self.add_param('B_symax',val=0.0, desc='Peak Stator Yoke flux density B_symax')
		self.add_param('S_Nmax',val=0.0, desc='Max rated Slip ')
		self.add_param('I_0',val=0.0, desc='Rotor current at no-load')
		
		self.add_param('highSpeedSide_cm',val=np.array([0.0, 0.0, 0.0]),desc=' high speed sidde COM [x, y, z]')
		self.add_param('highSpeedSide_length',val=0.0, desc='high speed side length')
		self.add_param('Gearbox_efficiency',val=0.0, desc='Gearbox efficiency')
		
		# Material properties
		self.add_param('rho_Fe',val=0.0,units='kg*m**-3', desc='Magnetic Steel density ')
		self.add_param('rho_Copper',val=0.0,units='kg*m**-3', desc='Copper density ')
		
		# DFIG generator design output
		# Magnetic loading
		self.add_output('B_g',val=0.0, desc='Peak air gap flux density B_g')
		self.add_output('B_g1',val=0.0,  desc='air gap flux density fundamental ')
		self.add_output('B_rymax',val=0.0,  desc='maximum flux density in rotor yoke')
		self.add_output('B_tsmax',val=0.0,  desc='maximum tooth flux density in stator')
		self.add_output('B_trmax',val=0.0, desc='maximum tooth flux density in rotor')
		
		#Stator design
		self.add_output('q1',val=0.0, desc='Slots per pole per phase')
		self.add_output('N_s',val=0.0,desc='Stator turns')
		self.add_output('S'		,val=0.0, desc='Stator slots')
		self.add_output('h_ys',val=0.0, desc='Stator Yoke height')
		self.add_output('b_s',val=0.0,desc='stator slot width')
		self.add_output('b_t',val=0.0, desc='stator tooth width')
		self.add_output('D_ratio',val=0.0,desc='Stator diameter ratio')
		self.add_output('A_Cuscalc',val=0.0, desc='Stator Conductor cross-section mm^2')
		self.add_output('Slot_aspect_ratio1',val=0.0, desc='Stator slot apsect ratio')
		
		#Rotor design
		self.add_output('h_yr',val=0.0, desc=' rotor yoke height')
		self.add_output('tau_p',val=0.0, desc='Pole pitch')
		self.add_output('p',val=0.0, desc='No of pole pairs')
		self.add_output('Q_r',val=0.0, desc='Rotor slots')
		self.add_output('N_r',val=0.0,desc='Rotor turns')
		self.add_output('b_r',val=0.0, desc='rotor slot width')
		self.add_output('b_trmin',val=0.0,desc='minimum tooth width')
		self.add_output('b_tr',val=0.0, desc='rotor tooth width')
		self.add_output('A_Curcalc',val=0.0, desc='Rotor Conductor cross-section mm^2')
		self.add_output('Slot_aspect_ratio2',val=0.0, desc='Rotor slot apsect ratio')
		
		# Electrical performance
		self.add_output('E_p',val=0.0, desc='Stator phase voltage')
		self.add_output('f',val=0.0, desc='Generator output frequency')
		self.add_output('I_s',val=0.0, desc='Generator output phase current')
		self.add_output('A_1' ,val=0.0, desc='Specific current loading')
		self.add_output('J_s',val=0.0, desc='Stator winding Current density')
		self.add_output('J_r',val=0.0, desc='Rotor winding Current density')
		self.add_output('R_s',val=0.0, desc='Stator resistance')
		self.add_output('R_R',val=0.0, desc='Rotor resistance')
		self.add_output('L_r',val=0.0, desc='Rotor inductance')
		self.add_output('L_s',val=0.0, desc='Stator synchronising inductance')
		self.add_output('L_sm',val=0.0, desc='mutual inductance')
		
		# Objective functions
		self.add_output('Mass',val=0.0, desc='Actual mass')
		self.add_output('K_rad',val=0.0, desc='Stack length ratio')
		self.add_output('Losses',val=0.0, desc='Total loss')
		self.add_output('gen_eff',val=0.0, desc='Generator efficiency')
		
		# Mass Outputs
		self.add_output('Copper',val=0.0, desc='Copper Mass')
		self.add_output('Iron',val=0.0, desc='Electrical Steel Mass')
		self.add_output('Structural_mass'	,val=0.0, desc='Structural Mass')
		
		
		# Structural performance
		self.add_output('TC1',val=0.0, desc='Torque constraint -stator')
		self.add_output('TC2',val=0.0, desc='Torque constraint-rotor')
		
		# Other parameters
		self.add_output('Current_ratio',val=0.0,desc='Rotor current ratio')
		self.add_output('Overall_eff',val=0.0, desc='Overall drivetrain efficiency')
		self.add_output('I',val=np.array([0.0, 0.0, 0.0]),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
		self.add_output('cm', val=np.array([0.0, 0.0, 0.0]),desc='COM [x,y,z]')
		
		self.gen_sizing = generator_sizing()
		
	def solve_nonlinear(self, inputs, outputs, resid):
		(outputs['B_g'],outputs['B_g1'], outputs['B_rymax'], outputs['B_tsmax'],outputs['B_trmax'],\
		outputs['q1'],outputs['N_s'],outputs['S'],outputs['h_ys'],outputs['b_s'],outputs['b_t'],\
		outputs['D_ratio'],outputs['A_Cuscalc'],outputs['Slot_aspect_ratio1'],outputs['h_yr'],\
		outputs['tau_p'],outputs['p'], outputs['Q_r'],outputs['N_r'], outputs['b_r'],outputs['b_trmin'],\
		outputs['b_tr'],outputs['A_Curcalc'],outputs['Slot_aspect_ratio2'],outputs['E_p'], outputs['f'], \
		outputs['I_s'], outputs['A_1'],outputs['J_s'],outputs['J_r'],outputs['R_s'], outputs['R_R'],\
		outputs['L_r'],outputs['L_s'],outputs['L_sm'],outputs['Mass'],outputs['K_rad'],outputs['Losses'],\
		outputs['gen_eff'],outputs['Copper'],outputs['Iron'],outputs['Structural_mass'], outputs['TC1'],outputs['TC2'],\
		outputs['Current_ratio'],outputs['Overall_eff'],outputs['cm'],outputs['I'])\
		= self.gen_sizing.compute(inputs['r_s'], inputs['l_s'], inputs['h_s'], inputs['h_r'],inputs['Gearbox_efficiency'],inputs['machine_rating'],
		inputs['n_nom'], inputs['B_symax'], inputs['S_Nmax'],inputs['I_0'],inputs['rho_Fe'], inputs['rho_Copper'], \
		inputs['highSpeedSide_cm'],inputs['highSpeedSide_length'])

		return outputs


class generator_sizing(object):
	
	def __init__(self):
		
		pass
		
	def	compute(self,r_s, l_s,h_s,h_r,Gearbox_efficiency,machine_rating,n_nom,B_symax,S_Nmax,I_0, \
	rho_Fe,rho_Copper,highSpeedSide_cm,highSpeedSide_length):
		
		#Create internal variables based on inputs
		self.r_s = r_s
		self.l_s = l_s
		self.h_s = h_s
		self.h_r = h_r
		self.I_0 = I_0
		
		self.machine_rating=machine_rating
		self.n_nom=n_nom
		self.Gearbox_efficiency=Gearbox_efficiency
		
		self.rho_Fe= rho_Fe
		self.rho_Copper=rho_Copper
		
		self.B_symax=B_symax
		self.S_Nmax=S_Nmax
		
		self.highSpeedSide_cm=highSpeedSide_cm
		self.highSpeedSide_length=highSpeedSide_length
		
		#Assign values to universal constants
		g1          =9.81                  # m/s^2 acceleration due to gravity
		sigma       =21.5e3                # shear stress
		mu_0        =pi*4e-7               # permeability of free space
		cofi        =0.9                   # power factor
		h_w         = 0.005                # wedge height
		m           =3                     # Number of phases
		rho_Cu       =1.8*10**(-8)*1.4     # copper resisitivity
		h_sy0  =0
		
		#Assign values to design constants
		b_so=0.004												 # Stator slot opening
		b_ro=0.004												 # rotor slot openinng
		self.q1=5                          # Stator slots per pole per phase
		q2=self.q1-1											 # Rotor slots per pole per phase
		k_sfil =0.65								 			 # Stator Slot fill factor
		P_Fe0h =4			               			 # specific hysteresis losses W/kg @ 1.5 T 
		P_Fe0e =1			                     # specific hysteresis losses W/kg @ 1.5 T 
		b_s_tau_s=0.45										 # Stator slot width /slot pitch ratio
		b_r_tau_r=0.45										 # Rotor slot width /slot pitch ratio
		y_tau_p=12./15										 # Stator coil span to pole pitch	
		y_tau_pr=10./12										 # Rotor coil span to pole pitch
		
		self.p=3													 # pole pairs
		freq=60														 # grid frequency in Hz
		k_fillr = 0.55 										# Rotor Slot fill factor
		
		K_rs     =1/(-1*self.S_Nmax)			 # Winding turns ratio between rotor and Staor		
		I_SN   = self.machine_rating/(sqrt(3)*3000) # Rated current
		I_SN_r =I_SN/K_rs									 # Stator rated current reduced to rotor
		
		# Calculating winding factor for stator and rotor	
		
		k_y1=sin(pi*0.5*y_tau_p)										# winding Chording factor
		k_q1=sin(pi/6)/(self.q1*sin(pi/(6*self.q1))) # winding zone factor
		k_y2=sin(pi*0.5*y_tau_pr)										# winding Chording factor
		k_q2=sin(pi/6)/(q2*sin(pi/(6*q2)))					# winding zone factor
		k_wd1=k_y1*k_q1															# Stator winding factor
		k_wd2=k_q2*k_y2															# Rotor winding factor
		
		dia=2*self.r_s             												 # air gap diameter
		g=(0.1+0.012*(self.machine_rating)**(1./3))*0.001  #air gap length in m
		self.K_rad=self.l_s/dia														 # Aspect ratio
		r_r=self.r_s-g             												 #rotor radius
		self.tau_p=(pi*dia/(2*self.p))										 #pole pitch
		
		self.S=2*self.p*self.q1*m													# Stator Slots
		N_slots_pp=self.S/(m*self.p*2)										# Number of stator slots per pole per phase
		n  = self.S/2*self.p/self.q1        							#no of slots per pole per phase
		tau_s=self.tau_p/(m*self.q1)        							#slot pitch
		self.b_s=b_s_tau_s*tau_s;            							#Stator slot width
		self.b_t=tau_s-self.b_s              							#Stator tooth width
		
		self.Q_r=2*self.p*m*q2														# Rotor Slots
		tau_r=pi*(dia-2*g)/self.Q_r												# Rotor Slot pitch
		self.b_r=b_r_tau_r*tau_r													# Rotot Slot width
		self.b_tr=tau_r-self.b_r													# Rotor tooth width
		
		# Calculating equivalent slot openings
		
		mu_rs				=0.005
		mu_rr				=0.005
		W_s					=(self.b_s/mu_rs)*1e-3  #in m
		W_r					=(self.b_r/mu_rr)*1e-3  #in m
		
		self.Slot_aspect_ratio1=self.h_s/self.b_s
		self.Slot_aspect_ratio2=self.h_r/self.b_r
		
		# Calculating Carter factor for stator,rotor and effective air gap length
		
		gamma_s	= (2*W_s/g)**2/(5+2*W_s/g)
		K_Cs=(tau_s)/(tau_s-g*gamma_s*0.5)  #page 3-13
		gamma_r		= (2*W_r/g)**2/(5+2*W_r/g)
		K_Cr=(tau_r)/(tau_r-g*gamma_r*0.5)  #page 3-13
		K_C=K_Cs*K_Cr
		g_eff=K_C*g
		
		om_m=2*pi*self.n_nom/60															# mechanical frequency
		om_e=self.p*om_m																		# Electrical frequency
		K_s=0.3   																					# Saturation factor for Iron
		n_c1=2    																					#number of conductors per coil
		a1 =2    																						# number of parallel paths
		self.N_s=round(2*self.p*N_slots_pp*n_c1/a1)					# Stator winding turns per phase
		
		self.N_r=round(self.N_s*k_wd1*K_rs/k_wd2)						# Rotor winding turns per phase
		n_c2=self.N_r/(self.Q_r/m)													# rotor turns per coil
		
		# Calculating peak flux densities and back iron thickness
		
		self.B_g1=mu_0*3*self.N_r*self.I_0*2**0.5*k_y2*k_q2/(pi*self.p*g_eff*(1+K_s))
		self.B_g=self.B_g1*K_C
		self.h_ys = self.B_g*self.tau_p/(self.B_symax*pi)
		self.B_rymax = self.B_symax
		self.h_yr=self.h_ys
		self.B_tsmax=self.B_g*tau_s/(self.b_t) 
		
		d_se=dia+2*(self.h_ys+self.h_s+h_w)  								# stator outer diameter
		self.D_ratio=d_se/dia																# Diameter ratio
		self.f = self.n_nom*self.p/60
		
		# Stator slot fill factor
		if (2*self.r_s>2):
			K_fills=0.65
		else:
			K_fills=0.4
			
		# Stator winding calculation
		
		# End connection length  for stator winding coils
		
		l_fs=2*(0.015+y_tau_p*self.tau_p/2/cos(40))+pi*(self.h_s)
		l_Cus = 2*self.N_s*(l_fs+self.l_s)/a1             # Length of Stator winding 
		
		# Conductor cross-section
		A_s = self.b_s*(self.h_s-h_w)
		A_scalc=self.b_s*1000*(self.h_s*1000-h_w*1000)
		A_Cus = A_s*self.q1*self.p*K_fills/self.N_s
		self.A_Cuscalc = A_scalc*self.q1*self.p*K_fills/self.N_s
		
		# Stator winding resistance
		
		self.R_s=l_Cus*rho_Cu/A_Cus
		tau_r_min=pi*(dia-2*(g+self.h_r))/self.Q_r
		
		# Peak magnetic loading on the rotor tooth
		
		self.b_trmin=tau_r_min-b_r_tau_r*tau_r_min
		self.B_trmax = self.B_g*tau_r/self.b_trmin
		
		# Calculating leakage inductance in  stator
		
		K_01=1-0.033*(W_s**2/g/tau_s)
		sigma_ds=0.0042
		K_02=1-0.033*(W_r**2/g/tau_r)
		sigma_dr=0.0062
		
		L_ssigmas=(2*mu_0*self.l_s*n_c1**2*self.S/m/a1**2)*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
		L_ssigmaew=(2*mu_0*self.l_s*n_c1**2*self.S/m/a1**2)*0.34*self.q1*(l_fs-0.64*self.tau_p*y_tau_p)/self.l_s #end winding leakage inductance
		L_ssigmag=(2*mu_0*self.l_s*n_c1**2*self.S/m/a1**2)*(0.9*tau_s*self.q1*k_wd1*K_01*sigma_ds/g_eff) # tooth tip leakage inductance
		L_ssigma=(L_ssigmas+L_ssigmaew+L_ssigmag)  																											# stator leakage inductance
		self.L_sm =6*mu_0*self.l_s*self.tau_p*(k_wd1*self.N_s)**2/(pi**2*(self.p)*g_eff*(1+K_s))
		self.L_s=(L_ssigmas+L_ssigmaew+L_ssigmag)  																											# stator  inductance
		
		# Calculating leakage inductance in  rotor
		
		l_fr=(0.015+y_tau_pr*tau_r/2/cos(40*pi/180))+pi*(self.h_r)  # Rotor end connection length
		L_rsl=(mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*((self.h_r-h_w)/(3*self.b_r)+h_w/b_ro)  #slot leakage inductance
		L_rel= (mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*0.34*q2*(l_fr-0.64*tau_r*y_tau_pr)/self.l_s #end winding leakage inductance                  #end winding leakage inductance
		L_rtl=(mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*(0.9*tau_s*q2*k_wd2*K_02*sigma_dr/g_eff) # tooth tip leakage inductance
		self.L_r=(L_rsl+L_rtl+L_rel)/K_rs**2  # rotor leakage inductance
		sigma1=1-(self.L_sm**2/self.L_s/self.L_r)
		
		#Rotor Field winding
		
		# conductor cross-section
		diff=self.h_r-h_w
		A_Cur=k_fillr*self.p*q2*self.b_r*diff/self.N_r
		self.A_Curcalc=A_Cur*1e6
		
		L_cur=2*self.N_r*(l_fr+self.l_s)							# rotor winding length
		R_r=rho_Cu*L_cur/A_Cur												# Rotor resistance
		
		# Equivalent rotor resistance reduced to stator
		self.R_R=R_r/(K_rs**2)
		om_s=(self.n_nom)*2*pi/60        							# synchronous speed in rad/s
		P_e=self.machine_rating/(1-self.S_Nmax)    		#Air gap power
		
		# Calculating No-load voltage
		self.E_p=om_s*self.N_s*k_wd1*self.r_s*self.l_s*self.B_g1*sqrt(2)
		I_r=P_e/m/self.E_p         										# rotor active current
		
		I_sm=self.E_p/(2*pi*freq*(self.L_s+self.L_sm)) # stator reactive current
		self.I_s=sqrt((I_r**2+I_sm**2))								#Stator current
		I_srated=self.machine_rating/3/K_rs/self.E_p	#Rated current
		
		# Calculating winding current densities and specific current loading
		
		self.J_s=self.I_s/(self.A_Cuscalc)
		self.J_r=I_r/(self.A_Curcalc)
		self.A_1=2*m*self.N_s*self.I_s/pi/(2*self.r_s)
		self.Current_ratio=self.I_0/I_srated           # Ratio of magnetization current to rated current
		
		# Calculating masses of the electromagnetically active materials
		
		V_Cuss=m*l_Cus*A_Cus
		V_Cusr=m*L_cur*A_Cur
		V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-(2*m*self.q1*self.p*self.b_s*self.h_s*self.l_s))
		V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)
		V_Fert=pi*self.l_s*(r_r**2-(r_r-self.h_r)**2)-2*m*q2*self.p*self.b_r*self.h_r*self.l_s
		V_Fery=self.l_s*pi*((r_r-self.h_r)**2-(r_r-self.h_r-self.h_yr)**2)
		self.Copper=(V_Cuss+V_Cusr)*self.rho_Copper
		M_Fest=V_Fest*self.rho_Fe
		M_Fesy=V_Fesy*self.rho_Fe
		M_Fert=V_Fert*self.rho_Fe
		M_Fery=V_Fery*self.rho_Fe
		self.Iron=M_Fest+M_Fesy+M_Fert+M_Fery
		M_gen=(self.Copper)+(self.Iron)
		
		#K_gen=self.Cu*self.C_Cu+(self.Iron)*self.C_Fe #%M_pm*K_pm;
		
		L_tot=self.l_s
		self.Structural_mass=0.0002*M_gen**2+0.6457*M_gen+645.24
		self.Mass=M_gen+self.Structural_mass
		
		# Calculating Losses and efficiency 
		# 1. Copper losses
		
		K_R=1.2 # skin effect correction coefficient
		P_Cuss=m*self.I_s**2*self.R_s*K_R                         # Copper loss-stator
		P_Cusr=m*I_r**2*self.R_R															# Copper loss-rotor
		P_Cusnom=P_Cuss+P_Cusr																#  Copper loss-total
		
		# Iron Losses ( from Hysteresis and eddy currents)      
		P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))										# Hysteresis losses in stator yoke
		P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2) 							# Eddy losses in stator yoke
		P_Hyd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))										# Hysteresis losses in stator teeth
		P_Ftd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)								# Eddy losses in stator teeth
		P_Hyyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0h*abs(self.S_Nmax)*om_e/(2*pi*60)) # Hysteresis losses in rotor yoke
		P_Ftyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0e*(abs(self.S_Nmax)*om_e/(2*pi*60))**2) #Eddy losses in rotor yoke
		P_Hydr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0h*abs(self.S_Nmax)*om_e/(2*pi*60))	# Hysteresis losses in rotor teeth
		P_Ftdr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0e*(abs(self.S_Nmax)*om_e/(2*pi*60))**2) # Eddy losses in rotor teeth
		P_add=0.5*self.machine_rating/100																									# additional losses
		P_Fesnom=P_Hyys+P_Ftys+P_Hyd+P_Ftd+P_Hyyr+P_Ftyr+P_Hydr+P_Ftdr 							# Total iron loss
		delta_v=1																																		# allowable brush voltage drop
		p_b=3*delta_v*I_r																														# Brush loss
		
		self.Losses=P_Cusnom+P_Fesnom+p_b+P_add
		self.gen_eff=(P_e-self.Losses)*100/P_e
		self.Overall_eff=self.gen_eff*self.Gearbox_efficiency
		
		# Calculating stator winding current density
		self.J_s=self.I_s/self.A_Cuscalc
		
		# Calculating  electromagnetic torque
		T_e=self.p *(self.machine_rating*1.01)/(2*pi*freq*(1-self.S_Nmax))
		
		# Calculating for tangential stress constraints
		
		self.TC1=T_e/(2*pi*sigma)
		self.TC2=self.r_s**2*self.l_s
		
		# Calculating mass moments of inertia and center of mass
		self.I = np.array([0.0, 0.0, 0.0])
		r_out=d_se*0.5
		self.I[0]   = (0.5*self.Mass*r_out**2)
		self.I[1]   = (0.25*self.Mass*r_out**2+(1/12)*self.Mass*self.l_s**2) 
		self.I[2]   = self.I[1]
		self.cm = np.array([0.0, 0.0, 0.0])
		self.cm[0]  = self.highSpeedSide_cm[0] + self.highSpeedSide_length/2. + self.l_s/2.
		self.cm[1]  = self.highSpeedSide_cm[1]
		self.cm[2]  = self.highSpeedSide_cm[2]
		
		return(self.B_g, self.B_g1, self.B_rymax,self.B_tsmax,self.B_trmax, self.q1,\
		self.N_s,self.S,self.h_ys,self.b_s,self.b_t,self.D_ratio,self.A_Cuscalc,self.Slot_aspect_ratio1,\
		self.h_yr,self.tau_p,self.p,self.Q_r,self.N_r,self.b_r,self.b_trmin, self.b_tr,self.A_Curcalc,\
		self.Slot_aspect_ratio2,self.E_p, self.f,self.I_s, self.A_1,self.J_s, self.J_r,self.R_s,self.R_R,\
		self.L_r, self.L_s,self.L_sm,self.Mass,self.K_rad,self.Losses,self.gen_eff,\
		self.Copper,self.Iron,self.Structural_mass,self.TC1,self.TC2,self.Current_ratio,self.Overall_eff,\
		self.cm,self.I)
		
class DFIG_Cost(Component):
	
	
	""" Provides a material cost estimate for a DFIG generator. Manufacturing costs are excluded"""
	
	def __init__(self):
		super(DFIG_Cost, self).__init__()
		
		# Inputs
		# Specific cost of material by type
		self.add_param('C_Cu',val=0.0, desc='Specific cost of copper')
		self.add_param('C_Fe',val=0.0,desc='Specific cost of magnetic steel/iron')
		self.add_param('C_Fes',val=0.0,desc='Specific cost of structural steel')
		
		# Mass of each material type
		
		self.add_param('Copper',val=0.0, desc='Copper mass')
		self.add_param('Iron',val=0.0, desc='Iron mass')
		self.add_param('Structural_mass',val=0.0, desc='Structural mass')
		
		# Outputs
		self.add_output('Costs',val=0.0,desc='Total cost')
		
		self.gen_costs=generator_costing()
		
	def solve_nonlinear(self,inputs,outputs,resid):
		(outputs['Costs'])=self.gen_costs.compute(inputs['Copper'],inputs['C_Cu'], \
		inputs['Iron'],inputs['C_Fe'],inputs['C_Fes'],inputs['Structural_mass'])
		return outputs

class generator_costing(object):
	
	def __init__(self):
		pass
	
	def	compute(self,Copper,C_Cu,Iron,C_Fe,C_Fes,Structural_mass):
		self.Copper=Copper
		self.Iron=Iron
		self.Structural_mass=Structural_mass
		
		
		# Material cost as a function of material mass and specific cost of material
		K_gen=self.Copper*C_Cu+(self.Iron)*C_Fe #%M_pm*K_pm; # 
		Cost_str=C_Fes*self.Structural_mass
		Costs=K_gen+Cost_str
		return(Costs)
		
		
####################################################OPTIMISATION SET_UP ###############################################################		
		
class DFIG_Opt(Group):
	
	""" Creates a new Group containing DFIG and DFIG_Cost"""
	
	def __init__(self):
		super(DFIG_Opt, self).__init__()
		
		self.add('machine_rating', IndepVarComp('machine_rating',val=0.0),promotes=['*'])
		self.add('n_nom', IndepVarComp('n_nom', val=0.0),promotes=['*'])
		
		self.add('highSpeedSide_cm', IndepVarComp('highSpeedSide_cm',val=np.array([0.0, 0.0, 0.0])),promotes=['*'])
		self.add('highSpeedSide_length',IndepVarComp('highSpeedSide_length',val=0.0),promotes=['*'])
		
		self.add('Gearbox_efficiency',IndepVarComp('Gearbox_efficiency', val=0.0),promotes=['*'])
		self.add('r_s',IndepVarComp('r_s', val=0.0),promotes=['*'])
		self.add('l_s',IndepVarComp('l_s', val=0.0),promotes=['*'])
		self.add('h_s' ,IndepVarComp('h_s', val=0.0),promotes=['*'])
		self.add('h_r' ,IndepVarComp('h_r', val=0.0),promotes=['*'])
		self.add('S_Nmax' ,IndepVarComp('S_Nmax', val=0.0),promotes=['*'])
		self.add('B_symax',IndepVarComp('B_symax', val=0.0),promotes=['*'])
		self.add('I_0',IndepVarComp('I_0', val=0.0),promotes=['*'])
		
		self.add('rho_Fe',IndepVarComp('rho_Fe',0.0),promotes=['*'])
		self.add('rho_Copper',IndepVarComp('rho_Copper',0.0),promotes=['*'])
		
		# add DFIG component, create constraint equations
		self.add('DFIG',DFIG(),promotes=['*'])
		self.add('TC', ExecComp('TC =TC2-TC1'),promotes=['*'])
		
		# add DFIG_Cost component
		self.add('DFIG_Cost',DFIG_Cost(),promotes=['*'])
		
		self.add('C_Cu',IndepVarComp('C_Cu',val=0.0),promotes=['*'])
		self.add('C_Fe',IndepVarComp('C_Fe',val=0.0),promotes=['*'])
		self.add('C_Fes',IndepVarComp('C_Fes',val=0.0),promotes=['*'])


def DFIG_Opt_example():
	opt_problem=Problem(root=DFIG_Opt())
	
	#Example optimization of a DFIG generator for costs on a 5 MW reference turbine
	
	
	# add optimizer and set-up problem (using user defined input on objective function)
	
	opt_problem.driver=pyOptSparseDriver()
	opt_problem.driver.options['optimizer'] = 'CONMIN'
	opt_problem.driver.add_objective('Costs')					# Define Objective
	opt_problem.driver.opt_settings['IPRINT'] = 4
	opt_problem.driver.opt_settings['ITRM'] = 3
	opt_problem.driver.opt_settings['ITMAX'] = 10
	opt_problem.driver.opt_settings['DELFUN'] = 1e-3
	opt_problem.driver.opt_settings['DABFUN'] = 1e-3
	opt_problem.driver.opt_settings['IFILE'] = 'CONMIN_DFIG.out'
	opt_problem.root.deriv_options['type']='fd'
	
	# Specificiency target efficiency(%)
	Eta_Target = 93.0
	
	# Set up design variables and bounds for a DFIG designed for a 5MW turbine
	
	opt_problem.driver.add_desvar('r_s', lower=0.2, upper=1.0)
	opt_problem.driver.add_desvar('l_s', lower=0.4, upper=2.0)
	opt_problem.driver.add_desvar('h_s', lower=0.045, upper=0.1)
	opt_problem.driver.add_desvar('h_r', lower=0.045, upper=0.1)
	opt_problem.driver.add_desvar('B_symax', lower=1.0, upper=2.0-1e-4)
	opt_problem.driver.add_desvar('S_Nmax', lower=-0.3, upper=-0.1)
	opt_problem.driver.add_desvar('I_0', lower=5., upper=100.0)
	
	# set up constraints for the DFIG generator
	opt_problem.driver.add_constraint('Overall_eff',lower=Eta_Target)		  			      #1
	opt_problem.driver.add_constraint('E_p',lower=500.0+1.0e-6,upper=5000-1.0e-6)			#2
	opt_problem.driver.add_constraint('TC',lower=0.0+1.0e-6)														#3
	opt_problem.driver.add_constraint('B_g',lower=0.7,upper=1.20)  										#4
	opt_problem.driver.add_constraint('B_trmax',upper=2.0-1.0e-6)											#6
	opt_problem.driver.add_constraint('B_tsmax',upper=2.0-1.0e-6) 											#7
	opt_problem.driver.add_constraint('A_1',upper=60000.0-1.0e-6)  										#8
	opt_problem.driver.add_constraint('J_s',upper=6.0)											        		#9
	opt_problem.driver.add_constraint('J_r',upper=6.0)  																#10
	opt_problem.driver.add_constraint('K_rad',lower=0.2,upper=1.5)  					      	#11 Boldea Chapter 3
	opt_problem.driver.add_constraint('D_ratio',lower=1.37,upper=1.4)  							  #12 boldea Chapter 3
	opt_problem.driver.add_constraint('Current_ratio',lower=0.1,upper=0.3)						#13 
	opt_problem.driver.add_constraint('Slot_aspect_ratio1',lower=4.0,upper=10.0)			#14
#	
	
	opt_problem.setup()
	
	# Specify Target machine parameters
	
	opt_problem['machine_rating']=5000000.0
	opt_problem['n_nom']=1200.0
	Objective_function = 'Costs'
	opt_problem['Gearbox_efficiency']=0.955
	opt_problem['r_s']= 0.61 #0.493167295965 #0.61 #meter
	opt_problem['l_s']= 0.49 #1.06173588215 #0.49 #meter
	opt_problem['h_s'] =0.08 #0.1 # 0.08 #meter
	opt_problem['h_r']= 0.1 # 0.0998797703231 #0.1 #meter
	opt_problem['I_0'] =40.0 # 40.0191207049 #40.0 #Ampere
	opt_problem['B_symax'] =1.3 #1.59611292026 #1.3 #Tesla
	opt_problem['S_Nmax'] = -0.2 #-0.3 #-0.2  #Tesla
			
	# Specific costs
	opt_problem['C_Cu']   =4.786                  # Unit cost of Copper $/kg
	opt_problem['C_Fe']= 0.556                    # Unit cost of Iron $/kg
	opt_problem['C_Fes'] =0.50139                   # specific cost of Structural_mass
	
	#Material properties
	opt_problem['rho_Fe'] = 7700.0                 #Steel density
	opt_problem['rho_Copper'] =8900.0                # Kg/m3 copper density
	
	opt_problem['highSpeedSide_cm']=np.array([0.0, 0.0, 0.0])
	opt_problem['highSpeedSide_length'] =2.0
	
	#Run optimization
	opt_problem.run()
	
	"""Uncomment to print solution to screen/ an excel file

	raw_data = {'Parameters': ['Rating','Objective function','Air gap diameter', "Stator length","K_rad","Diameter ratio", "Pole pitch(tau_p)",\
		" Number of Stator Slots","Stator slot height(h_s)","Slots/pole/phase","Stator slot width(b_s)", " Stator slot aspect ratio","Stator tooth width(b_t)",\
		"Stator yoke height(h_ys)","Rotor slots", "Rotor yoke height(h_yr)", "Rotor slot height(h_r)", "Rotor slot width(b_r)"," Rotor Slot aspect ratio",\
		"Rotor tooth width(b_t)", "Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density",\
		"Peak Stator tooth flux density","Peak rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage",\
		"Generator Output phase current","Optimal Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading",\
		"Stator resistance", "Stator leakage inductance", "Excited magnetic inductance"," Rotor winding turns","Conductor cross-section","Magnetization current",\
		"I_mag/Is"," Rotor Current density","Rotor resitance", " Rotor leakage inductance", "Generator Efficiency","Overall drivetrain Efficiency","Copper mass",\
		"Iron mass","Structural Steel mass","Total Mass","Total Material Cost"],\
		'Values': [opt_problem['machine_rating']/1e6,Objective_function,2*opt_problem['r_s'],opt_problem['l_s'],opt_problem['K_rad'],\
			opt_problem['D_ratio'],opt_problem['tau_p']*1000,opt_problem['S'],opt_problem['h_s']*1000,opt_problem['q1'],opt_problem['b_s']*1000,\
			opt_problem['Slot_aspect_ratio1'],opt_problem['b_t']*1000,opt_problem['h_ys']*1000,opt_problem['Q_r'],opt_problem['h_yr']*1000,\
			opt_problem['h_r']*1000,opt_problem['b_r']*1000,opt_problem['Slot_aspect_ratio2'],opt_problem['b_tr']*1000,opt_problem['B_g'],\
			opt_problem['B_g1'],opt_problem['B_symax'],opt_problem['B_rymax'],opt_problem['B_tsmax'],opt_problem['B_trmax'],opt_problem['p'],\
			opt_problem['f'],opt_problem['E_p'],opt_problem['I_s'],opt_problem['S_Nmax'],opt_problem['N_s'],opt_problem['A_Cuscalc'],\
			opt_problem['J_s'],opt_problem['A_1']/1000,opt_problem['R_s'],opt_problem['L_s'],opt_problem['L_sm'],opt_problem['N_r'],\
			opt_problem['A_Curcalc'],opt_problem['I_0'],opt_problem['Current_ratio'],opt_problem['J_r'],opt_problem['R_R'],opt_problem['L_r'],\
			opt_problem['gen_eff'],opt_problem['Overall_eff'],opt_problem['Copper']/1000,opt_problem['Iron']/1000,opt_problem['Structural_mass']/1000,\
			opt_problem['Mass']/1000,opt_problem['Costs']/1000],\
			'Limit': ['','','','','(0.2-1.5)','(1.37-1.4)','','','','','','(4-10)','','','','','','','','','(0.7-1.2)','','2.','2.','2.','2.','','',\
				'(500-5000)','','(-0.002-0.3)','','','(3-6)','<60','','','','','','','(0.1-0.3)','(3-6)','','','',Eta_Target,'','','','',''],\
			'Units':['MW','','m','m','-','-','mm','-','mm','','mm','','mm','mm','-','mm','mm','mm','-','mm','T','T','T','T','T','T','-','Hz','V','A','',\
				'turns','mm^2','A/mm^2','kA/m','ohms','','','turns','mm^2','A','','A/mm^2','ohms','p.u','%','%','Tons','Tons','Tons','Tons','$1000']}
	df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print(df)
	df.to_excel('DFIG_'+str(opt_problem['machine_rating']/1e6)+'MW_1.7.x.xlsx')
	
  """
	
if __name__=="__main__":
	
	# Run an example optimization of SCIG generator on cost
	DFIG_Opt_example()
