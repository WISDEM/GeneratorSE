"""py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.api import Group, Problem, Component,ExecComp,IndepVarComp,ScipyOptimizer,pyOptSparseDriver
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.drivers import *
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
import pandas as pd


class SCIG(Component):
	
	""" Estimates overall mass dimensions and Efficiency of Squirrel cage Induction generator. """
	
	def __init__(self):
		
		super(SCIG, self).__init__()
		
		# SCIG generator design inputs
		
		self.add_param('r_s', val=0.0, units ='m', desc='airgap radius r_s')
		self.add_param('l_s', val=0.0, units ='m', desc='Stator core length l_s')
		self.add_param('h_s', val=0.0, units ='m', desc='Stator slot height')
		self.add_param('h_r',val=0.0, units ='m', desc='Rotor slot height')
		
		self.add_param('machine_rating',val=0.0, units ='W', desc='Machine rating')
		self.add_param('n_nom',val=0.0, units ='rpm', desc='rated speed')
		
		self.add_param('B_symax',val=0.0, desc='Peak Stator Yoke flux density B_symax')
		self.add_param('I_0',val=0.0, desc='no-load excitation current')
		
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
		self.add_output('b_r',val=0.0, desc='rotor slot width')
		self.add_output('b_trmin',val=0.0,desc='minimum tooth width')
		self.add_output('b_tr',val=0.0, desc='rotor tooth width')
		self.add_output('A_bar',val=0.0, desc='Rotor Conductor cross-section mm^2')
		self.add_output('Slot_aspect_ratio2',val=0.0, desc='Rotor slot apsect ratio')
		self.add_output('r_r',val=0.0,desc='rotor radius')
		
		# Electrical performance
		
		self.add_output('E_p',val=0.0, desc='Stator phase voltage')
		self.add_output('f',val=0.0, desc='Generator output frequency')
		self.add_output('I_s',val=0.0, desc='Generator output phase current')
		self.add_output('A_1' ,val=0.0, desc='Specific current loading')
		self.add_output('J_s',val=0.0, desc='Stator winding Current density')
		self.add_output('J_r',val=0.0, desc='Rotor winding Current density')
		self.add_output('R_s',val=0.0, desc='Stator resistance')
		self.add_output('R_R',val=0.0, desc='Rotor resistance')
		self.add_output('L_s',val=0.0, desc='Stator synchronising inductance')
		self.add_output('L_sm',val=0.0, desc='mutual inductance')
		
		# Structural performance
		self.add_output('TC1',val=0.0, desc='Torque constraint -stator')
		self.add_output('TC2',val=0.0, desc='Torque constraint-rotor')
		
		# Mass Outputs
		
		self.add_output('Copper',val=0.0, desc='Copper Mass')
		self.add_output('Iron',val=0.0, desc='Electrical Steel Mass')
		self.add_output('Structural_mass'	,val=0.0, desc='Structural Mass')
		
		# Objective functions
		self.add_output('Mass',val=0.0, desc='Actual mass')
		self.add_output('K_rad',val=0.0, desc='Stack length ratio')
		self.add_output('Losses',val=0.0, desc='Total loss')
		self.add_output('gen_eff',val=0.0, desc='Generator efficiency')
		
		# Other parameters
		
		self.add_output('S_N',val=0.0,desc='Slip')
		self.add_output('D_ratio_UL',val=0.0, desc='Dia ratio upper limit')
		self.add_output('D_ratio_LL',val=0.0, desc='Dia ratio Lower limit')
		self.add_output('K_rad_UL',val=0.0, desc='Aspect ratio upper limit')
		self.add_output('K_rad_LL',val=0.0, desc='Aspect ratio Lower limit')
		
		self.add_output('Overall_eff',val=0.0, desc='Overall drivetrain efficiency')
		self.add_output('I',val=np.array([0.0, 0.0, 0.0]),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
		self.add_output('cm', val=np.array([0.0, 0.0, 0.0]),desc='COM [x,y,z]')
		self.gen_sizing = generator_sizing()
		
	def solve_nonlinear(self, inputs, outputs, resid):
		(outputs['B_tsmax'],outputs['B_trmax'], outputs['B_rymax'], outputs['B_g'],outputs['B_g1'],\
		outputs['q1'],outputs['N_s'],outputs['S'],outputs['h_ys'],outputs['b_s'],outputs['b_t'],\
		outputs['D_ratio'],outputs['D_ratio_UL'],outputs['D_ratio_LL'],outputs['A_Cuscalc'],outputs['Slot_aspect_ratio1'],outputs['h_yr'],\
		outputs['tau_p'],outputs['p'], outputs['Q_r'], outputs['b_r'],outputs['b_trmin'],\
		outputs['b_tr'],outputs['r_r'],outputs['S_N'],outputs['A_bar'],outputs['Slot_aspect_ratio2'],outputs['E_p'], outputs['f'], \
		outputs['I_s'], outputs['A_1'],outputs['J_s'],outputs['J_r'],outputs['R_s'], outputs['R_R'],\
		outputs['L_s'],outputs['L_sm'],outputs['Mass'],outputs['K_rad'],\
		outputs['K_rad_UL'],outputs['K_rad_LL'],outputs['Losses'],outputs['gen_eff'],outputs['Copper'],outputs['Iron'],outputs['Structural_mass'], outputs['TC1'],outputs['TC2'],\
		outputs['Overall_eff'],outputs['cm'],outputs['I'])\
		= self.gen_sizing.compute(inputs['r_s'], inputs['l_s'], inputs['h_s'], inputs['h_r'],inputs['Gearbox_efficiency'],inputs['machine_rating'],
		inputs['n_nom'], inputs['B_symax'],inputs['I_0'],inputs['rho_Fe'], inputs['rho_Copper'], \
		inputs['highSpeedSide_cm'],inputs['highSpeedSide_length'])

		return outputs  
     

class generator_sizing(object):
	
	def __init__(self):
		
		pass
		
	def	compute(self,r_s, l_s,h_s,h_r,Gearbox_efficiency,machine_rating,n_nom,B_symax,I_0, \
	rho_Fe,rho_Copper,highSpeedSide_cm,highSpeedSide_length):
		
		
		#Create internal variables based on inputs
		
		self.r_s = r_s
		self.l_s = l_s
		self.h_s = h_s
		self.h_r = h_r
		
		self.machine_rating=machine_rating
		self.n_nom=n_nom
		self.Gearbox_efficiency=Gearbox_efficiency
		self.I_0=I_0
		self.rho_Fe= rho_Fe
		self.rho_Copper=rho_Copper
		
		self.B_symax=B_symax
		
		self.highSpeedSide_cm=highSpeedSide_cm
		self.highSpeedSide_length=highSpeedSide_length
		
		#Assign values to universal constants
		
		g1          =9.81                  # m/s^2 acceleration due to gravity
		sigma       =21.5e3                # shear stress 
		mu_0        =pi*4e-7               # permeability of free space
		cofi        =0.9                   # power factor
		h_w         = 0.005                # wedge height
		m           =3                     # Number of phases
		
		#Assign values to design constants
		
		self.q1     =6                      # Number of slots per pole per phase
		b_s_tau_s   =0.45                   # Stator Slot width/Slot pitch ratio
		b_r_tau_r   =0.45                   # Rotor Slot width/Slot pitch ratio
		self.S_N    =-0.002                 # Slip
		y_tau_p     =12./15                 # Coil span/pole pitch
		freq        =60											# frequency in Hz
		
		k_y1        =sin(pi*0.5*y_tau_p)    # winding Chording factor
		k_q1        =sin(pi/6)/self.q1/sin(pi/6/self.q1)# # zone factor
		k_wd        =k_y1*k_q1              # Calcuating winding factor
		P_Fe0h      =4                      #specific hysteresis losses W/kg @ 1.5 T @50 Hz
		P_Fe0e      =1                      #specific hysteresis losses W/kg @ 1.5 T @50 Hz
		rho_Cu      =1.8*10**(-8)*1.4       # Copper resistivity
		
		n_1         =self.n_nom/(1-self.S_N) # actual rotor speed
		
		dia         =2*self.r_s             			# air gap diameter
		self.p      = 3  													# number of pole pairs
		self.K_rad=self.l_s/dia										# Aspect ratio
		self.K_rad_LL=0.5   											# lower limit on aspect ratio
		self.K_rad_UL =1.5   											# upper limit on aspect ratio
		
		# Calculating air gap length
		
		g=(0.1+0.012*(self.machine_rating)**(1./3))*1e-3
		self.r_r          =self.r_s-g             #rotor radius
		
		self.tau_p=pi*dia/2/self.p								# Calculating pole pitch
		self.S=2*m*self.p*self.q1                 # Calculating Stator slots
		
		tau_s=self.tau_p/m/self.q1								# Stator slot pitch
		N_slots_pp  =self.S/(m*self.p*2)				  # Slots per pole per phase
		
		self.b_s          =b_s_tau_s*tau_s        # Calculating stator slot width
		b_so        =0.004;												#Stator slot opening wdth
		b_ro        =0.004;												#Rotor slot opening wdth
		self.b_t        =tau_s-self.b_s           #tooth width
		
		q2          =4               							# rotor slots per pole per phase
		self.Q_r    =2*self.p*m*q2								# Calculating Rotor slots
		tau_r       =pi*(dia-2*g)/self.Q_r				# rotot slot pitch
		self.b_r          =b_r_tau_r*tau_r				# Rotor slot width
		self.b_tr=tau_r-self.b_r									# rotor tooth width
		tau_r_min=pi*(dia-2*(g+self.h_r))/self.Q_r
		self.b_trmin=tau_r_min-b_r_tau_r*tau_r_min # minumum rotor tooth width
		
		
		# Calculating equivalent slot openings
		
		mu_rs       =0.005
		mu_rr       =0.005
		W_s         =(self.b_s/mu_rs)*1e-3       # Stator
		W_r         =(self.b_r/mu_rr)*1e-3				# Rotor
		
		self.Slot_aspect_ratio1=self.h_s/self.b_s  # Stator slot aspect ratio
		self.Slot_aspect_ratio2=self.h_r/self.b_r  # Rotor slot aspect ratio
		
		# Calculating Carter factor for stator,rotor and effective air gap length
		gamma_s     = (2*W_s/g)**2/(5+2*W_s/g)
		K_Cs        =(tau_s)/(tau_s-g*gamma_s*0.5)  #page 3-13 Boldea Induction machines Chapter 3
		gamma_r     = (2*W_r/g)**2/(5+2*W_r/g)
		K_Cr        =(tau_r)/(tau_r-g*gamma_r*0.5)  #page 3-13 Boldea Boldea Induction machines Chapter 3
		K_C         =K_Cs*K_Cr
		g_eff       =K_C*g
		
		om_m        =2*pi*self.n_nom/60							# mechanical frequency
		om_e        =self.p*om_m										# electrical frequency
		self.f      =self.n_nom*self.p/60
		K_s         =0.3  													# saturation factor for Iron
		n_c         =2    													#number of conductors per coil
		a1          =2    													# number of parallel paths
		
		# Calculating stator winding turns
		self.N_s    =round(2*self.p*N_slots_pp*n_c/a1)
		
		# Calculating Peak flux densities
		
		self.B_g1   =mu_0*3*self.N_s*self.I_0*sqrt(2)*k_y1*k_q1/(pi*self.p*g_eff*(1+K_s))
		self.B_g    =self.B_g1*K_C
		self.B_rymax=self.B_symax
		
		# calculating back iron thickness
		
		self.h_ys= self.B_g*self.tau_p/(self.B_symax*pi)
		self.h_yr= self.h_ys
		
		d_se        =dia+2*(self.h_ys+self.h_s+h_w)  # stator outer diameter
		self.D_ratio=d_se/dia                        # Diameter ratio
		
		# limits for Diameter ratio depending on pole pair
		
		if (2*self.p==2):
			self.D_ratio_LL =1.65
			self.D_ratio_UL =1.69
		elif (2*self.p==4):
			self.D_ratio_LL =1.46
			self.D_ratio_UL =1.49
		elif (2*self.p==6):
			self.D_ratio_LL =1.37
			self.D_ratio_UL =1.4
		elif (2*self.p==8):
			self.D_ratio_LL =1.27
			self.D_ratio_UL =1.3
		else:
			self.D_ratio_LL =1.2
			self.D_ratio_UL =1.24
			
		# Stator slot fill factor
		
		if (2*self.r_s>2):
			K_fills=0.65
		else:
			K_fills=0.4
		
		# Stator winding length and cross-section
		l_fs=2*(0.015+y_tau_p*self.tau_p/2/cos(40*pi/180))+pi*(self.h_s) # end connection
		l_Cus = 2*self.N_s*(l_fs+self.l_s)/a1                	           #shortpitch
		A_s = self.b_s*(self.h_s-h_w)																		 #Slot area
		A_scalc=self.b_s*1000*(self.h_s*1000-h_w*1000)						       #Conductor cross-section (mm^2)
		A_Cus = A_s*self.q1*self.p*K_fills/self.N_s											 #Conductor cross-section (m^2)
		
		self.A_Cuscalc = A_scalc*self.q1*self.p*K_fills/self.N_s
		
		# Stator winding resistance
		self.R_s          =l_Cus*rho_Cu/A_Cus
		
		# Calculating no-load voltage
		om_s        =(self.n_nom)*2*pi/60																 # rated angular frequency			
		P_e         =self.machine_rating/(1-self.S_N)										 # Electrical power
		self.E_p    =om_s*self.N_s*k_wd*self.r_s*self.l_s*self.B_g1*sqrt(2)
		
		S_GN=(self.machine_rating-self.S_N*self.machine_rating)
		T_e         =self.p *(S_GN)/(2*pi*freq*(1-self.S_N))
		I_srated    =self.machine_rating/3/self.E_p/cofi
		
		#Rotor design
		k_fillr     = 0.7																		#	Rotor slot fill factor
		diff        = self.h_r-h_w
		self.A_bar      = self.b_r*diff											# bar cross section
		Beta_skin   = sqrt(pi*mu_0*freq/2/rho_Cu)           #co-efficient for skin effect correction
		k_rm        = Beta_skin*self.h_r                    #co-efficient for skin effect correction
		J_b         = 6e+06																	# Bar current density
		K_i         = 0.864
		I_b         = 2*m*self.N_s*k_wd*I_srated/self.Q_r    # bar current
		
		# Calculating bar resistance
		
		R_rb        =rho_Cu*k_rm*(self.l_s)/(self.A_bar)
		I_er        =I_b/(2*sin(pi*self.p/self.Q_r))				# End ring current
		J_er        = 0.8*J_b																# End ring current density
		A_er        =I_er/J_er															# End ring cross-section
		b           =self.h_r																# End ring dimension
		a           =A_er/b 																# End ring dimension
		D_er=(self.r_s*2-2*g)-0.003													# End ring diameter
		l_er=pi*(D_er-b)/self.Q_r														#  End ring segment length
		
		# Calculating end ring resistance
		R_re=rho_Cu*l_er/(2*A_er*(sin(pi*self.p/self.Q_r))**2)
		
		# Calculating equivalent rotor resistance
		self.R_R=(R_rb+R_re)*4*m*(k_wd*self.N_s)**2/self.Q_r
		
		# Calculating Rotor and Stator teeth flux density
		self.B_trmax = self.B_g*tau_r/self.b_trmin
		self.B_tsmax=tau_s*self.B_g/self.b_t
		
		# Calculating Equivalent core lengths
		l_r=self.l_s+(4)*g  # for axial cooling
		l_se =self.l_s+(2/3)*g
		K_fe=0.95           # Iron factor
		L_e=l_se *K_fe   # radial cooling
		
		# Calculating leakage inductance in  stator
		L_ssigmas=(2*mu_0*self.l_s*self.N_s**2/self.p/self.q1)*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
		L_ssigmaew=(2*mu_0*self.l_s*self.N_s**2/self.p/self.q1)*0.34*self.q1*(l_fs-0.64*self.tau_p*y_tau_p)/self.l_s #end winding leakage inductance
		L_ssigmag=2*mu_0*self.l_s*self.N_s**2/self.p/self.q1*(5*(g*K_C/b_so)/(5+4*(g*K_C/b_so))) # tooth tip leakage inductance
		self.L_s=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
		self.L_sm =6*mu_0*self.l_s*self.tau_p*(k_wd*self.N_s)**2/(pi**2*(self.p)*g_eff*(1+K_s))
		
		lambda_ei=2.3*D_er/(4*self.Q_r*self.l_s*(sin(pi*self.p/self.Q_r)**2))*log(4.7*dia/(a+2*b))
		lambda_b=self.h_r/3/self.b_r+h_w/b_ro
		L_i=pi*dia/self.Q_r
		
		# Calculating leakage inductance in  rotor
		L_rsl=(mu_0*self.l_s)*((self.h_r-h_w)/(3*self.b_r)+h_w/b_ro)  #slot leakage inductance
		L_rel=mu_0*(self.l_s*lambda_b+2*lambda_ei*L_i)                  #end winding leakage inductance
		L_rtl=(mu_0*self.l_s)*(0.9*tau_r*0.09/g_eff) # tooth tip leakage inductance
		L_rsigma=(L_rsl+L_rtl+L_rel)*4*m*(k_wd*self.N_s)**2/self.Q_r  # rotor leakage inductance
		
		# Calculating rotor current
		I_r=sqrt(-self.S_N*P_e/m/self.R_R)
		
		I_sm=self.E_p/(2*pi*freq*self.L_sm)
		# Calculating stator currents and specific current loading
		self.I_s=sqrt((I_r**2+I_sm**2))
		
		self.A_1=2*m*self.N_s*self.I_s/pi/(2*self.r_s)
		
		# Calculating masses of the electromagnetically active materials
		
		V_Cuss=m*l_Cus*A_Cus																				# Volume of copper in stator
		V_Cusr=(self.Q_r*self.l_s*self.A_bar+pi*(D_er*A_er-A_er*b))	# Volume of copper in rotor
		V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-2*m*self.q1*self.p*self.b_s*self.h_s*self.l_s) # Volume of iron in stator teeth
		V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)				# Volume of iron in stator yoke
		self.r_r=self.r_s-g																							# rotor radius
		
		V_Fert=pi*self.l_s*(self.r_r**2-(self.r_r-self.h_r)**2)-2*m*q2*self.p*self.b_r*self.h_r*self.l_s # Volume of iron in rotor teeth
		V_Fery=self.l_s*pi*((self.r_r-self.h_r)**2-(self.r_r-self.h_r-self.h_yr)**2)										 # Volume of iron in rotor yoke
		self.Copper=(V_Cuss+V_Cusr)*self.rho_Copper							    # Mass of Copper
		M_Fest=V_Fest*self.rho_Fe																		# Mass of stator teeth
		M_Fesy=V_Fesy*self.rho_Fe																		# Mass of stator yoke
		M_Fert=V_Fert*self.rho_Fe																		# Mass of rotor tooth
		M_Fery=V_Fery*self.rho_Fe																		# Mass of rotor yoke
		self.Iron=M_Fest+M_Fesy+M_Fert+M_Fery
		
		self.Active_mass=(self.Copper+self.Iron)
		L_tot=self.l_s
		self.Structural_mass=0.0001*self.Active_mass**2+0.8841*self.Active_mass-132.5
		self.Mass=self.Active_mass+self.Structural_mass
		
		# Calculating Losses and efficiency
		
		# 1. Copper losses
		
		K_R=1.2 # skin effect correction coefficient
		P_Cuss=m*self.I_s**2*self.R_s*K_R  # Copper loss-stator
		P_Cusr=m*I_r**2*self.R_R					 # Copper loss-rotor
		P_Cusnom=P_Cuss+P_Cusr						 # Copper loss-total
		
		# Iron Losses ( from Hysteresis and eddy currents)          
		P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))         # Hysteresis losses in stator yoke
		P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)    # Eddy losses in stator yoke
		P_Hyd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))					# Hysteresis losses in stator tooth
		P_Ftd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)     # Eddy losses in stator tooth
		P_Hyyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0h*abs(self.S_N)*om_e/(2*pi*60)) # Hysteresis losses in rotor yoke
		P_Ftyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0e*(abs(self.S_N)*om_e/(2*pi*60))**2) # Eddy losses in rotor yoke
		P_Hydr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0h*abs(self.S_N)*om_e/(2*pi*60))			# Hysteresis losses in rotor tooth
		P_Ftdr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0e*(abs(self.S_N)*om_e/(2*pi*60))**2) # Eddy losses in rotor tooth
		
		# Calculating Additional losses
		P_add=0.5*self.machine_rating/100
		P_Fesnom=P_Hyys+P_Ftys+P_Hyd+P_Ftd+P_Hyyr+P_Ftyr+P_Hydr+P_Ftdr
		self.Losses=P_Cusnom+P_Fesnom+P_add
		self.gen_eff=(P_e-self.Losses)*100/P_e
		self.Overall_eff=self.gen_eff*self.Gearbox_efficiency
		
		# Calculating current densities in the stator and rotor
		self.J_s=self.I_s/self.A_Cuscalc
		self.J_r=I_r/(self.A_bar)/1e6
		
		# Calculating Tangential stress constraints
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
		
		return(self.B_tsmax, self.B_trmax, self.B_rymax,self.B_g,self.B_g1, self.q1,\
		self.N_s,self.S,self.h_ys,self.b_s,self.b_t,self.D_ratio,self.D_ratio_UL,self.D_ratio_LL,self.A_Cuscalc,self.Slot_aspect_ratio1,\
		self.h_yr,self.tau_p,self.p,self.Q_r,self.b_r,self.b_trmin, self.b_tr,self.r_r,self.S_N,self.A_bar,\
		self.Slot_aspect_ratio2,self.E_p, self.f,self.I_s, self.A_1,self.J_s, self.J_r,self.R_s,self.R_R,\
		self.L_s,self.L_sm, self.Mass,self.K_rad,self.K_rad_UL,self.K_rad_LL,self.Losses,self.gen_eff,\
		self.Copper,self.Iron,self.Structural_mass,self.TC1,self.TC2,self.Overall_eff,\
		self.cm,self.I)

class SCIG_Cost(Component):
	
	""" Provides a material cost estimate for a DFIG generator. Manufacturing costs are excluded"""
	
	def __init__(self):
		super(SCIG_Cost, self).__init__()
		
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

class SCIG_Opt(Group):
	
	""" Creates a new Group containing DFIG and DFIG_Cost"""
	
	def __init__(self):
		
		super(SCIG_Opt, self).__init__()
		self.add('machine_rating', IndepVarComp('machine_rating',0.0),promotes=['*'])
		self.add('n_nom', IndepVarComp('n_nom', val=0.0),promotes=['*'])
		
		self.add('highSpeedSide_cm', IndepVarComp('highSpeedSide_cm',val=np.array([0.0, 0.0, 0.0])),promotes=['*'])
		self.add('highSpeedSide_length',IndepVarComp('highSpeedSide_length',val=0.0),promotes=['*'])
		
		self.add('Gearbox_efficiency',IndepVarComp('Gearbox_efficiency', val=0.0),promotes=['*'])
		self.add('r_s',IndepVarComp('r_s', val=0.0),promotes=['*'])
		self.add('l_s',IndepVarComp('l_s', val=0.0),promotes=['*'])
		self.add('h_s' ,IndepVarComp('h_s', val=0.0),promotes=['*'])
		self.add('h_r' ,IndepVarComp('h_r', val=0.0),promotes=['*'])
		self.add('B_symax',IndepVarComp('B_symax', val=0.0),promotes=['*'])
		self.add('I_0',IndepVarComp('I_0', val=0.0),promotes=['*'])
		
		self.add('rho_Fe',IndepVarComp('rho_Fe',0.0),promotes=['*'])
		self.add('rho_Copper',IndepVarComp('rho_Copper',0.0),promotes=['*'])
		
		# add DFIG component, create constraint equations
		self.add('SCIG',SCIG(),promotes=['*'])
		self.add('TC', ExecComp('TC =TC2-TC1'),promotes=['*'])
		self.add('K_rad_L',ExecComp('K_rad_L=K_rad-K_rad_LL'),promotes=['*'])
		self.add('K_rad_U',ExecComp('K_rad_U=K_rad-K_rad_UL'),promotes=['*'])
		self.add('D_ratio_L',ExecComp('D_ratio_L=D_ratio-D_ratio_LL'),promotes=['*'])
		self.add('D_ratio_U',ExecComp('D_ratio_U=D_ratio-D_ratio_UL'),promotes=['*'])
		
		# add DFIG_Cost component
		self.add('SCIG_Cost',SCIG_Cost(),promotes=['*'])
		
		self.add('C_Cu',IndepVarComp('C_Cu',val=0.0),promotes=['*'])
		self.add('C_Fe',IndepVarComp('C_Fe',val=0.0),promotes=['*'])
		self.add('C_Fes',IndepVarComp('C_Fes',val=0.0),promotes=['*'])

def SCIG_Opt_example():
	
	opt_problem=Problem(root=SCIG_Opt())
	
	#Example optimization of a SCIG generator for costs on a 5 MW reference turbine
	
	# add optimizer and set-up problem (using user defined input on objective function)
	
	opt_problem.driver=pyOptSparseDriver()
	opt_problem.driver.options['optimizer'] = 'CONMIN'
	opt_problem.driver.opt_settings['IPRINT'] = 4
	opt_problem.driver.opt_settings['ITRM'] = 3
	opt_problem.driver.opt_settings['ITMAX'] = 10
	opt_problem.driver.opt_settings['DELFUN'] = 1e-3
	opt_problem.driver.opt_settings['DABFUN'] = 1e-3
	opt_problem.driver.opt_settings['IFILE'] = 'CONMIN_SCIG_out'
	opt_problem.root.deriv_options['type']='fd'
	
	# Specificiency target efficiency(%)
	Eta_Target = 93.0
	
	# Set up design variables and bounds for a SCIG designed for a 5MW turbine
	
	opt_problem.driver.add_desvar('r_s', lower=0.2, upper=1.0)
	opt_problem.driver.add_desvar('l_s', lower=0.4, upper=2.0)
	opt_problem.driver.add_desvar('h_s', lower=0.04, upper=0.1)
	opt_problem.driver.add_desvar('h_r', lower=0.04, upper=0.1)
	opt_problem.driver.add_desvar('B_symax', lower=1.0, upper=2.0)
	opt_problem.driver.add_desvar('I_0', lower=5.0, upper=200.0)
#	
#	set up constraints for the SCIG generator
#	
	opt_problem.driver.add_constraint('Overall_eff',lower=Eta_Target)          		# 1
	opt_problem.driver.add_constraint('E_p',lower=500.0+1.0e-6,upper=5000.0-1.0e-6)#2               		# 3
	opt_problem.driver.add_constraint('TC',lower=0.0+8.791e-3)                   		# 3
	opt_problem.driver.add_constraint('B_g',lower=0.7,upper=1.2)               		# 4
	opt_problem.driver.add_constraint('B_rymax',upper=2.0-1.0e-6)              		# 5
	opt_problem.driver.add_constraint('B_trmax',upper=2.0-1.0e-6)              		# 6
	opt_problem.driver.add_constraint('B_tsmax',upper=2.0-1.0e-6)              		# 7
	opt_problem.driver.add_constraint('A_1',upper=60000.0-1.0e-6)             		# 8
	opt_problem.driver.add_constraint('J_s',upper=6.0)                         		# 9
	opt_problem.driver.add_constraint('J_r',upper=6.0)                         		# 12
	opt_problem.driver.add_constraint('K_rad_L',lower=0.0)   											# 13 #boldea Chapter 3
	opt_problem.driver.add_constraint('K_rad_U',upper=0.0)												# 14
	opt_problem.driver.add_constraint('D_ratio_L',lower=0.0)											# 15
	opt_problem.driver.add_constraint('D_ratio_U',upper=0.0)											# 16
	opt_problem.driver.add_constraint('Slot_aspect_ratio1',lower=4.0,upper=10.0)  # 17
	
	opt_problem.setup()
	
	# Specify Target machine parameters
	
	opt_problem['machine_rating']=5000000.0
	opt_problem['n_nom']=1200.0
	Objective_function = 'Costs'
	opt_problem['Gearbox_efficiency']=0.955
	opt_problem['r_s']= 0.55 #0.484689156353 #0.55 #meter
	opt_problem['l_s']= 1.30 #1.27480124244 #1.3 #meter
	opt_problem['h_s'] =0.090 #0.098331868116 # 0.090 #meter
	opt_problem['h_r'] =0.050 #0.04 # 0.050 #meter
	opt_problem['I_0'] = 140  #139.995232826 #140  #Ampere
	opt_problem['B_symax'] = 1.4 #1.86140258387 #1.4 #Tesla
	
	# Specific costs
	opt_problem['C_Cu']   =4.786                  # Unit cost of Copper $/kg
	opt_problem['C_Fe']= 0.556                    # Unit cost of Iron $/kg
	opt_problem['C_Fes'] =0.50139                   # specific cost of Structural_mass
	
	#Material properties
	opt_problem['rho_Fe'] = 7700.0                 #Steel density
	opt_problem['rho_Copper'] =8900.0                # Kg/m3 copper density
	
	#Run optimization
	opt_problem.driver.add_objective(Objective_function)					# Define Objective					# Define Objective
	opt_problem.run()
	
	""" Uncomment to print solution to screen/an excel file
	
	raw_data = {"Parameters": ["Rating","Objective function","Air gap diameter", "Stator length","Lambda ratio","Diameter ratio", "Pole pitch(tau_p)",\
		"Number of Stator Slots","Stator slot height(h_s)","Stator slot width(b_s)","Stator Slot aspect ratio", "Stator tooth width(b_t)", "Stator yoke height(h_ys)",\
		"Rotor slots", "Rotor slot height(h_r)", "Rotor slot width(b_r)","Rotor tooth width(b_tr)","Rotor yoke height(h_yr)","Rotor Slot_aspect_ratio","Peak air gap flux density",\
		"Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak Rotor tooth flux density",\
		"Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Slip","Stator Turns","Conductor cross-section",\
		"Stator Current density","Specific current loading","Stator resistance", "Excited magnetic inductance","Magnetization current","Conductor cross-section",\
		" Rotor Current density","Rotor resitance", "Generator Efficiency","Overall drivetrain Efficiency","Copper mass","Iron Mass", "Structural mass","Total Mass","Total Material Cost"],\
		"Values": [opt_problem['machine_rating']/1e6,Objective_function,2*opt_problem['r_s'],opt_problem['l_s'],opt_problem['K_rad'],opt_problem['D_ratio'],opt_problem['tau_p']*1000,\
		opt_problem['S'],opt_problem['h_s']*1000,opt_problem['b_s']*1000,opt_problem['Slot_aspect_ratio1'],opt_problem['b_t']*1000,opt_problem['h_ys']*1000,opt_problem['Q_r'],\
  	opt_problem['h_r']*1000,opt_problem['b_r']*1000,opt_problem['b_tr']*1000,opt_problem['h_yr']*1000,opt_problem['Slot_aspect_ratio2'],opt_problem['B_g'],\
  	opt_problem['B_g1'],opt_problem['B_symax'],opt_problem['B_rymax'],opt_problem['B_tsmax'],opt_problem['B_trmax'],opt_problem['p'],opt_problem['f'],\
  	opt_problem['E_p'],opt_problem['I_s'],opt_problem['S_N'],opt_problem['N_s'],opt_problem['A_Cuscalc'],opt_problem['J_s'],opt_problem['A_1']/1000,\
  	opt_problem['R_s'],opt_problem['L_sm'],opt_problem['I_0'],opt_problem['A_bar']*1e6,opt_problem['J_r'],opt_problem['R_R'],opt_problem['gen_eff'],\
  	opt_problem['Overall_eff'],opt_problem['Copper']/1000,opt_problem['Iron']/1000,opt_problem['Structural_mass']/1000,opt_problem['Mass']/1000,\
  	opt_problem['Costs']/1000],\
  	"Limit": ['','','','',"("+str(opt_problem['K_rad_LL'])+"-"+str(opt_problem['K_rad_UL'])+")","("+str(opt_problem['D_ratio_LL'])+"-"+str(opt_problem['D_ratio_UL'])+")",\
  	'','','','','(4-10)','','','','','','','','(4-10)','(0.7-1.2)','','2','2','2','2','','(10-60)','(500-5000)','','(-30% to -0.2%)','','','(3-6)','<60','','','',\
  	'','','','',Eta_Target,'','','','',''],\
  	"Units":['MW','','m','m','-','-','mm','-','mm','mm','-','mm','mm','-','mm','mm','mm','mm','','T','T','T','T','T','T','-','Hz','V','A','%','turns','mm^2','A/mm^2',\
  	'kA/m','ohms','p.u','A','mm^2','A/mm^2','ohms','%','%','Tons','Tons','Tons','Tons','$1000']}
  	
  	df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
  	print(df)
  	df.to_excel('SCIG_'+str(opt_problem['machine_rating']/1e6)+'_MW_1.7.x.xlsx')
  
  """ 

if __name__=="__main__":

    # Run an example optimization of SCIG generator on cost
    SCIG_Opt_example()
