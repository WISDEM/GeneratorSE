"""PMSG_disc.py
Created by Latha Sethuraman, Katherine Dykes. 
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

from openmdao.api import Group, Problem, Component,ExecComp,IndepVarComp,ScipyOptimizer,pyOptSparseDriver
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.drivers import *
import numpy as np
from numpy import array,float,min,sign
from math import pi, cos,cosh, sqrt, radians, sin,sinh, exp, log10, log, tan, atan
import pandas as pd


class PMSG(Component):
	
	""" Estimates overall mass dimensions and Efficiency of PMSG -disc rotor generator. """
	
	def __init__(self):
		
		super(PMSG, self).__init__()
		
		# PMSG_disc generator design inputs
		
		self.add_param('r_s', val=0.0, units ='m', desc='airgap radius r_s')
		self.add_param('l_s', val=0.0, units ='m', desc='Stator core length l_s')
		self.add_param('h_s', val=0.0, units ='m', desc='Yoke height h_s')
		self.add_param('tau_p',val=0.0, units ='m', desc='Pole pitch self.tau_p')
		
		self.add_param('machine_rating',val=0.0, units ='W', desc='Machine rating')
		self.add_param('n_nom',val=0.0, units ='rpm', desc='rated speed')
		self.add_param('Torque',val=0.0, units ='Nm', desc='Rated torque ')
		self.add_param('h_m',val=0.0, units ='m', desc='magnet height')
		self.add_param('h_ys',val=0.0, units ='m', desc='Yoke height')
		self.add_param('h_yr',val=0.0, units ='m', desc='rotor yoke height')
		
		# structural design variables
		self.add_param('n_s' ,val=0.0, desc='number of stator arms n_s')
		self.add_param('b_st' , val=0.0, units ='m', desc='arm width b_st')
		self.add_param('d_s',val=0.0,units ='m', desc='arm depth d_s')
		self.add_param('t_ws' ,val=0.0,units ='m', desc='arm depth thickness ')
		
		self.add_param('t_d',val=0.0, units='m', desc='disc thickness')
		self.add_param('R_o',val=0.0, units ='m',desc='Shaft radius')
		
		# PMSG_disc generator design outputs
		# Magnetic loading
		self.add_output('B_symax' ,val=0.0, desc='Peak Stator Yoke flux density B_ymax')
		self.add_output('B_tmax',val=0.0, desc='Peak Teeth flux density')
		self.add_output('B_rymax',val=0.0, desc='Peak Rotor yoke flux density')
		self.add_output('B_smax',val=0.0, desc='Peak Stator flux density')
		self.add_output('B_pm1',val=0.0, desc='Fundamental component of peak air gap flux density')
		self.add_output('B_g' ,val=0.0, desc='Peak air gap flux density B_g')
		
		#Stator design
		self.add_output('N_s' ,val=0.0, desc='Number of turns in the stator winding')
		self.add_output('b_s',val=0.0, desc='slot width')
		self.add_output('b_t',val=0.0, desc='tooth width')
		self.add_output('A_Cuscalc',val=0.0, desc='Conductor cross-section mm^2')
		
		#Rotor magnet dimension
		self.add_output('b_m',val=0.0, desc='magnet width')
		self.add_output('p',val=0.0, desc='No of pole pairs')
		
		# Electrical performance
		self.add_output('E_p',val=0.0, desc='Stator phase voltage')
		self.add_output('f',val=0.0, desc='Generator output frequency')
		self.add_output('I_s',val=0.0, desc='Generator output phase current')
		self.add_output('R_s',val=0.0, desc='Stator resistance')
		self.add_output('L_s',val=0.0, desc='Stator synchronising inductance')
		self.add_output('A_1' ,val=0.0, desc='Electrical loading')
		self.add_output('J_s',val=0.0, desc='Current density')
		
		# Objective functions
		self.add_output('Mass',val=0.0, desc='Actual mass')
		self.add_output('K_rad',val=0.0, desc='K_rad')
		self.add_output('Losses',val=0.0, desc='Total loss')
		self.add_output('gen_eff',val=0.0, desc='Generator efficiency')
	
		# Structural performance
		self.add_output('u_Ar',val=0.0, desc='Rotor radial deflection')
		self.add_output('y_Ar',val=0.0, desc='Rotor axial deflection')
		self.add_output('u_As',val=0.0, desc='Stator radial deflection')
		self.add_output('y_As',val=0.0, desc='Stator axial deflection')
		self.add_output('z_A_s',val=0.0, desc='Stator circumferential deflection')  
		self.add_output('u_all_r',val=0.0, desc='Allowable radial rotor')
		self.add_output('u_all_s',val=0.0, desc='Allowable radial stator')
		self.add_output('y_all',val=0.0, desc='Allowable axial')
		self.add_output('z_all_s',val=0.0, desc='Allowable circum stator')
		self.add_output('z_all_r',val=0.0, desc='Allowable circum rotor')
		self.add_output('b_all_s',val=0.0, desc='Allowable arm')
		self.add_output('TC1',val=0.0, desc='Torque constraint')
		self.add_output('TC2',val=0.0, desc='Torque constraint-rotor')
		self.add_output('TC3',val=0.0, desc='Torque constraint-stator')
		
		# Other parameters
		self.add_output('R_out',val=0.0, desc='Outer radius')
		self.add_output('S'		,val=0.0, desc='Stator slots')
		self.add_output('Slot_aspect_ratio',val=0.0, desc='Slot aspect ratio')
		
		# Mass Outputs
		self.add_output('mass_PM',val=0.0, desc='Magnet mass')
		self.add_output('Copper',val=0.0, desc='Copper Mass')
		self.add_output('Iron',val=0.0, desc='Electrical Steel Mass')
		self.add_output('Structural_mass'	,val=0.0, desc='Structural Mass')
		
		# Material properties
		self.add_param('rho_Fes',val=0.0,units='kg*m**-3', desc='Structural Steel density ')
		self.add_param('rho_Fe',val=0.0,units='kg*m**-3', desc='Magnetic Steel density ')
		self.add_param('rho_Copper',val=0.0,units='kg*m**-3', desc='Copper density ')
		self.add_param('rho_PM',val=0.0,units='kg*m**-3', desc='Magnet density ')
		
		#inputs/outputs for interface with drivese
		self.add_param('main_shaft_cm',val= np.array([0.0, 0.0, 0.0]), desc='Main Shaft CM')
		self.add_param('main_shaft_length',val=0.0, desc='main shaft length')
		self.add_output('I',val=np.array([0.0, 0.0, 0.0]),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
		self.add_output('cm', val=np.array([0.0, 0.0, 0.0]),desc='COM [x,y,z]')
		
		self.gen_sizing = generator_sizing()
		
	def solve_nonlinear(self, inputs, outputs, resid):
		(outputs['B_symax'], outputs['B_tmax'], outputs['B_rymax'], outputs['B_smax'], outputs['B_pm1'], outputs['B_g'], outputs['N_s'], outputs['b_s'], \
		outputs['b_t'], outputs['A_Cuscalc'], outputs['b_m'], outputs['p'], outputs['E_p'], outputs['f'], \
		outputs['I_s'], outputs['R_s'], outputs['L_s'], outputs['A_1'], outputs['J_s'], outputs['Mass'],\
		outputs['K_rad'],outputs['Losses'], outputs['gen_eff'], outputs['u_Ar'], outputs['y_Ar'], \
		outputs['u_As'], outputs['y_As'], outputs['z_A_s'], outputs['u_all_r'], outputs['u_all_s'], \
		outputs['y_all'], outputs['z_all_s'],outputs['z_all_r'],  outputs['b_all_s'], outputs['TC1'], \
		outputs['TC2'], outputs['TC3'], outputs['R_out'], outputs['S'],outputs['Slot_aspect_ratio'],\
		outputs['cm'], outputs['I'],outputs['mass_PM'],outputs['Copper'],outputs['Iron'],outputs['Structural_mass'])\
		= self.gen_sizing.compute(inputs['r_s'], inputs['l_s'], inputs['h_s'], inputs['tau_p'], inputs['machine_rating'],
		inputs['n_nom'], inputs['Torque'], inputs['h_m'],inputs['h_ys'], inputs['h_yr'],inputs['rho_Fe'], inputs['rho_Copper'],inputs['b_st'], inputs['d_s'], \
		inputs['t_ws'], inputs['n_s'], inputs['t_d'],\
		inputs['R_o'], inputs['rho_Fes'],inputs['rho_PM'],inputs['main_shaft_cm'],inputs['main_shaft_length'])
		

		return outputs

class generator_sizing(object):
	
	def __init__(self):
		
		pass
		
	def	compute(self,r_s, l_s,h_s,tau_p,machine_rating,n_nom,Torque,h_m,h_ys,h_yr,
		rho_Fe,rho_Copper,b_st,d_s,t_ws,n_s,t_d,R_o,rho_Fes,rho_PM,main_shaft_cm,main_shaft_length):
			
		self.r_s=r_s
		self.l_s=l_s
		self.h_s=h_s
		self.tau_p=tau_p
		self.h_m=h_m
		self.h_ys=h_ys
		self.h_yr=h_yr
		self.machine_rating=machine_rating
		self.n_nom=n_nom
		self.Torque=Torque
		
		self.b_st=b_st
		self.d_s=d_s
		self.t_ws=t_ws
		self.n_s=n_s
		self.t_d=t_d
		
		self.R_o=R_o
		self.rho_Fe=rho_Fe
		self.rho_Copper=rho_Copper
		self.rho_Fes=rho_Fes
		self.rho_PM=rho_PM
		self.main_shaft_cm=main_shaft_cm
		self.main_shaft_length=main_shaft_length
		
		
		#Assign values to universal constants
		
		B_r    =1.2                 # Tesla remnant flux density
		g1      =9.81               # m/s^2 acceleration due to gravity
		E      =2e11                # N/m^2 young's modulus
		sigma  =40000.0                # shear stress assumed
		ratio  =0.7                 # ratio of magnet width to pole pitch(bm/self.tau_p)
		mu_0   =pi*4e-7             # permeability of free space
		mu_r   =1.06								# relative permeability
		phi    =90*2*pi/360         # tilt angle (rotor tilt -90 degrees during transportation)
		cofi   =0.85                # power factor
		
		#Assign values to design constants
		h_w    =0.005										# wedge height
		y_tau_p=1.0												# coiil span to pole pitch
		m      =3.0                    # no of phases
		q1     =1.0                    # no of slots per pole per phase
		b_s_tau_s=0.45							 # slot width/slot pitch ratio
		k_sfil =0.65								 # Slot fill factor
		P_Fe0h =4.0			               #specific hysteresis losses W/kg @ 1.5 T 
		P_Fe0e =1.0			               #specific hysteresis losses W/kg @ 1.5 T 
		rho_Cu=1.8*10**(-8)*1.4      # resistivity of copper
		b_so			=  0.004					 #stator slot opening
		k_fes =0.9									#useful iron stack length
		T =   self.Torque
		v=0.3                        # poisson's ratio
		
		# back iron thickness for rotor and stator
		self.t_s =self.h_ys    
		self.t =self.h_yr
		
		# Aspect ratio
		self.K_rad=self.l_s/(2*self.r_s) # aspect ratio
		
		###################################################### Electromagnetic design#############################################
		
		dia				=  2*self.r_s              # air gap diameter
		g         =  0.001*dia							 # air gap length
		self.b_m  =0.7*self.tau_p							# magnet width
		l_u       =k_fes * self.l_s                   #useful iron stack length
		We				=self.tau_p
		l_b       = 2*self.tau_p  										#end winding length
		l_e       =self.l_s+2*0.001*self.r_s     			# equivalent core length
		r_r				=  self.r_s-g                       #rotor radius
		self.p		=  round(pi*dia/(2*self.tau_p))			# pole pairs
		self.f    =  self.n_nom*self.p/60							# frequency
		self.S				= 2*self.p*q1*m 								# Stator slots
		N_conductors=self.S*2
		self.N_s=N_conductors/2/3											# Stator turns per phase
		tau_s=pi*dia/self.S														# slot pitch
		self.b_s	=  b_s_tau_s*tau_s    							#slot width
		self.b_t	=  tau_s-(self.b_s)          				#tooth width
		self.Slot_aspect_ratio=self.h_s/self.b_s
		alpha_p		=  pi/2*0.7
		
		# Calculating Carter factor for statorand effective air gap length
		gamma			=  4/pi*(b_so/2/(g+self.h_m/mu_r)*atan(b_so/2/(g+self.h_m/mu_r))-log(sqrt(1+(b_so/2/(g+self.h_m/mu_r))**2)))
		k_C				=  tau_s/(tau_s-gamma*(g+self.h_m/mu_r))   # carter coefficient
		g_eff			=  k_C*(g+self.h_m/mu_r)
		
		# angular frequency in radians
		om_m			=  2*pi*self.n_nom/60
		om_e			=  self.p*om_m/2
		
		# Calculating magnetic loading
		self.B_pm1	 		=  B_r*self.h_m/mu_r/(g_eff)
		self.B_g=  B_r*self.h_m/mu_r/(g_eff)*(4.0/pi)*sin(alpha_p)
		self.B_symax=self.B_g*self.b_m*l_e/(2*self.h_ys*l_u)
		self.B_rymax=self.B_g*self.b_m*l_e/(2*self.h_yr*self.l_s)
		self.B_tmax	=self.B_g*tau_s/self.b_t
		
		
		k_wd			= sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
		L_t=self.l_s+2*self.tau_p
		
		# Stator winding length ,cross-section and resistance
		l_Cus			= 2*(self.N_s)*(2*self.tau_p+L_t)
		A_s				= self.b_s*(self.h_s-h_w)*q1*self.p
		A_scalc   = self.b_s*1000*(self.h_s*1000-h_w*1000)*q1*self.p
		A_Cus			= A_s*k_sfil/(self.N_s)
		self.A_Cuscalc = A_scalc *k_sfil/(self.N_s)
		self.R_s	= l_Cus*rho_Cu/A_Cus
		
		# Calculating leakage inductance in  stator
		L_m				= 2*m*k_wd**2*(self.N_s)**2*mu_0*self.tau_p*L_t/pi**2/g_eff/self.p
		L_ssigmas=2*mu_0*self.l_s*self.N_s**2/self.p/q1*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
		L_ssigmaew=(2*mu_0*self.l_s*self.N_s**2/self.p/q1)*0.34*g*(l_e-0.64*self.tau_p*y_tau_p)/self.l_s #end winding leakage inductance
		L_ssigmag=2*mu_0*self.l_s*self.N_s**2/self.p/q1*(5*(g*k_C/b_so)/(5+4*(g*k_C/b_so))) # tooth tip leakage inductance#tooth tip leakage inductance
		L_ssigma	= (L_ssigmas+L_ssigmaew+L_ssigmag)
		self.L_s  = L_m+L_ssigma
		
		# Calculating no-load voltage induced in the stator and stator current
		self.E_p	= 2*(self.N_s)*L_t*self.r_s*k_wd*om_m*self.B_g/sqrt(2)
		
		Z=(self.machine_rating/(m*self.E_p))
		G=(self.E_p**2-(om_e*self.L_s*Z)**2)
		
		self.I_s= sqrt(Z**2+(((self.E_p-G**0.5)/(om_e*self.L_s)**2)**2))
		
		self.B_smax=sqrt(2)*self.I_s*mu_0/g_eff
		
		# Calculating stator current and electrical loading
		self.J_s	= self.I_s/self.A_Cuscalc
		I_snom		=(self.machine_rating/m/self.E_p/cofi) #rated current
		I_qnom		=self.machine_rating/(m*self.E_p)
		X_snom		=om_e*(L_m+L_ssigma)
		self.A_1 = 6*self.N_s*self.I_s/(pi*dia)
		
		#Calculating electromagnetically active mass
		
		V_Cus 	=m*l_Cus*A_Cus     # copper volume
		V_Fest	=L_t*2*self.p*q1*m*self.b_t*self.h_s   # volume of iron in stator tooth
		V_Fesy	=L_t*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2) # volume of iron in stator yoke
		V_Fery	=L_t*pi*((r_r-self.h_m)**2-(r_r-self.h_m-self.h_yr)**2) # volume of iron in rotor yoke
		self.Copper		=V_Cus*self.rho_Copper  
		M_Fest	=V_Fest*self.rho_Fe               # mass of stator tooth
		M_Fesy	=V_Fesy*self.rho_Fe								# mass of stator yoke
		M_Fery	=V_Fery*self.rho_Fe								# mass of rotor yoke
		self.Iron		=M_Fest+M_Fesy+M_Fery
		
		#Calculating losses"
		#1.Copper losses
		K_R=1.2                                   # Skin effect correction co-efficient
		P_Cu		=m*I_snom**2*self.R_s*K_R
		
		# Iron Losses ( from Hysteresis and eddy currents)
		P_Hyys	=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60)) # Hysteresis losses in stator yoke
		P_Ftys	=M_Fesy*((self.B_symax/1.5)**2)*(P_Fe0e*(om_e/(2*pi*60))**2)# Eddy losses in stator yoke
		P_Fesynom=P_Hyys+P_Ftys
		P_Hyd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60)) # Hysteresis losses in stator teeth
		P_Ftd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2) # Eddy losses in stator teeth
		P_Festnom=P_Hyd+P_Ftd
		P_ad=0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd ) # additional stray losses due to leakage flux
		pFtm =300 																	# specific magnet loss
		P_Ftm=pFtm*2*self.p*self.b_m*self.l_s					#magnet losses
		
		self.Losses=P_Cu+P_Festnom+P_Fesynom+P_ad+P_Ftm
		self.gen_eff=self.machine_rating*100/(self.machine_rating+self.Losses)
		
		################################################## Structural  Design ############################################################
		
		## Structural deflection calculations
		#rotor structure
		R					= self.r_s-g-self.h_m-0.5*self.t   # mean radius of the rotor rim
		l=L_t
		b					=self.R_o													 # Shaft radius
		R_b				=R-0.5*self.t											# Inner radius of the rotor
		R_a=R+0.5*self.h_yr													# Outer radius of rotor yoke
		a=R-0.5*self.t;
		a_1=R_b
		c					=R/500
		self.u_all_r    =c/20  																			# allowable radial deflection
		self.y_all    =2*l/100    																	# allowable axial deflection
		R_1				= R-self.t*0.5      															#inner radius of rotor cylinder
		K					=4*(sin(ratio*pi/2))/pi
		q3					= self.B_g**2/2/mu_0   													# normal component of Maxwell's stress
		
		self.mass_PM   =(2*pi*(R+0.5*self.t)*l*self.h_m*ratio*self.rho_PM)           # magnet mass
		mass_st_lam=self.rho_Fe*2*pi*R*l*self.h_yr                                     # mass of rotor yoke steel
		
		
		# Calculation of radial deflection of rotor
		# cylindrical shell function and circular plate parameters for disc rotor based on Table 11.2 Roark's formulas
		lamb   =((3*(1-v**2)/R_a**2/self.h_yr**2)**0.25)
		x1=lamb*l
		C_2=cosh(x1)*sin(x1)+sinh(x1)*cos(x1)
		C_3=sinh(x1)*sin(x1)
		C_4=cosh(x1)*sin(x1)-sinh(x1)*cos(x1)
		C_11=(sinh(x1))**2-(sin(x1))**2
		C_13=cosh(x1)*sinh(x1)-cos(x1)*sin(x1)
		C_14=(sinh(x1)**2+sin(x1)**2)
		C_a1=cosh(x1*0.5)*cos(x1*0.5)
		C_a2=cosh(x1*0.5)*sin(x1*0.5)+sinh(x1*0.5)*cos(x1*0.5)
		F_1_x0=cosh(lamb*0)*cos(lamb*0)
		F_1_ls2=cosh(lamb*0.5*self.l_s)*cos(lamb*0.5*self.l_s)
		F_2_x0=cosh(lamb*0)*sin(lamb*0)+sinh(lamb*0)*cos(lamb*0)
		F_2_ls2=cosh(x1/2)*sin(x1/2)+sinh(x1/2)*cos(x1/2)
		
		if (self.l_s<2*a):
			a=self.l_s/2
		else:
			a=self.l_s*0.5-1
			
		F_a4_x0=cosh(lamb*(0))*sin(lamb*(0))-sinh(lamb*(0))*cos(lamb*(0))
		F_a4_ls2=cosh(pi/180*lamb*(0.5*self.l_s-a))*sin(pi/180*lamb*(0.5*self.l_s-a))-sinh(pi/180*lamb*(0.5*self.l_s-a))*cos(pi/180*lamb*(0.5*self.l_s-a))
		
		D_r=E*self.h_yr**3/(12*(1-v**2))
		D_ax=E*self.t_d**3/(12*(1-v**2))
		
		# Radial deflection analytical model from McDonald's thesis defined in parts
		Part_1 =R_b*((1-v)*R_b**2+(1+v)*self.R_o**2)/(R_b**2-self.R_o**2)/E
		Part_2 =(C_2*C_a2-2*C_3*C_a1)/2/C_11
		Part_3 = (C_3*C_a2-C_4*C_a1)/C_11
		Part_4 =((0.25/D_r/lamb**3))
		Part_5=q3*R_b**2/(E*(R_a-R_b))
		f_d = Part_5/(Part_1-self.t_d*(Part_4*Part_2*F_2_ls2-Part_3*2*Part_4*F_1_ls2-Part_4*F_a4_ls2))
		fr=f_d*self.t_d
		self.u_Ar				=abs(Part_5+fr/2/D_r/lamb**3*((-F_1_x0/C_11)*(C_3*C_a2-C_4*C_a1)+(F_2_x0/2/C_11)*(C_2*C_a2-2*C_3*C_a1)-F_a4_x0/2))
		
		# Calculation of Axial deflection of rotor
		W=0.5*g1*sin(phi)*((l-self.t_d)*self.h_yr*self.rho_Fes)        # uniform annular line load acting on rotor cylinder assumed as an annular plate 
		w=self.rho_Fes*g1*sin(phi)*self.t_d	                           # disc assumed as plate with a uniformly distributed pressure between
		a_i=self.R_o
		
		# Flat circular plate constants according to Roark's table 11.2
		C_2p= 0.25*(1-(((self.R_o/R)**2)*(1+(2*log(R/self.R_o)))))
		C_3p=(self.R_o/4/R)*((1+(self.R_o/R)**2)*log(R/self.R_o)+(self.R_o/R)**2-1)
		C_6= (self.R_o/4/R_a)*((self.R_o/R_a)**2-1+2*log(R_a/self.R_o))
		C_5=0.5*(1-(self.R_o/R)**2)
		C_8= 0.5*(1+v+(1-v)*((self.R_o/R)**2))
		C_9=(self.R_o/R)*(0.5*(1+v)*log(R/self.R_o) + (1-v)/4*(1-(self.R_o/R)**2))
		
		# Flat circular plate loading constants
		L_11=(1 + 4*(self.R_o/a_1)**2 - 5*(self.R_o/a_1)**4 - 4*((self.R_o/a_1)**2)*log(a_1/self.R_o)*(2+(self.R_o/a_1)**2))/64
		L_14=(1/16)*(1-(self.R_o/R_b)**4-4*(self.R_o/R_b)**2*log(R_b/self.R_o))
		y_ai=-W*(a_1**3)*(C_2p*(C_6*a_1/self.R_o - C_6)/C_5 - a_1*C_3p/self.R_o +C_3p)/D_ax  # Axial deflection of plate due to The deflection of an annular plate with a uniform annular line load
		
		# Axial Deflection due to uniformaly distributed pressure load
		M_rb=-w*R**2*(C_6*(R**2-self.R_o**2)*0.5/R/self.R_o-L_14)/C_5
		Q_b=w*0.5*(R**2-self.R_o**2)/self.R_o
		y_aii=M_rb*R_a**2*C_2p/D_ax+Q_b*R_a**3*C_3p/D_ax-w*R_a**4*L_11/D_ax
		
		self.y_Ar=abs(y_ai+y_aii)
		
		self.z_all_r=0.05*2*pi*R/360  # allowable torsional deflection of rotor
		
				
		#stator structure deflection calculation
		
		self.R_out=(R/0.995+self.h_s+self.h_ys)
		
		a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)) # cross-sectional area of stator armms
		A_st      =l*self.t_s                     # cross-sectional area of rotor cylinder
		N_st			= round(self.n_s)
		theta_s		=pi*1/N_st  # half angle between spokes
		I_st       =l*self.t_s**3/12                      # second moment of area of stator cylinder
		I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
		I_arm_tor_s	= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
		R_st 			=self.r_s+self.h_s+self.h_ys*0.5
		k_2       = sqrt(I_st/A_st)  # radius of gyration
		self.b_all_s		=2*pi*self.R_o/N_st
		m2        =(k_2/R_st)**2
		c1        =R_st/500
		R_1s      = R_st-self.t_s*0.5
		d_se=dia+2*(self.h_ys+self.h_s+h_w)  # stator outer diameter
		
		# Calculation of radial deflection of stator
		Numers=R_st**3*((0.25*(sin(theta_s)-(theta_s*cos(theta_s)))/(sin(theta_s))**2)-(0.5/sin(theta_s))+(0.5/theta_s))
		Povs=((theta_s/(sin(theta_s))**2)+1/tan(theta_s))*((0.25*R_st/A_st)+(0.25*R_st**3/I_st))
		Qovs=R_st**3/(2*I_st*theta_s*(m2+1))
		Lovs=(R_1s-self.R_o)*0.5/a_s
		Denoms=I_st*(Povs-Qovs+Lovs)
		
		self.u_As				=(q3*R_st**2/E/self.t_s)*(1+Numers/Denoms)
		
		# Calculation of axial deflection of stator
		mass_st_lam_s= M_Fest+pi*l*self.rho_Fe*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2)
		W_is			=0.5*g1*sin(phi)*(self.rho_Fes*l*self.d_s**2)                          # length of stator arm beam at which self-weight acts
		W_iis     =g1*sin(phi)*(mass_st_lam_s+V_Cus*self.rho_Copper)/2/N_st							 # weight of stator cylinder and teeth
		w_s         =self.rho_Fes*g1*sin(phi)*a_s*N_st																	 # uniformly distributed load of the arms
		
		l_is      =R_st-self.R_o																													# distance at which the weight of the stator cylinder acts
		l_iis     =l_is																																		# distance at which the weight of the stator cylinder acts
		l_iiis    =l_is																																		# distance at which the weight of the stator cylinder acts
		self.u_all_s    = c1/20
		
		X_comp1 = (W_is*l_is**3/12/E/I_arm_axi_s)																				# deflection component due to stator arm beam at which self-weight acts
		X_comp2 =(W_iis*l_iis**4/24/E/I_arm_axi_s)																			# deflection component due to 1/nth of stator cylinder
		X_comp3 =w_s*l_iiis**4/24/E/I_arm_axi_s																					# deflection component due to weight of arms
		
		self.y_As       =X_comp1+X_comp2+X_comp3  # axial deflection
		
		# Stator circumferential deflection
		self.z_all_s     =0.05*2*pi*R_st/360  # allowable torsional deflection
		self.z_A_s  =2*pi*(R_st+0.5*self.t_s)*l/(2*N_st)*sigma*(l_is+0.5*self.t_s)**3/3/E/I_arm_tor_s

		mass_stru_steel  =2*(N_st*(R_1s-self.R_o)*a_s*self.rho_Fes)
		
		self.TC1=T*1.0/(2*pi*sigma)     # Torque/shear stress
		self.TC2=R**2*l              # Evaluating Torque constraint for rotor
		self.TC3=R_st**2*l           # Evaluating Torque constraint for stator
		self.Structural_mass=mass_stru_steel+(pi*(R**2-R_o**2)*self.t_d*self.rho_Fes)
		
		self.Mass =  self.Structural_mass+self.Iron+self.Copper+self.mass_PM
		
		self.I = np.array([0.0, 0.0, 0.0])
		
		# Calculating mass moments of inertia and center of mass
		self.I[0]   = (0.5*self.Mass*self.R_out**2)
		self.I[1]   = (0.25*self.Mass*self.R_out**2+(1/12)*self.Mass*self.l_s**2)
		self.I[2]   = self.I[1]
		
		self.cm = np.array([0.0, 0.0, 0.0])
		
		self.cm[0]  = self.main_shaft_cm[0] + self.main_shaft_length/2. + self.l_s/2
		self.cm[1]  = self.main_shaft_cm[1]
		self.cm[2]  = self.main_shaft_cm[2]
		
		
		return(self.B_symax, self.B_tmax, self.B_rymax,self.B_smax, self.B_pm1, self.B_g, self.N_s, self.b_s, \
		self.b_t, self.A_Cuscalc, self.b_m, self.p, self.E_p, self.f,self.I_s, self.R_s, self.L_s, self.A_1,\
		self.J_s,self.Mass,self.K_rad, self.Losses, self.gen_eff,self.u_Ar,self.y_Ar,\
		self.u_As,self.y_As,self.z_A_s,self.u_all_r,self.u_all_s,self.y_all,self.z_all_s,\
		self.z_all_r,self.b_all_s, \
		self.TC1,self.TC2,self.TC3,self.R_out,self.S,self.Slot_aspect_ratio,\
		self.cm,self.I,self.mass_PM,self.Copper,self.Iron,self.Structural_mass)

####################################################Cost Analysis#######################################################################

class PMSG_Cost(Component):
	
	""" Provides a material cost estimate for a PMSG _disc generator. Manufacturing costs are excluded"""
	
	def __init__(self):
		
		super(PMSG_Cost, self).__init__()
		
		# Inputs
		# Specific cost of material by type
		self.add_param('C_Cu',val=0.0, desc='Specific cost of copper')
		self.add_param('C_Fe',val=0.0,desc='Specific cost of magnetic steel/iron')
		self.add_param('C_Fes',val=0.0,desc='Specific cost of structural steel')
		self.add_param('C_PM',val=0.0,desc='Specific cost of Magnet')
		
		# Mass of each material type
		self.add_param('Copper',val=0.0, desc='Copper mass')
		self.add_param('Iron',val=0.0, desc='Iron mass')
		self.add_param('mass_PM' ,val=0.0, desc='Magnet mass')
		self.add_param('Structural_mass',val=0.0, desc='Structural mass')
		
		# Outputs
		
		self.add_output('Costs',val=0.0,desc='Total cost')
		
		self.gen_costs=generator_costing()
		
	def solve_nonlinear(self,inputs,outputs,resid):
		(outputs['Costs'])=self.gen_costs.compute(inputs['Copper'],inputs['C_Cu'], \
		inputs['Iron'],inputs['C_Fe'],inputs['C_Fes'],inputs['mass_PM'],inputs['C_PM'],inputs['Structural_mass'])
		return outputs
		
class generator_costing(object):
		
	def __init__(self):
		pass
		
	def	compute(self,Copper,C_Cu,Iron,C_Fe,C_Fes,mass_PM,C_PM,Structural_mass):
		self.Copper=Copper
		self.mass_PM=mass_PM
		self.Iron=Iron
		self.Structural_mass=Structural_mass
		
		# Material cost as a function of material mass and specific cost of material
		K_gen=self.Copper*C_Cu+self.Iron*C_Fe+C_PM*self.mass_PM
		Cost_str=C_Fes*self.Structural_mass
		Costs=K_gen+Cost_str

		return(Costs)
		
####################################################OPTIMISATION SET_UP ############################################################### 

class PMSG_disc_Opt(Group):
	
	""" Creates a new Group containing PMSG and an optimizer"""
	def __init__(self):
		super(PMSG_disc_Opt, self).__init__()
		
		self.add('machine_rating', IndepVarComp('machine_rating',0.0),promotes=['*'])
		self.add('Torque',IndepVarComp('Torque', 0.0),promotes=['*'])
		self.add('n_nom', IndepVarComp('n_nom', 0.0),promotes=['*'])
		
		self.add('main_shaft_cm', IndepVarComp('main_shaft_cm',np.array([0.0, 0.0, 0.0])),promotes=['*'])
		self.add('main_shaft_length',IndepVarComp('main_shaft_length',val=0.0),promotes=['*'])
		
		self.add('r_s',IndepVarComp('r_s',0.0),promotes=['*'])
		self.add('l_s',IndepVarComp('l_s',0.0),promotes=['*'])
		self.add('h_s',IndepVarComp('h_s',0.0),promotes=['*'])
		self.add('tau_p',IndepVarComp('tau_p',0.0),promotes=['*'])
		self.add('h_m' ,IndepVarComp('h_m',0.0),promotes=['*'])
		self.add('h_ys',IndepVarComp('h_ys',0.0),promotes=['*'])
		self.add('h_yr',IndepVarComp('h_yr',0.0),promotes=['*'])
		
		self.add('n_s',IndepVarComp('n_s',0.0),promotes=['*'])
		self.add('b_st',IndepVarComp('b_st',0.0),promotes=['*'])
		self.add('d_s',IndepVarComp('d_s',0.0),promotes=['*'])
		self.add('t_ws',IndepVarComp('t_ws',0.0),promotes=['*'])
		self.add('t_d',IndepVarComp('t_d',0.0),promotes=['*'])
		self.add('R_o',IndepVarComp('R_o',0.0),promotes=['*'])
		
		self.add('rho_Fes',IndepVarComp('rho_Fes',0.0),promotes=['*'])
		self.add('rho_Fe',IndepVarComp('rho_Fe',0.0),promotes=['*'])
		self.add('rho_Copper',IndepVarComp('rho_Copper',0.0),promotes=['*'])
		self.add('rho_PM',IndepVarComp('rho_PM',0.0),promotes=['*'])
		
		# add PMSG component, create constraint equations
		self.add('PMSG',PMSG(),promotes=['*'])
		self.add('con_uAs', ExecComp('con_uAs =u_all_s-u_As'),promotes=['*'])
		self.add('con_zAs', ExecComp('con_zAs =z_all_s-z_A_s'),promotes=['*'])
		self.add('con_yAs', ExecComp('con_yAs =y_all-y_As'),promotes=['*'])
		self.add('con_bst', ExecComp('con_bst =b_all_s-b_st'),promotes=['*'])
		self.add('con_uAr', ExecComp('con_uAr =u_all_r-u_Ar'),promotes=['*'])
		self.add('con_yAr', ExecComp('con_yAr =y_all-y_Ar'),promotes=['*'])
		self.add('con_TC2', ExecComp('con_TC2 =TC2-TC1'),promotes=['*'])
		self.add('con_TC3', ExecComp('con_TC3 =TC3-TC1'),promotes=['*'])
		self.add('con_Bsmax', ExecComp('con_Bsmax =B_g-B_smax'),promotes=['*'])
		
		# add PMSG_Cost component, connect i/o
		self.add('PMSG_Cost',PMSG_Cost(),promotes=['*'])
		self.add('C_Cu',IndepVarComp('C_Cu',val=0.0),promotes=['*'])
		self.add('C_Fe',IndepVarComp('C_Fe',val=0.0),promotes=['*'])
		self.add('C_Fes',IndepVarComp('C_Fes',val=0.0),promotes=['*'])
		self.add('C_PM',IndepVarComp('C_PM',val=0.0),promotes=['*'])
		
		#self.connect('PMSG.Copper','PMSG_Cost.Copper')
		
def PMSG_disc_Opt_example():
	
	#Example optimization of a PMSG_disc rotor generator for costs on a 5 MW reference turbine
	
	opt_problem = Problem(root=PMSG_disc_Opt())
	
	# add optimizer and set-up problem (using user defined input on objective function)
	opt_problem.driver=pyOptSparseDriver()
	opt_problem.driver.options['optimizer'] = 'CONMIN'
	#opt_problem.driver.add_objective('Costs')					# Define Objective
	opt_problem.driver.opt_settings['IPRINT'] = 4
	opt_problem.driver.opt_settings['ITRM'] = 3
	opt_problem.driver.opt_settings['ITMAX'] = 10
	opt_problem.driver.opt_settings['DELFUN'] = 1e-3
	opt_problem.driver.opt_settings['DABFUN'] = 1e-3
	opt_problem.driver.opt_settings['IFILE'] = 'CONMIN_PMSG_disc.out'
	opt_problem.root.deriv_options['type']='fd'
#	
	# Set bounds for design variables for a PMSG designed for a 5MW turbine
	
	opt_problem.driver.add_desvar('r_s',lower=0.5,upper=9.0)
	opt_problem.driver.add_desvar('l_s', lower=0.5, upper=2.5)
	opt_problem.driver.add_desvar('h_s', lower=0.04, upper=0.1)
	opt_problem.driver.add_desvar('tau_p', lower=0.04, upper=0.1)
	opt_problem.driver.add_desvar('h_m', lower=0.005, upper=0.10)
	opt_problem.driver.add_desvar('h_yr', lower=0.045, upper=0.25)
	opt_problem.driver.add_desvar('h_ys', lower=0.045, upper=0.25)
	opt_problem.driver.add_desvar('t_d', lower=0.1, upper=0.25)
	opt_problem.driver.add_desvar('n_s', lower=5., upper=15.)
	opt_problem.driver.add_desvar('b_st', lower=0.1, upper=1.5)
	opt_problem.driver.add_desvar('d_s', lower=0.1, upper=1.5)
	opt_problem.driver.add_desvar('t_ws', lower=0.001, upper=0.2)
##	
##	# Specificiency target efficiency(%)
	Eta_Target = 93.0	
#	
#	# set up constraints for the PMSG_arms generator
#	
	opt_problem.driver.add_constraint('B_symax',upper=2.0-1.0e-6)										#1
	opt_problem.driver.add_constraint('B_rymax',upper=2.0-1.0e-6)										#2
	opt_problem.driver.add_constraint('B_tmax',upper=2.0-1.0e-6)									  #3
	opt_problem.driver.add_constraint('B_g',lower=0.70,upper=1.20)  								#4
	opt_problem.driver.add_constraint('con_Bsmax',lower=0.0+1.0e-6) 								#5                
	opt_problem.driver.add_constraint('E_p',lower=500.0,upper=5000.0)								#6
	opt_problem.driver.add_constraint('con_uAs',lower=0.0+1.0e-6)										#7
	opt_problem.driver.add_constraint('con_zAs',lower=0.0+1.0e-6)										#8-
	opt_problem.driver.add_constraint('con_yAs',lower=0.0+1.0e-6)  									#9-
	opt_problem.driver.add_constraint('con_uAr',lower=0.0+1.0e-6)										#11
	opt_problem.driver.add_constraint('con_yAr',lower=0.0+1.0e-6) 									#12-
	opt_problem.driver.add_constraint('con_TC2',lower=0.0+1.0e-6)										#13
	opt_problem.driver.add_constraint('con_TC3',lower=0.0+1.0e-6)										#14
	opt_problem.driver.add_constraint('con_bst',lower=0.0+1.0e-6)											#10
	opt_problem.driver.add_constraint('A_1',upper=60000.0-1.0e-6)											#15
	opt_problem.driver.add_constraint('J_s',upper=6.0) 									   					#16-
	opt_problem.driver.add_constraint('A_Cuscalc',lower=5.0) 												#17
	opt_problem.driver.add_constraint('K_rad',lower=0.2+1.0e-6,upper=0.27)						#18
	opt_problem.driver.add_constraint('Slot_aspect_ratio',lower=4.0,upper=10.0)			#19
	opt_problem.driver.add_constraint('gen_eff',lower=Eta_Target)										#20
	
	#Define Objective
	
	opt_problem.driver.add_objective('Costs')
	
	opt_problem.setup()
	
	# Specify Target machine parameters
	opt_problem['machine_rating']=5000000.0
	opt_problem['Torque']=4.143289e6
	
	opt_problem['n_nom']=12.1

	
	# Initialize design variables
	opt_problem['r_s']= 3.49 #3.494618182
	opt_problem['l_s']= 1.5 #1.506103927
	opt_problem['h_s']= 0.06 #0.06034976
	opt_problem['tau_p'] =0.07 #0.07541515 
	opt_problem['h_m'] = 0.0105 #0.0090100202 
	opt_problem['h_ys'] =0.085 #0.084247994 #
	opt_problem['h_yr'] = 0.055 #0.0545789687
	opt_problem['n_s'] = 5.0 #5.0
	opt_problem['b_st'] = 0.460 #0.46381
	opt_problem['t_d'] = 0.105 #0.10 
	opt_problem['d_s']= 0.350 #0.35031 #
	opt_problem['t_ws']= 0.150 #=0.14720 #
	opt_problem['R_o'] =0.43 #0.43
	
	# Provide specific costs for materials
	opt_problem['C_Cu']   =4.786
	opt_problem['C_Fe']	= 0.556
	opt_problem['C_Fes'] =0.50139
	opt_problem['C_PM']  =95.0
	
	opt_problem['main_shaft_cm']=np.array([0.0, 0.0, 0.0])
	opt_problem['main_shaft_length'] =2.0
	
	# Provide Material properties
	
	opt_problem['rho_Fe'] = 7700.0                 #Steel density
	opt_problem['rho_Fes'] = 7850.0                #Steel density
	opt_problem['rho_Copper'] =8900.0                  # Kg/m3 copper density
	opt_problem['rho_PM'] =7450.0
	
	#Run optimization
	
	
	opt_problem.run()

	""" Uncomment to print solution to screen/an excel file
	

	
	raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor disc Thickness',' Stator Radial deflection', 'Stator Axial deflection','Stator circum deflection',' Rotor Radial deflection', 'Rotor Axial deflection','Air gap diameter',\
		'Overall Outer diameter', 'Stator length', 'l/d ratio','Slot_aspect_ratio','Pole pitch', 'Stator slot height','Stator slotwidth','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental','Peak stator yoke flux density','Peak rotor yoke flux density',\
		'Flux density above magnet','Maximum Stator flux density','Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage', 'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns','Conductor cross-section','Stator Current density ',\
		'Specific current loading','Generator Efficiency ','Iron mass','Magnet mass','Copper mass','Mass of Arms and disc', 'Total Mass', 'Total Material Cost'],
		'Values': [opt_problem['machine_rating']/1000000,opt_problem['n_s'],opt_problem['d_s']*1000,opt_problem['b_st']*1000,opt_problem['t_ws']*1000,opt_problem['t_d']*1000,opt_problem['u_As']*1000,opt_problem['y_As']*1000,opt_problem['z_A_s']*1000,opt_problem['u_Ar']*1000,opt_problem['y_Ar']*1000,2*opt_problem['r_s'],\
		opt_problem['R_out']*2,opt_problem['l_s'],opt_problem['K_rad'],opt_problem['Slot_aspect_ratio'],opt_problem['tau_p']*1000,opt_problem['h_s']*1000,opt_problem['b_s']*1000,opt_problem['b_t']*1000,opt_problem['h_ys']*1000,opt_problem['h_yr']*1000,opt_problem['h_m']*1000,opt_problem['b_m']*1000,opt_problem['B_g'],opt_problem['B_symax'],\
		opt_problem['B_rymax'],opt_problem['B_pm1'],opt_problem['B_smax'],opt_problem['B_tmax'],opt_problem['p'],opt_problem['f'],opt_problem['E_p'],opt_problem['I_s'],opt_problem['R_s'],opt_problem['L_s'],opt_problem['S'],opt_problem['N_s'],opt_problem['A_Cuscalc'],opt_problem['J_s'],opt_problem['A_1']/1000,opt_problem['gen_eff'],\
		opt_problem['Iron']/1000,opt_problem['mass_PM']/1000,opt_problem['Copper']/1000,opt_problem['Structural_mass']/1000,opt_problem['Mass']/1000,opt_problem['Costs']/1000],
		'Limit': ['','','',opt_problem['b_all_s']*1000,'','',opt_problem['u_all_s']*1000,opt_problem['y_all']*1000,opt_problem['z_all_s']*1000,opt_problem['u_all_r']*1000,opt_problem['y_all']*1000,'','','','(0.2-0.27)','(4-10)','','','','','','','','','<2','<2','<2','<2','<2',opt_problem['B_g'],'<2','','>500','','','','','','5',\
		'3-6','60','>93%','','','','','',''],
		'Units':['MW','unit','mm','mm','mm','','mm','mm','mm','mm','mm','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','slots','turns','mm^2','A/mm^2','kA/m','%','tons','tons','tons','tons','tons','k$']}
	df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print df
	df.to_excel('PMSG_'+str(opt_problem['machine_rating']/1e6)+'_discRotor_MW_1.7.x.xlsx')
	
	 """
		
if __name__=="__main__":
  
  # Run an example optimization of PMSG generator on cost
  PMSG_disc_Opt_example()
