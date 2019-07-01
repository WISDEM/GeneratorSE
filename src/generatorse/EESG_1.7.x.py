"""EESG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

from openmdao.api import Group, Problem, Component,ExecComp,IndepVarComp,ScipyOptimizer,pyOptSparseDriver
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.drivers import *
import numpy as np
from numpy import array,float,min,sign
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
import pandas



class EESG(Component):
	
	""" Estimates overall mass dimensions and Efficiency of Electrically Excited Synchronous generator. """
	
	def __init__(self):
		
		super(EESG, self).__init__()
		# EESG generator design inputs
		
		self.add_param('r_s', val=0.0, units ='m', desc='airgap radius r_s')
		self.add_param('l_s', val=0.0, units ='m', desc='Stator core length l_s')
		self.add_param('h_s', val=0.0, units ='m', desc='Yoke height h_s')
		self.add_param('tau_p',val=0.0, units ='m', desc='Pole pitch self.tau_p')
		
		self.add_param('machine_rating',val=0.0, units ='W', desc='Machine rating')
		self.add_param('n_nom',val=0.0, units ='rpm', desc='rated speed')
		self.add_param('Torque',val=0.0, units ='Nm', desc='Rated torque ')
		self.add_param('I_f',val=0.0000,units='A',desc='Excitation current')
		self.add_param('N_f',val=0.0,units='A',desc='field turns')
		self.add_param('h_ys',val=0.0, units ='m', desc='Yoke height')
		self.add_param('h_yr',val=0.0, units ='m', desc='rotor yoke height')
		
		# structural design variables
		self.add_param('n_s' ,val=0.0, desc='number of stator arms n_s')
		self.add_param('b_st' , val=0.0, units ='m', desc='arm width b_st')
		self.add_param('d_s',val=0.0,units ='m', desc='arm depth d_s')
		self.add_param('t_ws' ,val=0.0,units ='m', desc='arm depth thickness self.t_wr')
		self.add_param('n_r' ,val=0.0, desc='number of arms n')
		self.add_param('b_r' ,val=0.0,units ='m', desc='arm width b_r')
		self.add_param('d_r' ,val=0.0, units ='m', desc='arm depth d_r')
		self.add_param('t_wr' ,val=0.0, units ='m', desc='arm depth thickness self.t_wr')
		self.add_param('R_o',val=0.0, units ='m',desc='Shaft radius')
		
		# EESG generator design outputs
		
		# Magnetic loading
		self.add_output('B_symax' ,val=0.0, desc='Peak Stator Yoke flux density B_ymax')
		self.add_output('B_tmax',val=0.0, desc='Peak Teeth flux density')
		self.add_output('B_rymax',val=0.0, desc='Peak Rotor yoke flux density')
		self.add_output('B_gfm',val=0.0, desc='Average air gap flux density B_g')
		self.add_output('B_g' ,val=0.0, desc='Peak air gap flux density B_g')
		self.add_output('B_pc',val=0.0, desc='Pole core flux density')
		
		# Stator design
		self.add_output('N_s' ,val=0.0, desc='Number of turns in the stator winding')
		self.add_output('b_s',val=0.0, desc='slot width')
		self.add_output('b_t',val=0.0, desc='tooth width')
		self.add_output('A_Cuscalc',val=0.0, desc='Conductor cross-section mm^2')
		self.add_output('S',val=0.0, desc='Stator slots')
		
		# # Output parameters : Rotor design
		self.add_output('h_p',val=0.0, desc='Pole height')
		self.add_output('b_p',val=0.0, desc='Pole width')
		self.add_output('p',val=0.0, desc='No of pole pairs')
		self.add_output('n_brushes',val=0.0, desc='number of brushes')
		self.add_output('A_Curcalc',val=0.0, desc='Rotor Conductor cross-section')
		
		# Output parameters : Electrical performance
		self.add_output('E_s',val=0.0, desc='Stator phase voltage')
		self.add_output('f',val=0.0, desc='Generator output frequency')
		self.add_output('I_s',val=0.0, desc='Generator output phase current')
		self.add_output('R_s',val=0.0, desc='Stator resistance')
		self.add_output('R_r',val=0.0, desc='Rotor resistance')
		self.add_output('L_m',val=0.0, desc='Stator synchronising inductance')
		self.add_output('J_s',val=0.0, desc='Stator Current density')
		self.add_output('J_f',val=0.0, desc='rotor Current density')
		self.add_output('A_1',val=0.0, desc='Specific current loading')
		self.add_output('Load_mmf_ratio',val=0.0, desc='mmf_ratio')
		
		# Objective functions and output
		self.add_output('Mass',val=0.0, desc='Actual mass')
		self.add_output('K_rad',val=0.0, desc='K_rad')
		self.add_output('Losses',val=0.0, desc='Total loss')
		self.add_output('gen_eff',val=0.0, desc='Generator efficiency')
		
		# Structural performance
		self.add_output('u_Ar',val=0.0, desc='Rotor radial deflection')
		self.add_output('y_Ar',val=0.0, desc='Rotor axial deflection')
		self.add_output('z_A_r',val=0.0, desc='Rotor circumferential deflection')
		self.add_output('u_As',val=0.0, desc='Stator radial deflection')
		self.add_output('y_As',val=0.0, desc='Stator axial deflection')
		self.add_output('z_A_s',val=0.0, desc='Stator circumferential deflection')  
		self.add_output('u_all_r',val=0.0, desc='Allowable radial rotor')
		self.add_output('u_all_s',val=0.0, desc='Allowable radial stator')
		self.add_output('y_all',val=0.0, desc='Allowable axial')
		self.add_output('z_all_s',val=0.0, desc='Allowable circum stator')
		self.add_output('z_all_r',val=0.0, desc='Allowable circum rotor')
		self.add_output('b_all_s',val=0.0, desc='Allowable arm')
		self.add_output('b_all_r',val=0.0, desc='Allowable arm dimensions')
		self.add_output('TC1',val=0.0, desc='Torque constraint')
		self.add_output('TC2',val=0.0, desc='Torque constraint-rotor')
		self.add_output('TC3',val=0.0, desc='Torque constraint-stator')
		
		#Material properties
		self.add_param('rho_Fes',val=0.0,units='kg*m**-3', desc='Structural Steel density ')
		self.add_param('rho_Fe',val=0.0,units='kg*m**-3', desc='Magnetic Steel density ')
		self.add_param('rho_Copper',val=0.0,units='kg*m**-3', desc='Copper density ')
		
		# Mass Outputs
		self.add_output('Copper',val=0.0, desc='Copper Mass')
		self.add_output('Iron',val=0.0, desc='Electrical Steel Mass')
		self.add_output('Structural_mass'	,val=0.0, desc='Structural Mass')
		
		# Other parameters
		self.add_output('Power_ratio',val=0.0, desc='Power_ratio')
		self.add_output('Slot_aspect_ratio',val=0.0,desc='Stator slot aspect ratio')
		self.add_output('R_out',val=0.0, desc='Outer radius')
		
		#inputs/outputs for interface with drivese
		self.add_param('main_shaft_cm',val= np.array([0.0, 0.0, 0.0]),desc='Main Shaft CM')
		self.add_param('main_shaft_length',val=0.0, desc='main shaft length')
		self.add_output('I',val=np.array([0.0, 0.0, 0.0]),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
		self.add_output('cm', val=np.array([0.0, 0.0, 0.0]),desc='COM [x,y,z]')
		self.gen_sizing = generator_sizing()

	def solve_nonlinear(self, inputs, outputs, resid):
		(outputs['B_symax'], outputs['B_tmax'], outputs['B_rymax'], outputs['B_gfm'], outputs['B_g'],outputs['B_pc'], outputs['N_s'], outputs['b_s'], \
		outputs['b_t'], outputs['A_Cuscalc'],outputs['A_Curcalc'], outputs['b_p'], outputs['h_p'], outputs['p'], outputs['E_s'], outputs['f'], \
		outputs['I_s'], outputs['R_s'], outputs['L_m'], outputs['A_1'], outputs['J_s'], outputs['R_r'],outputs['Losses'], \
		outputs['Load_mmf_ratio'],outputs['Power_ratio'],outputs['n_brushes'],outputs['J_f'],outputs['K_rad'], outputs['gen_eff'], outputs['S'], 
		outputs['Slot_aspect_ratio'], outputs['Copper'],outputs['Iron'],outputs['u_Ar'], outputs['y_Ar'], \
		outputs['z_A_r'], outputs['u_As'], outputs['y_As'], outputs['z_A_s'], outputs['u_all_r'], outputs['u_all_s'], \
		outputs['y_all'], outputs['z_all_s'], outputs['z_all_r'], outputs['b_all_s'], outputs['b_all_r'], outputs['TC1'], \
		outputs['TC2'], outputs['TC3'], outputs['R_out'], outputs['Structural_mass'],outputs['Mass'],outputs['cm'], outputs['I']) \
		= self.gen_sizing.compute(inputs['r_s'], inputs['l_s'], inputs['h_s'], inputs['tau_p'], inputs['machine_rating'], 
		inputs['n_nom'], inputs['Torque'], inputs['I_f'],inputs['N_f'],inputs['h_ys'], inputs['h_yr'],inputs['rho_Fe'], inputs['rho_Copper'],inputs['b_st'], inputs['d_s'], \
		inputs['t_ws'], inputs['n_r'],inputs['n_s'], inputs['b_r'],inputs['d_r'], inputs['t_wr'], \
		inputs['R_o'], inputs['rho_Fes'],inputs['main_shaft_cm'],inputs['main_shaft_length'])

		return outputs


class generator_sizing(object):
	
	def __init__(self):
		pass
		
	def	compute(self,r_s, l_s,h_s,tau_p,machine_rating,n_nom,Torque,I_f,N_f,h_ys,h_yr, \
	rho_Fe,rho_Copper,b_st, d_s,t_ws, n_r,n_s, b_r,d_r, t_wr, \
	R_o, rho_Fes,main_shaft_cm,main_shaft_length):
		
		self.r_s=r_s
		self.l_s=l_s
		self.h_s=h_s
		self.tau_p=tau_p
		self.N_f=N_f
		self.I_f=I_f
		self.h_ys=h_ys
		self.h_yr=h_yr
		self.machine_rating=machine_rating
		self.n_nom=n_nom
		self.Torque=Torque
		
		self.b_st=b_st
		self.d_s=d_s
		self.t_ws=t_ws
		self.n_r=n_r
		self.n_s=n_s
		self.b_r=b_r
		self.d_r=d_r
		self.t_wr=t_wr
		
		self.R_o=R_o
		self.rho_Fe=rho_Fe
		self.rho_Copper=rho_Copper
		self.rho_Fes=rho_Fes
		self.main_shaft_cm=main_shaft_cm
		self.main_shaft_length=main_shaft_length
		
		#Assign values to universal constants
		g1     =9.81                # m/s^2 acceleration due to gravity
		E      =2e11                # N/m^2 young's modulus
		sigma  =48.373e3                # shear stress
		mu_0   =pi*4e-7        # permeability of free space
		phi    =90*2*pi/360
		
		#Assign values to design constants
		h_w    =0.005
		b_so	=  0.004							# Stator slot opening
		m      =3										# number of phases
		q1     =2										# no of stator slots per pole per phase
		b_s_tau_s=0.45							# ratio of slot width to slot pitch
		k_sfil =0.65								# Slot fill factor
		P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
		P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
		rho_Cu=1.8*10**(-8)*1.4			# resisitivity of copper
		k_fes =0.9								# iron fill factor
		y_tau_p=1                  # coil span/pole pitch fullpitch
		k_fillr = 0.7									# rotor slot fill factor
		k_s=0.2 														#magnetic saturation factor for iron
		T = self.Torque
		cos_phi=0.85					#power factor
		
		# back iron thickness for rotor and stator
		self.t_s =self.h_ys
		self.t =self.h_yr
		
		# Aspect ratio
		self.K_rad=self.l_s/(2*self.r_s)
		
		###################################################### Electromagnetic design#############################################
		
		alpha_p=pi/2*.7
		dia=2*self.r_s             # air gap diameter
		
		# air gap length and minimum values
		g=0.001*dia
		
		if(g<0.005):
			g=0.005
			
		r_r=self.r_s-g             						#rotor radius
		d_se=dia+2*self.h_s+2*self.h_ys  			# stator outer diameter
		self.p=round(pi*dia/(2*self.tau_p))  	# number of pole pairs
		self.S=2*self.p*q1*m   								# number of slots of stator phase winding
		N_conductors=self.S*2
		self.N_s=N_conductors/2/3							# Stator turns per phase
		alpha =180/self.S/self.p    #electrical angle
		
		tau_s=pi*dia/self.S										# slot pitch
		h_ps=0.1*self.tau_p										# height of pole shoe
		b_pc=0.4*self.tau_p										# width of pole core
		h_pc=0.6*self.tau_p										# height of pole core
		self.h_p=0.7*tau_p										# pole height
		self.b_p=self.h_p
		self.b_s=tau_s * b_s_tau_s        		#slot width
		self.Slot_aspect_ratio=self.h_s/self.b_s
		self.b_t=tau_s-self.b_s              #tooth width
		
		# Calculating carter factor and effective air gap
		g_a=g
		K_C1=(tau_s+10*g_a)/(tau_s-self.b_s+10*g_a)  # salient pole rotor
		g_1=K_C1*g
		
		# calculating angular frequency
		om_m=2*pi*self.n_nom/60
		om_e=60
		self.f    =  self.n_nom*self.p/60
		
		# Slot fill factor according to air gap radius
		
		if (2*self.r_s>2):
			K_fills=0.65
		else:
			K_fills=0.4
			
		# Calculating Stator winding factor	
		
		k_y1=sin(y_tau_p*pi/2)							# chording factor
		k_q1=sin(pi/6)/q1/sin(pi/6/q1)			# winding zone factor
		k_wd=k_y1*k_q1
		
		# Calculating stator winding conductor length, cross-section and resistance
		
		shortpitch=0
		l_Cus = 2*self.N_s*(2*(self.tau_p-shortpitch/m/q1)+self.l_s)   #length of winding
		A_s = self.b_s*(self.h_s-h_w)
		A_scalc=self.b_s*1000*(self.h_s*1000-h_w*1000)								# cross section in mm^2
		A_Cus = A_s*q1*self.p*K_fills/self.N_s
		self.A_Cuscalc = A_scalc*q1*self.p*K_fills/self.N_s
		self.R_s=l_Cus*rho_Cu/A_Cus
		
		#field winding design, conductor lenght, cross-section and resistance
		
		self.N_f=round(self.N_f)            # rounding the field winding turns to the nearest integer
		I_srated=self.machine_rating/(sqrt(3)*5000*cos_phi)
		l_pole=self.l_s-0.05+0.120  # 50mm smaller than stator and 120mm longer to accommodate end stack
		K_fe=0.95                    
		l_pfe=l_pole*K_fe
		l_Cur=4*self.p*self.N_f*(l_pfe+b_pc+pi/4*(pi*(r_r-h_pc-h_ps)/self.p-b_pc))
		A_Cur=k_fillr*h_pc*0.5/self.N_f*(pi*(r_r-h_pc-h_ps)/self.p-b_pc)
		self.A_Curcalc=k_fillr*h_pc*1000*0.5/self.N_f*(pi*(r_r-h_pc-h_ps)*1000/self.p-b_pc*1000)
		Slot_Area=A_Cur*2*self.N_f/k_fillr
		self.R_r=rho_Cu*l_Cur/A_Cur
		
		#field winding current density
		
		self.J_f=self.I_f/self.A_Curcalc
		
		# calculating air flux density
		
		self.B_gfm=mu_0*self.N_f*self.I_f/(g_1*(1+k_s))  #No load air gap flux density
		
		self.B_g=self.B_gfm*4*sin(0.5*self.b_p*pi/self.tau_p)/pi  # fundamental component
		self.B_symax=self.tau_p*self.B_g/pi/self.h_ys #stator yoke flux density
		L_fg=2*mu_0*self.p*self.l_s*4*self.N_f**2*((h_ps/(self.tau_p-self.b_p))+(h_pc/(3*pi*(r_r-h_pc-h_ps)/self.p-b_pc)))
		
		# calculating no load voltage and stator current
		
		self.E_s=2*self.N_s*self.l_s*self.r_s*k_wd*om_m*self.B_g/sqrt(2) #no load voltage
		self.I_s=(self.E_s-(self.E_s**2-4*self.R_s*self.machine_rating/m)**0.5)/(2*self.R_s)
		
		# Calculating stator winding current density and specific current loading
		
		self.A_1 = 6*self.N_s*self.I_s/(pi*dia)
		self.J_s=self.I_s/self.A_Cuscalc
		
		# Calculating magnetic loading in other parts of the machine
		
		delta_m=0  # Initialising load angle
		
		# peak flux density in pole core, rotor yoke and stator teeth
		
		self.B_pc=(1/b_pc)*((2*self.tau_p/pi)*self.B_g*cos(delta_m)+(2*mu_0*self.I_f*self.N_f*((2*h_ps/(self.tau_p-self.b_p))+(h_pc/(self.tau_p-b_pc)))))
		self.B_rymax= 0.5*b_pc*self.B_pc/self.h_yr
		self.B_tmax=(self.B_gfm+self.B_g)*tau_s*0.5/self.b_t
		
		# Calculating leakage inductances in the stator
		
		L_ssigmas=2*mu_0*self.l_s*self.N_s**2/self.p/q1*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
		L_ssigmaew=mu_0*1.2*self.N_s**2/self.p*1.2*(2/3*self.tau_p+0.01)                    #end winding leakage inductance
		L_ssigmag=2*mu_0*self.l_s*self.N_s**2/self.p/q1*(5*(g/b_so)/(5+4*(g/b_so))) # tooth tip leakage inductance
		L_ssigma=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
		
		# Calculating effective air gap
		
		At_g=g_1*self.B_gfm/mu_0
		At_t=self.h_s*(400*self.B_tmax+7*(self.B_tmax)**13)
		At_sy=self.tau_p*0.5*(400*self.B_symax+7*(self.B_symax)**13)
		At_pc=(h_pc+h_ps)*(400*self.B_pc+7*(self.B_pc)**13)
		At_ry=self.tau_p*0.5*(400*self.B_rymax+7*(self.B_rymax)**13)
		g_eff = (At_g+At_t+At_sy+At_pc+At_ry)*g_1/At_g
		
		self.L_m = 6*k_wd**2*self.N_s**2*mu_0*self.r_s*self.l_s/pi/g_eff/self.p**2
		B_r1=(mu_0*self.I_f*self.N_f*4*sin(0.5*(self.b_p/self.tau_p)*pi))/g_eff/pi
		
		# Calculating direct axis and quadrature axes inductances
		L_dm= (self.b_p/self.tau_p +(1/pi)*sin(pi*self.b_p/self.tau_p))*self.L_m
		L_qm=(self.b_p/self.tau_p -(1/pi)*sin(pi*self.b_p/self.tau_p)+2/(3*pi)*cos(self.b_p*pi/2*self.tau_p))*self.L_m
		
		# Calculating actual load angle
		
		delta_m=(atan(om_e*L_qm*self.I_s/self.E_s))
		L_d=L_dm+L_ssigma
		L_q=L_qm+L_ssigma
		I_sd=self.I_s*sin(delta_m)
		I_sq=self.I_s*cos(delta_m)
		
		# induced voltage
		
		E_p=om_e*L_dm*I_sd+sqrt(self.E_s**2-(om_e*L_qm*I_sq)**2)
		#M_sf =mu_0*8*self.r_s*self.l_s*k_wd*self.N_s*self.N_f*sin(0.5*self.b_p/self.tau_p*pi)/(self.p*g_eff*pi)
		#I_f1=sqrt(2)*(E_p)/(om_e*M_sf)
		#I_f2=(E_p/self.E_s)*self.B_g*g_eff*pi/(4*self.N_f*mu_0*sin(pi*self.b_p/2/self.tau_p))
		#phi_max_stator=k_wd*self.N_s*pi*self.r_s*self.l_s*2*mu_0*self.N_f*self.I_f*4*sin(0.5*self.b_p/self.tau_p/pi)/(self.p*pi*g_eff*pi)
		#M_sf=mu_0*8*self.r_s*self.l_s*k_wd*self.N_s*self.N_f*sin(0.5*b_p/self.tau_p/pi)/(self.p*g_eff*pi)
		
		L_tot=self.l_s+2*self.tau_p
		
		# Excitation power
		V_fn=500
		Power_excitation=V_fn*2*self.I_f   #total rated power in excitation winding
		self.Power_ratio =Power_excitation*100/self.machine_rating
		
		# Calculating Electromagnetically Active mass
		L_tot=self.l_s+2*self.tau_p
		V_Cuss=m*l_Cus*A_Cus             # volume of copper in stator
		V_Cusr=l_Cur*A_Cur							# volume of copper in rotor
		V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-2*m*q1*self.p*self.b_s*self.h_s*self.l_s) # volume of iron in stator tooth
		V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2) # volume of iron in stator yoke
		V_Fert=2*self.p*l_pfe*(h_pc*b_pc+self.b_p*h_ps)																# volume of iron in rotor pole
		V_Fery=l_pfe*pi*((r_r-h_ps-h_pc)**2-(r_r-h_ps-h_pc-self.h_yr)**2)							# # volume of iron in rotor yoke
		
		self.Copper=(V_Cuss+V_Cusr)*self.rho_Copper
		M_Fest=V_Fest*self.rho_Fe
		M_Fesy=V_Fesy*self.rho_Fe
		M_Fert=V_Fert*self.rho_Fe
		M_Fery=V_Fery*self.rho_Fe
		self.Iron=M_Fest+M_Fesy+M_Fert+M_Fery
		
		I_snom=self.machine_rating/(3*self.E_s*cos_phi)
		
		## Optional## Calculating mmf ratio
		F_1no_load=3*2**0.5*self.N_s*k_wd*self.I_s/(pi*self.p)
		Nf_If_no_load=self.N_f*self.I_f
		F_1_rated=(3*2**0.5*self.N_s*k_wd*I_srated)/(pi*self.p)
		Nf_If_rated=2*Nf_If_no_load
		self.Load_mmf_ratio=Nf_If_rated/F_1_rated
		
		## Calculating losses
		#1. Copper losses
		K_R=1.2
		P_Cuss=m*I_snom**2*self.R_s*K_R
		P_Cusr=self.I_f**2*self.R_r
		P_Cusnom_total=P_Cuss+P_Cusr
		
		#2. Iron losses ( Hysteresis and Eddy currents)
		P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60)) # Hysteresis losses in stator yoke
		P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2) # Eddy losses in stator yoke
		P_Fesynom=P_Hyys+P_Ftys
		P_Hyd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))    # Hysteresis losses in stator teeth
		P_Ftd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2) # Eddy losses in stator teeth
		P_Festnom=P_Hyd+P_Ftd
		
		# brushes
		delta_v=1
		self.n_brushes=(self.I_f*2/120)
		
		if (self.n_brushes<0.5):
			self.n_brushes=1
		else:
			self.n_brushes=round(self.n_brushes)
			
		#3. brush losses
		
		p_b=2*delta_v*(self.I_f)
		self.Losses=P_Cusnom_total+P_Festnom+P_Fesynom+p_b
		self.gen_eff=self.machine_rating*100/(self.Losses+self.machine_rating)
		
		################################################## Structural  Design ########################################################
		
		## Structural deflection calculations
		
		#rotor structure
		
		q3				= self.B_g**2/2/mu_0   # normal component of Maxwell's stress
		l					= self.l_s                          #l-stator core length
		l_b       = 2*self.tau_p  							#end winding length
		l_e       =self.l_s+2*0.001*self.r_s     # equivalent core length
		a_r				= (self.b_r*self.d_r)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr))  # cross-sectional area of rotor armms
		A_r				= l*self.t                     # cross-sectional area of rotor cylinder
		N_r				= round(self.n_r)
		theta_r		=pi/N_r                             # half angle between spokes
		I_r				=l*self.t**3/12                         # second moment of area of rotor cylinder
		I_arm_axi_r	=((self.b_r*self.d_r**3)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr)**3))/12  # second moment of area of rotor arm
		I_arm_tor_r	= ((self.d_r*self.b_r**3)-((self.d_r-2*self.t_wr)*(self.b_r-2*self.t_wr)**3))/12  # second moment of area of rotot arm w.r.t torsion
		R					= r_r-h_ps-h_pc-0.5*self.h_yr
		R_1				= R-self.h_yr*0.5                               # inner radius of rotor cylinder
		k_1				= sqrt(I_r/A_r)                               # radius of gyration
		m1				=(k_1/R)**2 
		c					=R/500
		
		self.u_all_r     =R/10000 # allowable radial deflection 
		self.b_all_r			=2*pi*self.R_o/N_r  # allowable circumferential arm dimension 
		
		# Calculating radial deflection of rotor structure according to Mc Donald's
		Numer=R**3*((0.25*(sin(theta_r)-(theta_r*cos(theta_r)))/(sin(theta_r))**2)-(0.5/sin(theta_r))+(0.5/theta_r))
		Pov=((theta_r/(sin(theta_r))**2)+1/tan(theta_r))*((0.25*R/A_r)+(0.25*R**3/I_r))
		Qov=R**3/(2*I_r*theta_r*(m1+1))
		Lov=(R_1-R_o)/a_r
		Denom=I_r*(Pov-Qov+Lov) # radial deflection % rotor
		self.u_Ar				=(q3*R**2/E/self.h_yr)*(1+Numer/Denom)
		
		# Calculating axial deflection of rotor structure
		
		w_r					=self.rho_Fes*g1*sin(phi)*a_r*N_r
		mass_st_lam=self.rho_Fe*2*pi*(R+0.5*self.h_yr)*l*self.h_yr                                    # mass of rotor yoke steel
		W				=g1*sin(phi)*(mass_st_lam+(V_Cusr*self.rho_Copper)+M_Fert)/N_r  # weight of rotor cylinder
		l_ir			=R                                      # length of rotor arm beam at which rotor cylinder acts
		l_iir			=R_1
		
		self.y_Ar       =(W*l_ir**3/12/E/I_arm_axi_r)+(w_r*l_iir**4/24/E/I_arm_axi_r)  # axial deflection
		
		#Calculating torsional deflection of rotor structure
		
		self.z_all_r     =0.05*2*pi*R/360  # allowable torsional deflection
		self.z_A_r       =(2*pi*(R-0.5*self.h_yr)*l/N_r)*sigma*(l_ir-0.5*self.h_yr)**3/3/E/I_arm_tor_r       # circumferential deflection
		
		#STATOR structure
		
		A_st      =l*self.t_s
		a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws))
		N_st			= round(self.n_s)
		theta_s		=pi/N_st 
		I_st      =l*self.t_s**3/12
		I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
		I_arm_tor_s	= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
		R_st 			=(self.r_s+self.h_s+self.h_ys*0.5)
		R_1s      = R_st-self.h_ys*0.5
		k_2       = sqrt(I_st/A_st)
		m2        =(k_2/R_st)**2
		
		# allowable deflections
		
		self.b_all_s			=2*pi*self.R_o/N_st
		self.u_all_s    = R_st/10000
		self.y_all     =2*l/100    # allowable axial deflection
		self.z_all_s     =0.05*2*pi*R_st/360  # allowable torsional deflection
		
		# Calculating radial deflection according to McDonald's
		
		Numers=R_st**3*((0.25*(sin(theta_s)-(theta_s*cos(theta_s)))/(sin(theta_s))**2)-(0.5/sin(theta_s))+(0.5/theta_s))
		Povs=((theta_s/(sin(theta_s))**2)+1/tan(theta_s))*((0.25*R_st/A_st)+(0.25*R_st**3/I_st))
		Qovs=R_st**3/(2*I_st*theta_s*(m2+1))
		Lovs=(R_1s-R_o)*0.5/a_s
		Denoms=I_st*(Povs-Qovs+Lovs)
		self.R_out=(R/0.995+self.h_s+self.h_ys)
		self.u_As				=(q3*R_st**2/E/self.t_s)*(1+Numers/Denoms)
		
		# Calculating axial deflection according to McDonald
		
		l_is      =R_st-self.R_o
		l_iis     =l_is
		l_iiis    =l_is
		mass_st_lam_s= M_Fest+pi*l*self.rho_Fe*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2)
		W_is			=g1*sin(phi)*(self.rho_Fes*l*self.d_s**2*0.5) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
		W_iis     =g1*sin(phi)*(V_Cuss*self.rho_Copper+mass_st_lam_s)/2/N_st
		w_s         =self.rho_Fes*g1*sin(phi)*a_s*N_st
		
		X_comp1 = (W_is*l_is**3/12/E/I_arm_axi_s)
		X_comp2 =(W_iis*l_iis**4/24/E/I_arm_axi_s)
		X_comp3 =w_s*l_iiis**4/24/E/I_arm_axi_s
		
		self.y_As       =X_comp1+X_comp2+X_comp3  # axial deflection
		
		# Calculating torsional deflection
		
		self.z_A_s  =2*pi*(R_st+0.5*self.t_s)*l/(2*N_st)*sigma*(l_is+0.5*self.t_s)**3/3/E/I_arm_tor_s 
		
		# tangential stress constraints
		
		self.TC1=T/(2*pi*sigma)
		self.TC2=R**2*l
		self.TC3=R_st**2*l
		
		mass_stru_steel  =2*(N_st*(R_1s-self.R_o)*a_s*self.rho_Fes)
		
		# Calculating inactive mass and total mass
		
		self.Structural_mass=mass_stru_steel+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes)
		
		self.Mass=self.Copper+self.Iron+self.Structural_mass
		
		self.I = np.array([0.0, 0.0, 0.0])
		# Calculating mass moments of inertia and center of mass
		self.I[0]   = (0.5*self.Mass*self.R_out**2)
		self.I[1]   = (0.25*self.Mass*self.R_out**2+(1/12)*self.Mass*self.l_s**2) 
		self.I[2]   = self.I[1]
		self.cm = np.array([0.0, 0.0, 0.0])
		self.cm[0]  = self.main_shaft_cm[0] + self.main_shaft_length/2. + self.l_s/2
		self.cm[1]  = self.main_shaft_cm[1]
		self.cm[2]  = self.main_shaft_cm[2]

		
		return(self.B_symax, self.B_tmax, self.B_rymax,self.B_gfm, self.B_g,self.B_pc, self.N_s, self.b_s, \
		self.b_t, self.A_Cuscalc,  self.A_Curcalc,self.b_p,self.h_p, self.p, self.E_s, self.f,self.I_s, self.R_s, self.L_m, self.A_1,\
		self.J_s,self.R_r, self.Losses,self.Load_mmf_ratio,self.Power_ratio,self.n_brushes,self.J_f,self.K_rad, self.gen_eff,\
		self.S, self.Slot_aspect_ratio, self.Copper,self.Iron,self.u_Ar,self.y_Ar,self.z_A_r,\
		self.u_As,self.y_As,self.z_A_s,self.u_all_r,self.u_all_s,self.y_all,self.z_all_s,self.z_all_r,self.b_all_s, \
		self.b_all_r,self.TC1,self.TC2,self.TC3,self.R_out,self.Structural_mass,self.Mass,self.cm,self.I)
  
####################################################Cost Analysis#######################################################################

class EESG_Cost(Component):
	""" Provides a material cost estimate for EESG. Manufacturing costs are excluded"""
	
	def __init__(self):
		
		super(EESG_Cost, self).__init__()
		
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
		K_gen=self.Copper*C_Cu+self.Iron*C_Fe
		Cost_str=C_Fes*self.Structural_mass
		Costs=K_gen+Cost_str
		return(Costs)


####################################################OPTIMISATION SET_UP ###############################################################

  
class EESG_Opt(Group):
	
	""" Creates a new Group containing EESG and EESG_Cost"""
		
	def __init__(self):
		super(EESG_Opt, self).__init__()
		
		self.add('machine_rating', IndepVarComp('machine_rating',0.0),promotes=['*'])
		self.add('Torque',IndepVarComp('Torque', val=0.0),promotes=['*'])
		self.add('n_nom', IndepVarComp('n_nom', val=0.0),promotes=['*'])
		
		self.add('main_shaft_cm', IndepVarComp('main_shaft_cm',val=np.array([0.0, 0.0, 0.0])),promotes=['*'])
		self.add('main_shaft_length',IndepVarComp('main_shaft_length',val=0.0),promotes=['*'])
		
		self.add('r_s',IndepVarComp('r_s',0.0),promotes=['*'])
		self.add('l_s',IndepVarComp('l_s',0.0),promotes=['*'])
		self.add('h_s',IndepVarComp('h_s',0.0),promotes=['*'])
		self.add('tau_p',IndepVarComp('tau_p',0.0),promotes=['*'])
		self.add('I_f',IndepVarComp('I_f',0.0),promotes=['*'])
		self.add('N_f',IndepVarComp('N_f',0.0),promotes=['*'])
		
		self.add('h_ys',IndepVarComp('h_ys',0.0),promotes=['*'])
		self.add('h_yr',IndepVarComp('h_yr',0.0),promotes=['*'])
		self.add('n_s',IndepVarComp('n_s',0.0),promotes=['*'])
		self.add('b_st',IndepVarComp('b_st',0.0),promotes=['*'])
		self.add('n_r',IndepVarComp('n_r',0.0),promotes=['*'])
		self.add('b_r',IndepVarComp('b_r',0.0),promotes=['*'])
		self.add('d_r',IndepVarComp('d_r',0.0),promotes=['*'])
		self.add('d_s',IndepVarComp('d_s',0.0),promotes=['*'])
		self.add('t_wr',IndepVarComp('t_wr',0.0),promotes=['*'])
		self.add('t_ws',IndepVarComp('t_ws',0.0),promotes=['*'])
		self.add('R_o',IndepVarComp('R_o',0.0),promotes=['*'])
		
		self.add('rho_Fes',IndepVarComp('rho_Fes',0.0),promotes=['*'])
		self.add('rho_Fe',IndepVarComp('rho_Fe',0.0),promotes=['*'])
		self.add('rho_Copper',IndepVarComp('rho_Copper',0.0),promotes=['*'])
		
		# add EESG component, create constraint equations
		
		self.add('EESG',EESG(),promotes=['*'])
		self.add('con_uAs', ExecComp('con_uAs =u_all_s-u_As'),promotes=['*'])
		self.add('con_zAs', ExecComp('con_zAs =z_all_s-z_A_s'),promotes=['*'])
		self.add('con_yAs', ExecComp('con_yAs =y_all-y_As'),promotes=['*'])
		self.add('con_bst', ExecComp('con_bst =b_all_s-b_st'),promotes=['*'])
		self.add('con_uAr', ExecComp('con_uAr =u_all_r-u_Ar'),promotes=['*'])
		self.add('con_zAr', ExecComp('con_zAr =z_all_r-z_A_r'),promotes=['*'])
		self.add('con_yAr', ExecComp('con_yAr =y_all-y_Ar'),promotes=['*'])
		self.add('con_br', ExecComp('con_br =b_all_r-b_r'),promotes=['*'])
		self.add('con_TC2', ExecComp('con_TC2 =TC2-TC1'),promotes=['*'])
		self.add('con_TC3', ExecComp('con_TC3 =TC3-TC1'),promotes=['*'])
		
		# add EESG_Cost component
		self.add('EESG_Cost',EESG_Cost(),promotes=['*'])
		self.add('C_Cu',IndepVarComp('C_Cu',val=0.0),promotes=['*'])
		self.add('C_Fe',IndepVarComp('C_Fe',val=0.0),promotes=['*'])
		self.add('C_Fes',IndepVarComp('C_Fes',val=0.0),promotes=['*'])
		

def EESG_Opt_example():
	opt_problem=Problem(root=EESG_Opt())
	
	#Example optimization of an EESG for costs on a 5 MW reference turbine
	
	# add optimizer and set-up problem (using user defined input on objective function)
#	
	opt_problem.driver=pyOptSparseDriver()
	opt_problem.driver.options['optimizer'] = 'CONMIN'
	opt_problem.driver.add_objective('Costs')					# Define Objective
	opt_problem.driver.opt_settings['IPRINT'] = 4
	opt_problem.driver.opt_settings['ITRM'] = 3
	opt_problem.driver.opt_settings['ITMAX'] = 10
	opt_problem.driver.opt_settings['DELFUN'] = 1e-3
	opt_problem.driver.opt_settings['DABFUN'] = 1e-3
	opt_problem.driver.opt_settings['IFILE'] = 'CONMIN_EESG.out'
	opt_problem.root.deriv_options['type']='fd'
	
	# Specificiency target efficiency(%)
	Eta_Target = 93.0	
	
	# Set bounds for design variables for an EESG designed for a 5MW turbine
	
	opt_problem.driver.add_desvar('r_s',lower=0.5,upper=9.0)
	opt_problem.driver.add_desvar('l_s', lower=0.5, upper=2.5)
	opt_problem.driver.add_desvar('h_s', lower=0.06, upper=0.15)
	opt_problem.driver.add_desvar('tau_p', lower=0.04, upper=0.2)
	opt_problem.driver.add_desvar('N_f', lower=10, upper=300)
	opt_problem.driver.add_desvar('I_f', lower=1, upper=500)
	opt_problem.driver.add_desvar('n_r', lower=5.0, upper=15.0)
	opt_problem.driver.add_desvar('h_yr', lower=0.01, upper=0.25)
	opt_problem.driver.add_desvar('h_ys', lower=0.01, upper=0.25)
	opt_problem.driver.add_desvar('b_r', lower=0.1, upper=1.5)
	opt_problem.driver.add_desvar('d_r', lower=0.1, upper=1.5)
	opt_problem.driver.add_desvar('t_wr', lower=0.001, upper=0.2)
	opt_problem.driver.add_desvar('n_s', lower=5.0, upper=15.0)
	opt_problem.driver.add_desvar('b_st', lower=0.1, upper=1.5)
	opt_problem.driver.add_desvar('d_s', lower=0.1, upper=1.5)
	opt_problem.driver.add_desvar('t_ws', lower=0.001, upper=0.2)
	
	# set up constraints for the PMSG_arms generator
	
	opt_problem.driver.add_constraint('B_symax',upper=2.0-1.0e-6)							#1
	opt_problem.driver.add_constraint('B_rymax',upper=2.0-1.0e-6)							#2
	opt_problem.driver.add_constraint('B_tmax',upper=2.0-1.0e-6)							#3
	opt_problem.driver.add_constraint('B_gfm',lower=0.617031,upper=1.057768)  #4
	opt_problem.driver.add_constraint('B_g',lower=0.7,upper=1.2)							#5
	opt_problem.driver.add_constraint('B_pc',upper=2.0)									  		#6
	opt_problem.driver.add_constraint('E_s',lower=500.0,upper=5000.0)					#7
	opt_problem.driver.add_constraint('con_uAs',lower=0.0+1.0e-6)							#8
	opt_problem.driver.add_constraint('con_zAs',lower=0.0+1.0e-6)							#9
	opt_problem.driver.add_constraint('con_yAs',lower=0.0+1.0e-6)  						#10
	opt_problem.driver.add_constraint('con_uAr',lower=0.0+1.0e-6)							#11
	opt_problem.driver.add_constraint('con_zAr',lower=0.0+1.0e-6)							#12
	opt_problem.driver.add_constraint('con_yAr',lower=0.0+1.0e-6) 						#13
	opt_problem.driver.add_constraint('con_TC2',lower=0.0+1.0e-6)							#14
	opt_problem.driver.add_constraint('con_TC3',lower=0.0+1e-6)								#15
	opt_problem.driver.add_constraint('con_br',lower=0.0+1e-6)								#16
	opt_problem.driver.add_constraint('con_bst',lower=0.0-1e-6)								#17
	opt_problem.driver.add_constraint('A_1',upper=60000.0-1e-6)								#18
	opt_problem.driver.add_constraint('J_s',upper=6.0) 									   		#19
	opt_problem.driver.add_constraint('J_f',upper=6.0)												#20
	opt_problem.driver.add_constraint('A_Cuscalc',lower=5.0,upper=300) 				#22
	opt_problem.driver.add_constraint('A_Curcalc',lower=10,upper=300)					#23
	opt_problem.driver.add_constraint('K_rad',lower=0.2+1e-6,upper=0.27)			#24
	opt_problem.driver.add_constraint('Slot_aspect_ratio',lower=4.0,upper=10.0)#25
	opt_problem.driver.add_constraint('gen_eff',lower=Eta_Target)							#26
	opt_problem.driver.add_constraint('n_brushes',upper=6)							      #27
	opt_problem.driver.add_constraint('Power_ratio',upper=2-1.0e-6)						#28
	
	opt_problem.setup()
	
	# Specify Target machine parameters
	opt_problem['machine_rating']=5000000.0
	opt_problem['Torque']=4.143289e6
	
	opt_problem['n_nom']=12.1
	
	# Initial design variables 
	opt_problem['r_s']=3.2
	opt_problem['l_s']=1.4
	opt_problem['h_s']= 0.060
	opt_problem['tau_p']= 0.170
	opt_problem['I_f']= 69
	opt_problem['N_f']= 100
	opt_problem['h_ys']= 0.130
	opt_problem['h_yr']= 0.120
	opt_problem['n_s']= 5
	opt_problem['b_st']= 0.470
	opt_problem['n_r']=5
	opt_problem['b_r']= 0.480
	opt_problem['d_r']= 0.510
	opt_problem['d_s']= 0.400
	opt_problem['t_wr']=0.140
	opt_problem['t_ws']=0.070
	opt_problem['R_o']=0.43      #10MW: 0.523950817,#5MW: 0.43, #3MW:0.363882632 #1.5MW: 0.2775  0.75MW: 0.17625
	
	# Costs
	opt_problem['C_Cu']=4.786
	opt_problem['C_Fe']= 0.556
	opt_problem['C_Fes']=0.50139
	
	#Material properties
	
	opt_problem['rho_Fe']= 7700                 #Magnetic Steel/iron density
	opt_problem['rho_Fes']= 7850                 #structural Steel density
	opt_problem['rho_Copper']=8900                  # Kg/m3 copper density
	
	opt_problem['main_shaft_cm']=np.array([0.0, 0.0, 0.0])
	opt_problem['main_shaft_length'] =2.0
	
	#Run optimization
	opt_problem.run()
		
	"""Uncomment to print solution to screen/an excel file 
		
	raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor Arms', 'Rotor Axial arm dimension','Rotor Circumferential arm dimension',\
		'Rotor Arm thickness', ' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Stator Radial deflection',' Stator Axial deflection',' Stator Circumferential deflection','Air gap diameter', 'Stator length',\
		'l/D ratio', 'Pole pitch', 'Stator slot height','Stator slot width','Slot aspect ratio','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Rotor pole height', 'Rotor pole width', 'Average no load flux density', \
		'Peak air gap flux density','Peak stator yoke flux density','Peak rotor yoke flux density','Stator tooth flux density','Rotor pole core flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage(rms value)', \
		'Generator Output phase current', 'Stator resistance', 'Synchronous inductance','Stator slots','Stator turns','Stator conductor cross-section','Stator Current density ','Specific current loading','Field turns','Conductor cross-section',\
		'Field Current','D.C Field resistance','MMF ratio at rated load(Rotor/Stator)','Excitation Power (% of Rated Power)','Number of brushes/polarity','Field Current density','Generator Efficiency', 'Iron mass', 'Copper mass','Mass of Arms','Total Mass','Total Cost'],\
		'Values': [opt_problem['machine_rating']/1e6,opt_problem['n_s'],opt_problem['d_s']*1000,opt_problem['b_st']*1000,opt_problem['t_ws']*1000,opt_problem['n_r'],opt_problem['d_r']*1000,opt_problem['b_r']*1000,opt_problem['t_wr']*1000,opt_problem['u_Ar']*1000,\
		opt_problem['y_Ar']*1000,opt_problem['z_A_r']*1000,opt_problem['u_As']*1000,opt_problem['y_As']*1000,opt_problem['z_A_s']*1000,2*opt_problem['r_s'],opt_problem['l_s'],opt_problem['K_rad'],opt_problem['tau_p']*1000,opt_problem['h_s']*1000,opt_problem['b_s']*1000,\
		opt_problem['Slot_aspect_ratio'],opt_problem['b_t']*1000,opt_problem['h_ys']*1000,opt_problem['h_yr']*1000,opt_problem['h_p']*1000,opt_problem['b_p']*1000,opt_problem['B_gfm'],opt_problem['B_g'],opt_problem['B_symax'],opt_problem['B_rymax'],opt_problem['B_tmax'],\
		opt_problem['B_pc'],opt_problem['p'],opt_problem['f'],opt_problem['E_s'],opt_problem['I_s'],opt_problem['R_s'],opt_problem['L_m'],opt_problem['S'],opt_problem['N_s'],opt_problem['A_Cuscalc'],opt_problem['J_s'],opt_problem['A_1']/1000,opt_problem['N_f'],opt_problem['A_Curcalc'],\
		opt_problem['I_f'],opt_problem['R_r'],opt_problem['Load_mmf_ratio'],opt_problem['Power_ratio'],opt_problem['n_brushes'],opt_problem['J_f'],opt_problem['gen_eff'],opt_problem['Iron']/1000,opt_problem['Copper']/1000,opt_problem['Structural_mass']/1000,\
		opt_problem['Mass']/1000,opt_problem['Costs']/1000],
		'Limit': ['','','',opt_problem['b_all_s']*1000,'','','',opt_problem['b_all_r']*1000,'',opt_problem['u_all_r']*1000,opt_problem['y_all']*1000,opt_problem['z_all_r']*1000,opt_problem['u_all_s']*1000,opt_problem['y_all']*1000,opt_problem['z_all_s']*1000,\
		'','','(0.2-0.27)','','','','(4-10)','','','','','','(0.62-1.05)','1.2','2','2','2','2','','(10-60)','','','','','','','','(3-6)','<60','','','','','','<2%','','(3-6)',Eta_Target,'','','','',''],
				'Units':['MW','unit','mm','mm','mm','unit','mm','mm','mm','mm','mm','mm','mm','mm','mm','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','om/phase',\
				'p.u','slots','turns','mm^2','A/mm^2','kA/m','turns','mm^2','A','ohm','%','%','brushes','A/mm^2','turns','%','tons','tons','tons','1000$']} 
		
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print df
		
	df.to_excel('EESG_'+str(opt_problem['machine_rating']/1e6)+'MW_1.7.x.xlsx')
		
	""" 

		
if __name__=="__main__":
	
	 # Run an example optimization of EESG generator on cost
	 EESG_Opt_example()
