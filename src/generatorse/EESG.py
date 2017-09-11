"""EESG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

from openmdao.main.api import Component,Assembly
from openmdao.main.datatypes.api import Float, Bool, Int, Str, Array, Enum
from openmdao.lib.drivers.api import COBYLAdriver, CONMINdriver,NEWSUMTdriver,SLSQPdriver,Genetic
import numpy as np
from numpy import array,float,min,sign
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan


class EESG(Component):
	
  	""" Estimates overall mass dimensions and Efficiency of Electrically Excited Synchronous generator. """
  	
  	# EESG generator design inputs
  	r_s = Float(iotype='in', desc='airgap radius r_s')
  	l_s = Float(iotype='in', desc='Stator core length l_s')
  	h_s = Float(iotype='in', desc='Yoke height h_s')
  	tau_p =Float(iotype='in', desc='Pole pitch self.tau_p')
  	machine_rating=Float(iotype='in', desc='Machine rating')
  	n_nom=Float(iotype='in', desc='rated speed')
  	Torque=Float(iotype='in', desc='Rated torque ')
  	I_f=Float(iotype='in', desc='Excitation current')
  	N_f=Float(iotype='in', desc='field turns')
  	h_ys=Float(0.0, iotype='in', desc='Yoke height')
  	h_yr=Float(0.0, iotype='in', desc='rotor yoke height')
  	
  	# structural design variables
  	n_s =Float(iotype='in', desc='number of stator arms n_s')
  	b_st = Float(iotype='in', desc='arm width b_r')
  	d_s= Float(iotype='in', desc='arm depth d_r')
  	t_ws =Float(iotype='in', desc='arm depth thickness self.t_wr')
  	n_r =Float(iotype='in', desc='number of arms n')
  	b_r = Float(iotype='in', desc='arm width b_r')
  	d_r = Float(iotype='in', desc='arm depth d_r')
  	t_wr =Float(iotype='in', desc='arm depth thickness self.t_wr')
  	t= Float(iotype='in', desc='rotor back iron')
  	t_s= Float(iotype='in', desc='stator back iron ')
  	R_o=Float(iotype='in', desc='Shaft radius')
  	
  	# Material properties
  	rho_Fes=Float(iotype='in', desc='Structural Steel density kg/m^3')
  	rho_Fe=Float(iotype='in', desc='Magnetic Steel density kg/m^3')
  	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
  	
  	# # Output parameters : Rotor design
  	h_p=Float(0.0, iotype='out', desc='Pole height')
  	b_p=Float(0.0, iotype='out', desc='Pole width')
  	p=Float(0.0, iotype='out', desc='No of pole pairs')
  	n_brushes=Float(0.0, iotype='out', desc='number of brushes')
  	A_Curcalc=Float(iotype='out', desc='Rotor Conductor cross-section')
  	
  	# Stator design
  	b_s=Float(iotype='out', desc='Stator slot width')
  	b_t=Float(iotype='out', desc='tooth width')
  	A_Cuscalc=Float(iotype='out', desc='Stator Conductor cross-section mm^2')
  	S=Float(0.0, iotype='out', desc='No of stator slots')
  	N_s=Float(iotype='out', desc='Stator turns')
  	
  	# Output parameters : Electrical performance
  	f=Float(0.0, iotype='out', desc='Output frequency')
  	E_s=Float(0.0, iotype='out', desc='Stator phase voltage')
  	I_s=Float(0.0, iotype='out', desc='Generator output phase current')
  	R_s=Float(0.0, iotype='out', desc='Stator resistance')
  	R_r=Float(0.0, iotype='out', desc='Rotor resistance')
  	L_m=Float(0.0, iotype='out', desc='Stator synchronising inductance')
  	J_s=Float(0.0, iotype='out', desc='Stator Current density')
  	J_f=Float(0.0, iotype='out', desc='rotor Current density')
  	A_1=Float(0.0, iotype='out', desc='Specific current loading')
  	Load_mmf_ratio=Float(iotype='out', desc='mmf_ratio')
  	gen_eff=Float(0.01, iotype='out', desc='Generator efficiency')
  	
  	
  	# Magnetic loading
  	B_g = Float(0.0, iotype='out', desc='Peak air gap flux density B_g')
  	B_gfm=Float(0.0, iotype='out', desc='Average air gap flux density B_g')
  	B_symax = Float(0.0, iotype='out', desc='Peak Stator Yoke flux density self.B_symax')
  	B_rymax = Float(0.0, iotype='out', desc='Peak rotor Yoke flux density B_rymax')
  	B_tmax= Float(0.0, iotype='out', desc='Peak tooth density self.B_tmax')
  	B_pc=Float(0.0, iotype='out', desc='Pole core flux density')
  	
  	# Objective functions and output
  	Costs= Float(0.0, iotype='out', desc='Total Costs')
  	Losses=Float(0.0, iotype='out', desc='Total Loss')
  	K_rad=Float(0.0, iotype='out', desc='K_rad')
  	Mass=Float(0.0, iotype='out', desc='Total loss')
  	
  	# Structural performance
  	u_Ar	=Float(iotype='out', desc='Rotor radial deflection')
  	y_Ar =Float(iotype='out', desc='Rotor axial deflection')
  	z_A_r=Float(iotype='out', desc='Rotor circumferential deflection')      # circumferential deflection
  	u_As=Float(iotype='out', desc='Stator radial deflection')
  	y_As =Float(iotype='out', desc='Stator axial deflection')
  	z_A_s=Float(iotype='out', desc='Stator circumferential deflection')
  	u_all_r=Float(0.0, iotype='out', desc='Allowable radial deflection')
  	u_all_s=Float(0.0, iotype='out', desc='Stator Allowable radial deflection')
  	y_all=Float(0.0, iotype='out', desc='Allowable axial deflection')
  	z_all_r=Float(0.0, iotype='out', desc='Allowable circum deflection')
  	z_all_s=Float(0.0, iotype='out', desc='Allowable circum deflection')
  	b_all_s=Float(0.0, iotype='out', desc='Allowable arm space')
  	b_all_r=Float(0.0, iotype='out', desc='Allowable arm space')
  	TC1	=Float(iotype='out', desc='Torque constraint')
  	TC2=Float(iotype='out', desc='Torque constraint-rotor')
  	TC3=Float(iotype='out', desc='Torque constraint-stator')
  	
  	# Other outputs
  	Iron=Float(0.01, iotype='out', desc='Iron mass')
  	Copper=Float(0.01, iotype='out', desc='Copper mass')
  	Structural_mass=Float(0.01, iotype='out', desc='Structure mass')
  	Power_ratio=Float(iotype='out', desc='Power_ratio')
  	Slot_aspect_ratio=Float(iotype='out', desc='Stator slot aspect ratio')
  	
  	#inputs/outputs for interface with drivese
  	main_shaft_cm = Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc='Main shaft CM')
  	main_shaft_length=Float(iotype='in', desc='main shaft length')
  	cm=Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='COM [x,y,z]')
  	I=Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
  	
  	def execute(self):
    	
    	  r_s = self.r_s
    	  l_s = self.l_s
    	  h_s = self.h_s
    	  tau_p =self.tau_p
    	  N_f=self.N_f
    	  I_f=self.I_f
    	  
    	  B_g = self.B_g
    	  B_symax = self.B_symax
    	  B_rymax=self.B_rymax
    	  B_tmax=self.B_tmax
    	  B_pc=self.B_pc
    	  B_gfm=self.B_gfm
    	  
    	  Losses=self.Losses
    	  Mass=self.Mass
    	  K_rad=self.K_rad
    	  b_s =self.b_s
    	  b_t= self.b_t
    	  h_ys = self.h_ys
    	  h_yr  = self.h_yr
    	  h_p =self.h_p
    	  b_p =self.b_p
    	  
    	  f=self.f
    	  E_s=self.E_s
    	  I_s=self.I_s
    	  R_s=self.R_s
    	  L_m=self.L_m
    	  J_s=self.J_s
    	  J_f=self.J_f
    	  gen_eff =self.gen_eff
    	  A_1=self.A_1
    	  
    	  N_s=self.N_s
    	  p=self.p
    	  A_Cuscalc=self.A_Cuscalc
    	  A_Curcalc=self.A_Curcalc
    	  n_brushes=self.n_brushes
    	  
    	  Iron=self.Iron
    	  Copper=self.Copper
    	  Structural_mass=self.Structural_mass
    	  
    	  R_r=self.R_r
    	  S=self.S
    	  t=self.t
    	  t_s=self.t_s
    	  u_Ar	=self.u_Ar
    	  y_Ar =self.y_Ar
    	  z_A_r=self.z_A_r
    	  u_As	=self.u_As
    	  y_As =self.y_As
    	  z_A_s=self.z_A_s
    	  u_all_r=self.u_all_r
    	  u_all_s=self.u_all_s
    	  y_all=self.y_all
    	  z_all_r=self.z_all_r
    	  z_all_s=self.z_all_s
    	  b_all_r=self.b_all_r
    	  b_all_s=self.b_all_s
    	  TC1=self.TC1
    	  TC2=self.TC2
    	  TC3=self.TC3
    	  
    	  n_s=self.n_s
    	  d_s=self.d_s
    	  b_st=self.b_st
    	  t_ws=self.t_ws
    	  n_r=self.n_r
    	  d_r=self.d_r
    	  b_r=self.b_r
    	  t_wr=self.t_wr
    	  R_o=self.R_o
    	  Power_ratio=self.Power_ratio
    	  Slot_aspect_ratio=self.Slot_aspect_ratio
    	  Torque=self.Torque
    	  machine_rating=self.machine_rating
    	  n_nom=self.n_nom
    	  
    	  rho_Fe= self.rho_Fe
    	  rho_Fes=self.rho_Fes
    	  rho_Copper=self.rho_Copper
    	  
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
    	  
    	  B_r1=(mu_0*self.I_f*self.N_f*4*sin(0.5*(b_p/self.tau_p)*pi))/g_eff/pi
    	  
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
    	  
    	  self.u_As				=(q3*R_st**2/E/self.t_s)*(1+Numers/Denoms)
    	  
    	  # Calculating axial deflection according to McDonald
    	  l_is      =R_st-self.R_o
    	  l_iis     =l_is
    	  l_iiis    =l_is
    	  mass_st_lam_s= M_Fest+pi*l*self.rho_Fe*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2)
    	  W_is			=g1*sin(phi)*(self.rho_Fes*l*self.d_s**2*0.5) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
    	  W_iis     =g1*sin(phi)*(V_Cuss*self.rho_Copper+mass_st_lam_s)/2/N_st
    	  w_s         =self.rho_Fes*g1*sin(phi)*a_s*N_st
    	  
    	  print (V_Cusr*self.rho_Copper+M_Fert)*g1
    	  
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
  
####################################################Cost Analysis#######################################################################
class EESG_Cost(Component):
	
	""" Provides a material cost estimate for an EESG generator. Manufacturing costs are excluded"""
	# Inputs
	# Specific cost of material by type
	C_Cu=Float( iotype='in', desc='Specific cost of copper')
	C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
	
	# Mass of each material type
	Copper=Float(iotype='in', desc='Copper mass')
	Iron=Float(iotype='in', desc='Iron mass')
	Structural_mass=Float(iotype='in', desc='Structural mass')
	# Outputs
	Costs= Float(iotype='out', desc='Total cost')
	
	def execute(self):
		Copper=self.Copper
		C_Cu=self.C_Cu
		Iron=self.Iron
		C_Fe=self.C_Fe
		C_Fes=self.C_Fes
		Costs=self.Costs
		Structural_mass=self.Structural_mass
		
		# Material cost as a function of material mass and specific cost of material
		K_gen=self.Copper*self.C_Cu+self.Iron*self.C_Fe
		Cost_str=self.C_Fes*self.Structural_mass
		self.Costs=K_gen+Cost_str
	

####################################################OPTIMISATION SET_UP ###############################################################

  
class EESG_Opt(Assembly):
	Eta_target=Float(iotype='in', desc='Target drivetrain efficiency')
	T_rated=Float(iotype='in', desc='Torque')
	N_rated=Float(iotype='in', desc='rated speed')
	P_rated=Float(iotype='in', desc='rated power')
	main_shaft_cm = Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc='High speed side CM')
	main_shaft_length=Float(iotype='in', desc='main shaft length')
	Objective_function=Str(iotype='in')
	Optimiser=Str(iotype = 'in')
	L=Float(iotype='out')
	Mass=Float(iotype='out')
	Efficiency=Float(iotype='out')
	r_s=Float(iotype='out', desc='Optimised radius')
	l_s=Float(iotype='out', desc='Optimised generator length')
	I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
	cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' Center of mass [x, y,z]')
	EESG_r_s= Float(iotype='in', units='m', desc='Air gap radius of a permanent magnet excited synchronous generator')
	EESG_l_s= Float(iotype='in', units='m', desc='Core length of the permanent magnet excited synchronous generator')
	EESG_h_s = Float(iotype='in', units='m', desc='Stator Slot height of the permanent magnet excited synchronous generator')
	EESG_tau_p = Float(iotype='in', units='m', desc='Pole pitch of the permanent magnet excited synchronous generator')
	EESG_I_f = Float(iotype='in', units='m', desc='No-load excitation current for the electrically excited synchronous generator')
	EESG_N_f = Float(iotype='in', units='m', desc='Field turns in electrically excited synchronous generator')
	EESG_h_ys = Float(iotype='in', units='m', desc='Stator yoke height of the permanent magnet excited synchronous generator')
	EESG_h_yr = Float(iotype='in', units='m', desc='Rotor yoke height of the permanent magnet excited synchronous generator')
	EESG_n_s = Float(iotype='in', units='m', desc='Stator Spokes of the permanent magnet excited synchronous generator')
	EESG_b_st = Float(iotype='in', units='m', desc='Circumferential arm dimension of stator spoke')
	EESG_n_r =Float(0.0, iotype='in', desc='Number of Rotor Spokes of the permanent magnet excited synchronous generator')
	EESG_b_r = Float(0.0, iotype='in', desc='Circumferential arm dimension of the rotor spoke')
	EESG_d_r = Float(0.0, iotype='in', desc='Rotor arm depth')
	EESG_d_s= Float(0.0, iotype='in', desc='Stator arm depth ')
	EESG_t_wr =Float(0.0, iotype='in', desc='Rotor arm thickness')
	EESG_t_ws =Float(0.0, iotype='in', desc='Stator arm thickness')
	EESG_R_o =Float(0.0, iotype='in', desc='Main shaft radius')
	C_Cu=Float( iotype='in', desc='Specific cost of copper')
	C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	C_Fes=Float(iotype='in', desc='Specific cost of structural steel')

	
	rho_Fes=Float(iotype='in', desc='Structural Steel density kg/m^3')
	rho_Fe=Float(iotype='in', desc='Magnetic Steel density kg/m^3')
	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')


  
	
	def __init__(self,Optimiser='',Objective_function='',print_results=''):
		
		super(EESG_Opt,self).__init__()
		
		""" Creates a new Assembly containing EESG and an optimizer"""
		
		# add EESG component, connect i/o
		self.add('EESG',EESG())
		self.connect('EESG_r_s','EESG.r_s')
		self.connect('EESG_l_s','EESG.l_s')
		self.connect('EESG_h_s','EESG.h_s')
		self.connect('EESG_tau_p','EESG.tau_p')
		self.connect('EESG_I_f','EESG.I_f')
		self.connect('EESG_N_f','EESG.N_f')
		self.connect('EESG_h_ys','EESG.h_ys')
		self.connect('EESG_h_yr','EESG.h_yr')
		self.connect('EESG_n_s','EESG.n_s')
		self.connect('EESG_b_st','EESG.b_st')
		self.connect('EESG_d_s','EESG.d_s')
		self.connect('EESG_t_ws','EESG.t_ws')
		self.connect('EESG_n_r','EESG.n_r')
		self.connect('EESG_b_r','EESG.b_r')
		self.connect('EESG_d_r','EESG.d_r')
		self.connect('EESG_t_wr','EESG.t_wr')
		self.connect('EESG_R_o','EESG.R_o')
		self.connect('P_rated','EESG.machine_rating')
		self.connect('N_rated','EESG.n_nom')
		self.connect('main_shaft_cm','EESG.main_shaft_cm')
		self.connect('main_shaft_length','EESG.main_shaft_length')
		self.connect('T_rated','EESG.Torque')
		self.connect('EESG.Mass','Mass')
		self.connect('EESG.gen_eff','Efficiency')
		self.connect('EESG.r_s','r_s')
		self.connect('EESG.l_s','l_s')
		self.connect('EESG.I','I')
		self.connect('EESG.cm','cm')
		self.connect('rho_Fe','EESG.rho_Fe')
		self.connect('rho_Fes','EESG.rho_Fes')
		self.connect('rho_Copper','EESG.rho_Copper')
		
		# add EESG_Cost component, connect i/o
		self.add('EESG_Cost',EESG_Cost())
		self.connect('C_Fe','EESG_Cost.C_Fe')
		self.connect('C_Fes','EESG_Cost.C_Fes')
		self.connect('C_Cu','EESG_Cost.C_Cu')
		self.connect('EESG.Iron','EESG_Cost.Iron')
		self.connect('EESG.Copper','EESG_Cost.Copper')
		self.connect('EESG.Structural_mass','EESG_Cost.Structural_mass')
		
		# add optimizer and set-up problem (using user defined input on objective function"
		self.Optimiser=Optimiser
		self.Objective_function=Objective_function
		Opt1=globals()[self.Optimiser]
		self.add('driver',Opt1())
		self.driver.add_objective(self.Objective_function)
		
		# set up design variables for the EESG generator
		self.driver.design_vars=['EESG_r_s','EESG_l_s','EESG_h_s','EESG_tau_p','EESG_h_m','EESG_n_r','EESG_n_s','EESG_b_r','EESG_b_st','EESG_d_r','EESG_d_st','EESG_t_wr','EESG_t_ws']
		self.driver.add_parameter('EESG_r_s', low=0.5, high=9)
		self.driver.add_parameter('EESG_l_s', low=0.5, high=2.5)
		self.driver.add_parameter('EESG_h_s', low=0.06, high=0.15)
		self.driver.add_parameter('EESG_tau_p', low=0.04, high=.2)
		self.driver.add_parameter('EESG_N_f', low=10, high=300)
		self.driver.add_parameter('EESG_I_f', low=1, high=500)
		self.driver.add_parameter('EESG_h_ys', low=0.01, high=0.25)
		self.driver.add_parameter('EESG_h_yr', low=0.01, high=0.25)
		self.driver.add_parameter('EESG_n_s', low=5., high=15.)
		self.driver.add_parameter('EESG_b_st', low=0.1, high=1.5)
		self.driver.add_parameter('EESG_d_s', low=0.1, high=1.5)   #UL=0.6
		self.driver.add_parameter('EESG_t_ws', low=0.001, high=0.2)
		self.driver.add_parameter('EESG_n_r', low=5., high=15.)
		self.driver.add_parameter('EESG_b_r', low=0.1, high=1.5)
		self.driver.add_parameter('EESG_d_r', low=0.1, high=1.5)   #UL=0.6
		self.driver.add_parameter('EESG_t_wr', low=0.001, high=0.2)
		
		self.driver.iprint=print_results
		
		# set up constraints for the PMSG_arms generator
					
		self.driver.add_constraint('EESG.B_symax<2')										  #1
		self.driver.add_constraint('EESG.B_rymax<2')										  #2
		self.driver.add_constraint('EESG.B_tmax<2')									      #3
		self.driver.add_constraint('EESG.B_gfm>=0.617031')            		#4
		self.driver.add_constraint('EESG.B_gfm<=1.057768')								#5
		self.driver.add_constraint('EESG.B_g>=0.7')                   		#6
		self.driver.add_constraint('EESG.B_g<=1.2')									  		#7
		self.driver.add_constraint('EESG.B_pc<=2')									  		#8
		self.driver.add_constraint('EESG.E_s>=500')											  #9
		self.driver.add_constraint('EESG.E_s<=5000')											#10
		self.driver.add_constraint('EESG.u_As<EESG.u_all_s')							#11
		self.driver.add_constraint('EESG.z_A_s<EESG.z_all_s')							#12
		self.driver.add_constraint('EESG.y_As<EESG.y_all')  							#13
		self.driver.add_constraint('EESG.u_Ar<EESG.u_all_r')							#14
		self.driver.add_constraint('EESG.z_A_r<EESG.z_all_r')							#15
		self.driver.add_constraint('EESG.y_Ar<EESG.y_all') 								#16
		self.driver.add_constraint('EESG.TC1<EESG.TC2')    								#17
		self.driver.add_constraint('EESG.TC1<EESG.TC3')    								#18
		self.driver.add_constraint('EESG.b_r<EESG.b_all_r')								#19
		self.driver.add_constraint('EESG.b_st<EESG.b_all_s')							#20
		self.driver.add_constraint('EESG.A_1<60000')											#21
		self.driver.add_constraint('EESG.J_s<=6') 												#22
		self.driver.add_constraint('EESG.J_f<=6')										      #23
		self.driver.add_constraint('EESG.A_Cuscalc>=5') 									#24
		self.driver.add_constraint('EESG.A_Cuscalc<=300')						      #25
		self.driver.add_constraint('EESG.A_Curcalc>=10')						      #26
		self.driver.add_constraint('EESG.A_Curcalc<=300')						      #27
		self.driver.add_constraint('EESG.K_rad>0.2')											#28
		self.driver.add_constraint('EESG.K_rad<=0.27')										#29
		self.driver.add_constraint('EESG.Slot_aspect_ratio>=4')						#30
		self.driver.add_constraint('EESG.Slot_aspect_ratio<=10')					#31
		self.driver.add_constraint('EESG.gen_eff>=Eta_target')			#32
		self.driver.add_constraint('EESG.n_brushes<=6')							      #33
		self.driver.add_constraint('EESG.Power_ratio<2')									#34

						
def EESG_Opt_example():
	
		#Example optimization of a EESG generator for costs on a 5 MW reference turbine
		
		opt_problem = EESG_Opt('CONMINdriver','EESG_Cost.Costs',1)
		
		# Initial design variables for a DD EESG designed for a 5MW turbine
		opt_problem.Eta_target=93.0
		opt_problem.P_rated=5e6
		opt_problem.T_rated=4.143289e6     # 0.75MW:250418.6168   ;1.5MW: 698729.0185; 3MW: 1.7904931e6
		opt_problem.N_rated=12.1
		opt_problem.EESG_r_s=3.2
		opt_problem.EESG_l_s= 1.4
		opt_problem.EESG_h_s = 0.060
		opt_problem.EESG_tau_p = 0.170
		opt_problem.EESG_I_f = 69
		opt_problem.EESG_N_f = 100
		opt_problem.EESG_h_ys = 0.130
		opt_problem.EESG_h_yr = 0.120
		opt_problem.EESG_n_s = 5
		opt_problem.EESG_b_st = 0.470
		opt_problem.EESG_n_r =5
		opt_problem.EESG_b_r = 0.480
		opt_problem.EESG_d_r = 0.510
		opt_problem.EESG_d_s= 0.400
		opt_problem.EESG_t_wr =0.140
		opt_problem.EESG_t_ws =0.070
		opt_problem.EESG_R_o =0.43      #10MW: 0.523950817,#5MW: 0.43, #3MW:0.363882632 #1.5MW: 0.2775  0.75MW: 0.17625
		
		# Costs
		opt_problem.C_Cu   =4.786
		opt_problem.C_Fe	= 0.556
		opt_problem.C_Fes =0.50139
		
		#Material properties
		
		opt_problem.rho_Fe = 7700                 #Magnetic Steel/iron density
		opt_problem.rho_Fes = 7850                 #structural Steel density
		opt_problem.rho_Copper =8900                  # Kg/m3 copper density
		
		#Run optimization
		
		opt_problem.run()
		
		""" Uncomment to print solution to an excel file"""
		
		import pandas
		
		raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor Arms', 'Rotor Axial arm dimension','Rotor Circumferential arm dimension','Rotor Arm thickness', ' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Stator Radial deflection',' Stator Axial deflection',' Stator Circumferential deflection','Air gap diameter', 'Stator length','l/D ratio', 'Pole pitch', 'Stator slot height','Stator slot width','Slot aspect ratio','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Rotor pole height', 'Rotor pole width', 'Average no load flux density', 'Peak air gap flux density','Peak stator yoke flux density','Peak rotor yoke flux density','Stator tooth flux density','Rotor pole core flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage(rms value)', 'Generator Output phase current', 'Stator resistance', 'Synchronous inductance','Stator slots','Stator turns','Stator conductor cross-section','Stator Current density ','Specific current loading','Field turns','Conductor cross-section','Field Current','D.C Field resistance','MMF ratio at rated load(Rotor/Stator)','Excitation Power (% of Rated Power)','Number of brushes/polarity','Field Current density','Generator Efficiency', 'Iron mass', 'Copper mass','Mass of Arms','Total Mass','Total Cost'],
		'Values': [opt_problem.EESG.machine_rating/1e6,opt_problem.EESG.n_s,opt_problem.EESG.d_s*1000,opt_problem.EESG.b_st*1000,opt_problem.EESG.t_ws*1000,opt_problem.EESG.n_r,opt_problem.EESG.d_r*1000,opt_problem.EESG.b_r*1000,opt_problem.EESG.t_wr*1000,opt_problem.EESG.u_Ar*1000,opt_problem.EESG.y_Ar*1000,opt_problem.EESG.z_A_r*1000,opt_problem.EESG.u_As*1000,opt_problem.EESG.y_As*1000,opt_problem.EESG.z_A_s*1000,2*opt_problem.EESG.r_s,opt_problem.EESG.l_s,opt_problem.EESG.K_rad,opt_problem.EESG.tau_p*1000,opt_problem.EESG.h_s*1000,opt_problem.EESG.b_s*1000,opt_problem.EESG.Slot_aspect_ratio,opt_problem.EESG.b_t*1000,opt_problem.EESG.h_ys*1000,opt_problem.EESG.h_yr*1000,opt_problem.EESG.h_p*1000,opt_problem.EESG.b_p*1000,opt_problem.EESG.B_gfm,opt_problem.EESG.B_g,opt_problem.EESG.B_symax,opt_problem.EESG.B_rymax,opt_problem.EESG.B_tmax,opt_problem.EESG.B_pc,opt_problem.EESG.p,opt_problem.EESG.f,opt_problem.EESG.E_s,opt_problem.EESG.I_s,opt_problem.EESG.R_s,opt_problem.EESG.L_m,opt_problem.EESG.S,opt_problem.EESG.N_s,opt_problem.EESG.A_Cuscalc,opt_problem.EESG.J_s,opt_problem.EESG.A_1/1000,opt_problem.EESG.N_f,opt_problem.EESG.A_Curcalc,opt_problem.EESG.I_f,opt_problem.EESG.R_r,opt_problem.EESG.Load_mmf_ratio,opt_problem.EESG.Power_ratio,opt_problem.EESG.n_brushes,opt_problem.EESG.J_f,opt_problem.EESG.gen_eff,opt_problem.EESG.Iron/1000,opt_problem.EESG.Copper/1000,opt_problem.EESG.Structural_mass/1000,opt_problem.EESG.Mass/1000,opt_problem.EESG_Cost.Costs/1000],
			'Limit': ['','','',opt_problem.EESG.b_all_s*1000,'','','',opt_problem.EESG.b_all_r*1000,'',opt_problem.EESG.u_all_r*1000,opt_problem.EESG.y_all*1000,opt_problem.EESG.z_all_r*1000,opt_problem.EESG.u_all_s*1000,opt_problem.EESG.y_all*1000,opt_problem.EESG.z_all_s*1000,'','','(0.2-0.27)','','','','(4-10)','','','','','','(0.62-1.05)','1.2','2','2','2','2','','(10-60)','','','','','','','','(3-6)','<60','','','','','','<2%','','(3-6)',opt_problem.Eta_target,'','','','',''],
				'Units':['MW','unit','mm','mm','mm','unit','mm','mm','mm','mm','mm','mm','mm','mm','mm','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','om/phase','p.u','slots','turns','mm^2','A/mm^2','kA/m','turns','mm^2','A','ohm','%','%','brushes','A/mm^2','turns','%','tons','tons','tons','1000$']} 
		
		df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
		print df
		df.to_excel('EESG_'+str(opt_problem.P_rated/1e6)+'MW.xlsx')
		
		""" """ 

		
if __name__=="__main__":
	
	 # Run an example optimization of EESG generator on cost
	 EESG_Opt_example()