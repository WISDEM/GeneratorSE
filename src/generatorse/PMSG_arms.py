<<<<<<< HEAD
"""PMDD.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.main.api import Assembly
from openmdao.lib.datatypes.api import Float,Array,Str
import numpy as np
from numpy import array
from numpy import float 
from numpy import min
from scipy.interpolate import interp1d
from openmdao.lib.drivers.api import COBYLAdriver, CONMINdriver,NEWSUMTdriver,SLSQPdriver,Genetic
import pandas as pd


class PMSG(Component):

 """ Evaluates the total cost """
 
 r_s = Float(iotype='in', desc='airgap radius r_s')
 l_s = Float(iotype='in', desc='Stator core length l_s')
 h_s = Float(iotype='in', desc='Yoke height h_s')
 tau_p =Float(iotype='in', desc='Pole pitch self.tau_p')
 B_g = Float(iotype='out', desc='Peak air gap flux density B_g')
 machine_rating=Float(iotype='in', desc='Machine rating')
 n_nom=Float(iotype='in', desc='rated speed')
 Torque=Float(iotype='in', desc='Rated torque ')
 h_m=Float(iotype='in', desc='magnet height')
 h_ys=Float(iotype='in', desc='Yoke height')
 h_yr=Float(iotype='in', desc='rotor yoke height')
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
 
 # Costs and material properties
 C_Cu=Float( iotype='in', desc='Specific cost of copper')
 C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
 C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
 C_PM=Float(iotype='in', desc='Specific cost of Magnet')
 
 rho_Fes=Float(iotype='in', desc='Structural Steel density kg/m^3')
 rho_Fe=Float(iotype='in', desc='Magnetic Steel density kg/m^3')
 rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
 rho_PM=Float(iotype='in', desc='Magnet density kg/m^3')
 
 # Magnetic loading
 B_symax = Float(iotype='out', desc='Peak Stator Yoke flux density B_ymax')
 B_tmax=Float(iotype='out', desc='Peak Teeth flux density')
 B_rymax=Float(iotype='out', desc='Peak Rotor yoke flux density')
 B_smax=Float(iotype='out', desc='Peak Stator flux density')
 B_pm1=Float(iotype='out', desc='Fundamental component of peak air gap flux density')
 
 #Stator design
 N_s =Float(iotype='out', desc='Number of turns in the stator winding')
 b_s=Float(iotype='out', desc='slot width')
 b_t=Float(iotype='out', desc='tooth width')
 A_Cuscalc=Float(iotype='out', desc='Conductor cross-section mm^2')
 
 #Rotor magnet dimension
 b_m=Float(iotype='out', desc='magnet width')
 p=Float(iotype='out', desc='No of pole pairs')
  
 # Electrical performance 
 E_p=Float(iotype='out', desc='Stator phase voltage')
 f=Float(iotype='out', desc='Generator output frequency')
 I_s=Float(iotype='out', desc='Generator output phase current')
 R_s=Float(iotype='out', desc='Stator resistance')
 L_s=Float(iotype='out', desc='Stator synchronising inductance')
 A_1 =Float(0,iotype='out', desc='Electrical loading')
 J_s=Float(iotype='out', desc='Current density')
 
 # Objective functions
 M_actual=Float(iotype='out', desc='Actual mass')
 Mass=Float(iotype='out', desc='Actual mass')
 Costs= Float(iotype='out', desc='Total cost')
 K_rad=Float(iotype='out', desc='K_rad')
 Losses=Float(iotype='out', desc='Total loss')
 
 gen_eff=Float(iotype='out', desc='Generator efficiency')
 Active=Float(iotype='out', desc='Generator efficiency')
 Stator=Float(iotype='out', desc='Generator efficiency')
 Rotor=Float(iotype='out', desc='Generator efficiency')
 mass_PM=Float(iotype='out', desc='Generator efficiency')
 M_Cus		=Float(iotype='out', desc='Generator efficiency')
 M_Fest	=Float(iotype='out', desc='Generator efficiency')
 M_Fesy	=Float(iotype='out', desc='Generator efficiency')
 M_Fery	=Float(iotype='out', desc='Generator efficiency')
 Iron=Float(iotype='out', desc='Electrical steel mass')
 
 # Structural performance
 Stator_delta_radial=Float(iotype='out', desc='Rotor radial deflection')
 Stator_delta_axial=Float(iotype='out', desc='Stator Axial deflection')
 Stator_circum=Float(iotype='out', desc='Rotor radial deflection')
 Rotor_delta_radial=Float(iotype='out', desc='Generator efficiency')
 Rotor_delta_axial=Float(iotype='out', desc='Rotor Axial deflection')
 Rotor_circum=Float(iotype='out', desc='Rotor circumferential deflection')
 
 P_gennom=Float(iotype='in', desc='Generator Power')
 u_Ar	=Float(iotype='out', desc='Rotor radial deflection')
 y_Ar =Float(iotype='out', desc='Rotor axial deflection')
 z_A_r=Float(iotype='out', desc='Rotor circumferential deflection')      # circumferential deflection
 u_As=Float(iotype='out', desc='Stator radial deflection')
 y_As =Float(iotype='out', desc='Stator axial deflection')
 z_A_s=Float(iotype='out', desc='Stator circumferential deflection')  
 u_all_r=Float(iotype='out', desc='Allowable radial rotor')
 u_all_s=Float(iotype='out', desc='Allowable radial stator')
 y_all=Float(iotype='out', desc='Allowable axial')
 z_all_s=Float(iotype='out', desc='Allowable circum stator')
 z_all_r=Float(iotype='out', desc='Allowable circum rotor')
 b_all_s=Float(iotype='out', desc='Allowable arm')
 b_all_r=Float(iotype='out', desc='Allowable arm dimensions')
 TC1	=Float(iotype='out', desc='Torque constraint')
 TC2=Float(iotype='out', desc='Torque constraint-rotor')
 TC3=Float(iotype='out', desc='Torque constraint-stator')
 R_out=Float(iotype='out', desc='Outer radius')
 N_s_max=Float(iotype='out', desc='Maximum number of turns per coil')
 S			=Float(iotype='out', desc='Stator slots')
 K_load=Float(iotype='out', desc='Load factor')
 Slot_aspect_ratio=Float(iotype='out', desc='Slot aspect ratio')
 
 main_shaft_cm = Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc='Main Shaft CM')
 main_shaft_length=Float(iotype='in', desc='main shaft length')
 cm=Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='COM [x,y,z]')
 I=Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
 
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  tau_p =self.tau_p
  B_g = self.B_g
  n_r =self.n_r
  n_s=self.n_s
  t= self.t
  t_s=self.t_s
  b_r = self.b_r
  d_r = self.d_r
  t_wr =self.t_wr
  b_st = self.b_st
  d_s = self.d_s
  t_ws =self.t_ws
  b_s =self.b_s
  b_t= self.b_t
  h_ys = self.h_ys
  h_yr  = self.h_yr
  h_m =self.h_m
  b_m =self.b_m
  f=self.f
  E_p=self.E_p
  I_s=self.I_s
  R_s=self.R_s
  L_s=self.L_s
  J_s=self.J_s
  M_actual=self.M_actual
  Mass=self.Mass
  gen_eff =self.gen_eff
  Losses=self.Losses
  A_1=self.A_1
  K_rad=self.K_rad
  N_s=self.N_s
  Active=self.Active
  Stator=self.Stator
  Rotor=self.Rotor
  mass_PM=self.mass_PM
  M_Cus		=self.M_Cus	
  M_Fest	=self.M_Fest
  M_Fesy	=self.M_Fesy
  M_Fery	=self.M_Fery
  Iron=self.Iron
  Stator_delta_radial=self.Stator_delta_radial
  Stator_delta_axial=self.Stator_delta_axial
  Stator_circum=self.Stator_circum
  Rotor_delta_radial=self.Rotor_delta_radial
  Rotor_delta_axial=self.Rotor_delta_axial
  Rotor_circum=self.Rotor_circum
  P_gennom=self.P_gennom
  u_As=self.u_As
  u_Ar=self.u_Ar
  y_As=self.y_As
  y_Ar=self.y_Ar
  z_A_s=self.z_A_s
  z_A_r=self.z_A_r
  z_all_s=self.z_all_s
  z_all_r=self.z_all_r
  R_out=self.R_out
  B_pm1=self.B_pm1
  B_tmax= self.B_tmax
  B_symax=self.B_symax
  B_rymax=self.B_rymax
  B_smax=self.B_smax
  B_pm1=self.B_pm1
  N_s_max=self.N_s_max
  A_Cuscalc=self.A_Cuscalc
  b_all_r=self.b_all_r
  b_all_s=self.b_all_s
  S=self.S
  Stator=self.Stator
  Rotor=self.Rotor
  K_load=self.K_load
  machine_rating=self.machine_rating
  Slot_aspect_ratio=self.Slot_aspect_ratio
  Torque=self.Torque
  main_shaft_cm = self.main_shaft_cm
  main_shaft_length=self.main_shaft_length
  cm=self.cm
  I=self.I
  TC1	=self.TC1
  TC2=self.TC2
  TC3=self.TC3
  R_o=self.R_o
  Costs=self.Costs
  n_nom = self.n_nom
  
  C_Cu=self.C_Cu
  C_Fe=self.C_Fe
  C_Fes=self.C_Fes
  C_PM =self.C_PM
  
  rho_Fe= self.rho_Fe
  rho_Fes=self.rho_Fes
  rho_Copper=self.rho_Copper
  rho_PM=self.rho_PM
  
	
  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign
  
  B_r    =1.2                 # Tesla remnant flux density 
  g1     =9.81                # m/s^2 acceleration due to gravity
  E      =2e11                # N/m^2 young's modulus
  sigma  =40e3                # shear stress assumed
  ratio  =0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
  mu_0   =pi*4e-7              # permeability of free space
  mu_r   =1.06								# relative permeability 
  phi    =90*2*pi/360         # tilt angle (rotor tilt -90 degrees during transportation)
  cofi   =0.85                 # power factor
  
  h_sy0  =0
  h_w    =0.005
  h_i    =0.001 										# coil insulation thickness
  h_s1   = 0.001
  h_s2   =0.004
  h_s3   =self.h_s-h_s1-h_s2
  h_cu   =(h_s3-4*h_i)*0.5
  self.t_s =self.h_ys
  self.t =self.h_yr
  y_tau_p=1
  self.K_rad=self.l_s/(2*self.r_s)
  m      =3                    # no of phases
  B_rmax=1.4                  
  q1     =1                    # no of slots per pole per phase
  b_s_tau_s=0.45
  k_sfil =0.65								 # Slot fill factor
  P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  rho_Cu=1.8*10**(-8)*1.4
  k_fes =0.9
  T =   self.Torque    #549296.586 #4143289.841 #9549296.586 #250418.6168  #698729.0185 #1.7904931e6 # #248679.5986 #9947183.943 #4143289.841 #9549296.586 #4143289.841 #9549296.586 #250418.6168 #698729.0185 #1.7904931e6 #4143289.841  #9947183.943 #for Todd 7.29e6 #9947183.943  #9549296.586 #9947183.943  #698729.0185 #1.7904931e6 #4143289.841 #9549296.586  #9947183.943 #4143289.841 #698729.0185 #9947183.943 #7.29e6 #4143289.841 #250418.6168 #1.7904931e6 #698729.0185 #1.7904931e6 # #9947183.943 #9549296.586 #4143289.841 #250418.6168 #698729.0185 #9549296.586 #4143289.841 #1.7904931e6
  self.P_gennom = self.machine_rating
  E_pnom   = 3000
   #10 #16 #10 #12.1 #10 #28.6 #20.5 #16 #12.1 #9.6 #7.2 #9.6 #12.1 #20.5 #9.6 #612.1 # # 16 #16 #12.1 #9.6 #10 #12.1 #28.6 #12.1 #12.1 #16
  
  gear      =1 
    
  self.K_load =1
  B_curve=(0.000,0.050,0.130,0.152,0.176,0.202,0.229,0.257,0.287,0.318,0.356,0.395,0.435,0.477,0.523,0.574,0.627,0.683,0.739,0.795,0.850,0.906,0.964,1.024,1.084,1.140,1.189,1.228,1.258,1.283,1.304,1.324,1.343,1.360,1.375,1.387,1.398,1.407,1.417,1.426,1.437,1.448,1.462,1.478,1.495,1.513,1.529,1.543,1.557,1.570,1.584,1.600,1.617,1.634,1.652,1.669,1.685,1.701,1.717,1.733,1.749,1.769,1.792,1.820,1.851,1.882,1.909,1.931,1.948,1.961,1.972,1.981)
  H_curve=(0,10,20,23,25,28,30,33,35,38,40,43,45,48,52,56,60,66,72,78,87,97,112,130,151,175,200,225,252,282,317,357,402,450,500,550,602,657,717,784,867,974,1117,1299,1515,1752,2000,2253,2521,2820,3167,3570,4021,4503,5000,5503,6021,6570,7167,7839,8667,9745,11167,12992,15146,17518,20000,22552,25417,28906,33333,38932)
  h_interp=interp1d(B_curve,H_curve,fill_value='extrapolate')
  U_Nrated=3.3e3
  # rotor structure      
  
  l					= self.l_s                          #l-stator core length
  l_u       =k_fes * self.l_s                   #useful iron stack length
  We				=self.tau_p
  l_b       = 2*self.tau_p  #end winding length
  l_e       =self.l_s+2*0.001*self.r_s     # equivalent core length
  self.b_m  =0.7*self.tau_p 
  constant=0.5
  a_r				= (self.b_r*self.d_r)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr))  # cross-sectional area of rotor armms
  a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws))
  A_r				= l*self.t 
  A_st      =l*self.t_s                     # cross-sectional area of rotor cylinder
  N_r				= round(self.n_r)
  N_st			= round(self.n_s)
  theta_r		=pi*1/N_r                             # half angle between spokes
  theta_s		=pi*1/N_st 
  I_r				=l*self.t**3/12                         # second moment of area of rotor cylinder
  I_st       =l*self.t_s**3/12 
  I_arm_axi_r	=((self.b_r*self.d_r**3)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr)**3))/12  # second moment of area of rotor arm
  I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
  I_arm_tor_r	= ((self.d_r*self.b_r**3)-((self.d_r-2*self.t_wr)*(self.b_r-2*self.t_wr)**3))/12  # second moment of area of rotot arm w.r.t torsion
  I_arm_tor_s	= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
  dia				=  2*self.r_s              # air gap diameter
  g         =  0.001*dia
  
  R					= self.r_s-g-self.h_m-0.5*self.t
  c					=R/500
  R_1				= R-self.t*0.5      
  K					=4*(sin(ratio*pi/2))/pi
  self.R_out=(R/0.995+self.h_s+self.h_ys)
  k_1				= sqrt(I_r/A_r)                               # radius of gyration
  k_2       = sqrt(I_st/A_st)
  m1				=(k_1/R)**2 
                                 
  l_ir			=R                                      # length of rotor arm beam at which rotor cylinder acts
  l_iir			=R_1 
  R_st 			=self.r_s+self.h_s+self.h_ys*0.5
  self.b_all_r		=2*pi*self.R_o/N_r 
  self.b_all_s		=2*pi*self.R_o/N_st                           
  
  K					=4*(sin(ratio*pi/2))/pi
 
  
  b_s1			=  0.003
  self.p		=  round(pi*dia/(2*self.tau_p))
  self.f    =  self.n_nom*self.p/60
  self.S				= 2*self.p*q1*m 
  N_conductors=self.S*2
  self.N_s=N_conductors/2/3
  tau_s=pi*dia/self.S
  
  self.b_s	=  b_s_tau_s*tau_s    #slot width 
  self.b_t	=  tau_s-(self.b_s)          #tooth width
  self.Slot_aspect_ratio=self.h_s/self.b_s
  k_t       =self.b_t/tau_s
  b_so			=  0.004
  b_cu			=  self.b_s -2*h_i         # conductor width
  gamma			=  4/pi*(b_so/2/(g+self.h_m/mu_r)*atan(b_so/2/(g+self.h_m/mu_r))-log(sqrt(1+(b_so/2/(g+self.h_m/mu_r))**2)))
  k_C				=  tau_s/(tau_s-gamma*(g+self.h_m/mu_r))   # carter coefficient
  g_eff			=  k_C*(g+self.h_m/mu_r)                   
  om_m			=  gear*2*pi*self.n_nom/60
  
  om_e			=  self.p*om_m/2
  
  alpha_p		=  pi/2*0.7
  self.B_pm1	 		=  B_r*self.h_m/mu_r/(g_eff)
  
  self.B_g=  B_r*self.h_m/mu_r/(g_eff)*(4/pi)*sin(alpha_p)
  self.B_symax=self.B_g*self.b_m*l_e/(2*self.h_ys*l_u)
  self.B_rymax=self.B_g*self.b_m*l_e/(2*self.h_yr*l)
  
  self.B_tmax	=self.B_g*tau_s/self.b_t
  q3					= self.B_g**2/2/mu_0   # normal stress
  m2        =(k_2/R_st)**2   
  c1        =R_st/500
  R_1s      = R_st-self.t_s*0.5
  d_se=dia+2*(self.h_ys+self.h_s+h_w)  # stator outer diameter
  
  self.mass_PM   =(2*pi*(R+0.5*self.t)*l*self.h_m*ratio*self.rho_PM)           # magnet mass
  
                                             
  w_r					=self.rho_Fes*g1*sin(phi)*a_r*N_r
  

  mass_st_lam=self.rho_Fe*2*pi*(R+0.5*self.t)*l*self.h_yr                                     # mass of rotor yoke steel  
  W				=g1*sin(phi)*(mass_st_lam/N_r+(self.mass_PM)/N_r)  # weight of rotor cylinder
  Numer=R**3*((0.25*(sin(theta_r)-(theta_r*cos(theta_r)))/(sin(theta_r))**2)-(0.5/sin(theta_r))+(0.5/theta_r))
  Pov=((theta_r/(sin(theta_r))**2)+1/tan(theta_r))*((0.25*R/A_r)+(0.25*R**3/I_r))
  Qov=R**3/(2*I_r*theta_r*(m1+1))
  Lov=(R_1-self.R_o)/a_r
  Denom=I_r*(Pov-Qov+Lov) # radial deflection % rotor
  
  
  self.u_Ar				=(q3*R**2/E/self.t)*(1+Numer/Denom)
  
  self.u_all_r    =c/20 # allowable radial deflection
  self.u_all_s    = c1/20
  y_a1=(W*l_ir**3/12/E/I_arm_axi_r)
  y_a2=(w_r*l_iir**4/24/E/I_arm_axi_r) 

  self.y_Ar       =y_a1+y_a2 # axial deflection
  self.y_all     =2*l/100    # allowable axial deflection

  
  self.z_all_s     =0.05*2*pi*R_st/360  # allowable torsional deflection
  self.z_all_r     =0.05*2*pi*R/360  # allowable torsional deflection
  
  self.z_A_r       =(2*pi*(R-0.5*self.t)*l/N_r)*sigma*(l_ir-0.5*self.t)**3/3/E/I_arm_tor_r       # circumferential deflection
  
  val_str_cost_rotor=self.C_PM*self.mass_PM+self.C_Fe*(mass_st_lam)+self.C_Fes*(N_r*(R_1-self.R_o)*a_r*self.rho_Fes)
  val_str_rotor		= self.mass_PM+((mass_st_lam)+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes))
  
  
  self.Rotor_delta_radial=self.u_Ar
  self.Rotor_delta_axial=self.y_Ar
  self.Rotor_circum=self.z_A_r

  
  r_m     	=  self.r_s+h_sy0+self.h_ys+self.h_s #magnet radius
  
  r_r				=  self.r_s-g             #rotor radius
  
  self.K_rad=self.l_s/dia 

  

  v_ys=constant*(self.tau_p+pi*(self.h_s+0.5*self.h_ys)/self.p)*h_interp(self.B_symax)
  v_d=h_interp(self.B_tmax)*(h_s3+0.5*h_s2)+h_interp(self.B_tmax)*(0.5*h_s2+h_s1)
  v_yr=constant*(self.tau_p+pi*(g+self.h_m+0.5*self.h_yr)/self.p)*h_interp(self.B_rymax)
  v_m=self.h_m*self.B_g/mu_r/mu_0
  v_g       =  g_eff*self.B_g/mu_0
  
  # stator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  k_wd			= sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
  beta_skew	= tau_s/self.r_s;
  k_wskew		= sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  L_t=self.l_s+2*self.tau_p
  
  self.E_p	= 2*(self.N_s)*L_t*self.r_s*k_wd*k_wskew*om_m*self.B_g/sqrt(2)
  
  l_Cus			= 2*(self.N_s)*(2*self.tau_p+L_t)
  A_s				= self.b_s*(self.h_s-h_w)*q1*self.p
  A_scalc   = self.b_s*1000*(self.h_s*1000-h_w*1000)*q1*self.p
  A_Cus			= A_s*k_sfil/(self.N_s)
  self.A_Cuscalc = A_scalc *k_sfil/(self.N_s)
  self.R_s	= l_Cus*rho_Cu/A_Cus

  L_m				= 2*m*k_wd**2*(self.N_s)**2*mu_0*self.tau_p*L_t/pi**2/g_eff/self.p
  L_ssigmas=2*mu_0*self.l_s*self.N_s**2/self.p/q1*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=(2*mu_0*self.l_s*self.N_s**2/self.p/q1)*0.34*g*(l_e-0.64*self.tau_p*y_tau_p)/self.l_s                                #end winding leakage inductance
  L_ssigmag=2*mu_0*self.l_s*self.N_s**2/self.p/q1*(5*(g*k_C/b_so)/(5+4*(g*k_C/b_so))) # tooth tip leakage inductance#tooth tip leakage inductance
  L_ssigma	= (L_ssigmas+L_ssigmaew+L_ssigmag)
  self.L_s  = L_m+L_ssigma
  Z=(self.P_gennom/(m*self.E_p))
  
  #self.I_s=Z
  G=(self.E_p**2-(om_e*self.L_s*Z)**2)
   
  self.I_s= sqrt(Z**2+(((self.E_p-G**0.5)/(om_e*self.L_s)**2)**2))
  
  self.J_s	= self.I_s/self.A_Cuscalc

  I_snom		=(self.P_gennom/m/self.E_p/cofi) #rated current

  I_qnom		=self.P_gennom/(m*self.E_p)
  X_snom		=om_e*(L_m+L_ssigma)
  
  self.B_smax=sqrt(2)*self.I_s*mu_0/g_eff
  pp=1  #number of parallel paths
  N_s_j=sqrt(3)*(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*self.J_s*pp*U_Nrated/self.P_gennom
  N_s_s=(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*0.5
  
  self.A_1 = 6*self.N_s*self.I_s/(pi*dia)
  N_s_EL=self.A_1*pi*(dia-g)/(q1*self.p*m)
  self.N_s_max=min([N_s_j,N_s_s,N_s_EL])
  

  #" masses %%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%"
  V_Cus 	=m*l_Cus*A_Cus     # copper volume
  V_Fest	=L_t*2*self.p*q1*m*self.b_t*self.h_s   # volume of iron in stator tooth
  V_Fesy	=L_t*pi*((self.r_s+self.h_s+self.h_ys+h_sy0)**2-(self.r_s+self.h_s)**2) # volume of iron in stator yoke
  V_Fery	=L_t*pi*((r_r-self.h_m)**2-(r_r-self.h_m-self.h_yr)**2)
  
  self.M_Cus		=V_Cus*self.rho_Copper
  self.M_Fest	=V_Fest*self.rho_Fe
  self.M_Fesy	=V_Fesy*self.rho_Fe
  self.M_Fery	=V_Fery*self.rho_Fe
  M_Fe		=self.M_Fest+self.M_Fesy+self.M_Fery
  M_gen		=(self.M_Cus)
  K_gen		=self.M_Cus*self.C_Cu
  
  mass_st_lam_s= self.M_Fest+pi*l*self.rho_Fe*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2) 
  W_is			=0.5*g1*sin(phi)*(self.rho_Fes*l*self.d_s**2) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
  W_iis     =g1*sin(phi)*(mass_st_lam_s+V_Cus*self.rho_Copper)/2/N_st
  w_s         =self.rho_Fes*g1*sin(phi)*a_s*N_st
  
  l_is      =R_st-self.R_o
  l_iis     =l_is 
  l_iiis    =l_is
    

  
  
  #stator structure deflection calculation
  mass_stru_steel  =2*(N_st*(R_1s-self.R_o)*a_s*self.rho_Fes)
  Numers=R_st**3*((0.25*(sin(theta_s)-(theta_s*cos(theta_s)))/(sin(theta_s))**2)-(0.5/sin(theta_s))+(0.5/theta_s))
  Povs=((theta_s/(sin(theta_s))**2)+1/tan(theta_s))*((0.25*R_st/A_st)+(0.25*R_st**3/I_st))
  Qovs=R_st**3/(2*I_st*theta_s*(m2+1))
  Lovs=(R_1s-self.R_o)*0.5/a_s
  Denoms=I_st*(Povs-Qovs+Lovs) 
 
  self.u_As				=(q3*R_st**2/E/self.t_s)*(1+Numers/Denoms)
  X_comp1 = (W_is*l_is**3/12/E/I_arm_axi_s)
  X_comp2 =(W_iis*l_iis**4/24/E/I_arm_axi_s)
  X_comp3 =w_s*l_iiis**4/24/E/I_arm_axi_s
  
  self.y_As       =X_comp1+X_comp2+X_comp3  # axial deflection
  self.z_A_s  =2*pi*(R_st+0.5*self.t_s)*l/(2*N_st)*sigma*(l_is+0.5*self.t_s)**3/3/E/I_arm_tor_s 
   #val_str_stator		= mass_stru_steel+mass_st_lam_s+((sign(u_As-u_all)+1)*(u_As-u_all)**3*1e100)+(((sign(y_As-y_all)+1)*(y_As-y_all)**3*1e100))+(((sign(z_As-z_all_s)+1)*(z_As-z_all_s)**3*1e100))+(((sign(self.b_st-b_all_s)+1)*(self.b_st-b_all_s)**3*1e50))+((sign((T/2/pi/sigma)-(R_st**2*l))+1)*((T/2/pi/sigma)-(R_st**2*l))**3*1e50)
  val_str_stator		= mass_stru_steel+mass_st_lam_s
  
  val_str_cost_stator =	self.C_Fes*mass_stru_steel+self.C_Fe*mass_st_lam_s
    
  val_str_mass=val_str_rotor+val_str_stator
  
  
  self.Stator_delta_radial=self.u_As
  self.Stator_delta_axial=self.y_As
  self.Stator_circum=self.z_A_s
  self.TC1=T/(2*pi*sigma)     # Desired shear stress 
  self.TC2=R**2*l              # Evaluating Torque constraint for rotor
  self.TC3=R_st**2*l           # Evaluating Torque constraint for stator
  
  self.Iron=mass_st_lam_s+(2*pi*t*l*(R+0.5*self.t)*self.rho_Fe)
  self.PM=self.mass_PM
  self.Copper=self.M_Cus
  self.Inactive=mass_stru_steel+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes)
  
  self.Stator=mass_st_lam_s+mass_stru_steel+self.Copper
  self.Rotor=((2*pi*t*l*(R+0.5*self.t)*self.rho_Fe)+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes))+self.PM
  self.M_actual	=self.Stator+self.Rotor
  self.Mass = self.M_actual
  self.Active=self.Iron+self.Copper+self.mass_PM
 #"% losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  P_Cu		=m*self.I_s**2*self.R_s
  
  K_R=1.2
  P_Sc=m*(self.R_s)*K_R*(I_snom)**2*1  #losses to include skin effect
  
  
  if (self.K_load==0):
  	P_Cusnom_total=P_Cu
  else:
    P_Cusnom_total=P_Sc*self.K_load**2
  
  
  
  
  #B_tmax	=B_pm*tau_s/self.b_t
  P_Hyys	=self.M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
  P_Ftys	=self.M_Fesy*((self.B_symax/1.5)**2)*(P_Fe0e*(om_e/(2*pi*60))**2)
  P_Fesynom=P_Hyys+P_Ftys
  P_Hyd=self.M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
  P_Ftd=self.M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)
  P_Festnom=P_Hyd+P_Ftd
  P_ad=0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd ) # additional stray losses due to leakage flux
  pFtm =300 # specific magnet loss
  P_Ftm=pFtm*2*self.p*self.b_m*self.l_s
  
  P_genlossnom=P_Cusnom_total+P_Festnom+P_Fesynom+P_ad+P_Ftm

  self.gen_eff=self.P_gennom*100/(self.P_gennom+P_genlossnom)
  
  self.Losses=P_genlossnom
  
  
  #"% cost of losses"
  val_str_cost=val_str_cost_rotor+val_str_cost_stator
  self.Costs=val_str_cost+K_gen
  
  self.I[0]   = (0.5*self.M_actual*self.R_out**2)
  self.I[1]   = (0.25*self.M_actual*self.R_out**2+(1/12)*self.M_actual*self.l_s**2) 
  self.I[2]   = self.I[1]
  cm[0]  = self.main_shaft_cm[0] + self.main_shaft_length/2. + self.l_s/2
  cm[1]  = self.main_shaft_cm[1]
  cm[2]  = self.main_shaft_cm[2]
  
  #print ((self.mass_PM))*g1,  (self.M_Fest+V_Cus*self.rho_Copper)*g1,q3
  
  
  

class Drive_PMSG(Assembly):
	Eta_target=Float(iotype='in', desc='Target drivetrain efficiency')
	T_rated=Float(iotype='in', desc='Torque')
	N=Float(iotype='in', desc='rated speed')
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
	I = Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
	cm = Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc=' Center of mass [x, y,z]')
	PMSG_r_s= Float(iotype='in', units='m', desc='Air gap radius of a permanent magnet excited synchronous generator')
	PMSG_l_s= Float(iotype='in', units='m', desc='Core length of the permanent magnet excited synchronous generator')
	PMSG_h_s = Float(iotype='in', units='m', desc='Stator Slot height of the permanent magnet excited synchronous generator')
	PMSG_tau_p = Float(iotype='in', units='m', desc='Pole pitch of the permanent magnet excited synchronous generator')
	PMSG_h_m = Float(iotype='in', units='m', desc='Magnet height of the permanent magnet excited synchronous generator')
	PMSG_h_ys = Float(iotype='in', units='m', desc='Stator yoke height of the permanent magnet excited synchronous generator')
	PMSG_h_yr = Float(iotype='in', units='m', desc='Rotor yoke height of the permanent magnet excited synchronous generator')
	PMSG_n_s = Float(iotype='in', units='m', desc='Stator Spokes of the permanent magnet excited synchronous generator')
	PMSG_b_st = Float(iotype='in', units='m', desc='Circumferential arm dimension of stator spoke')
	PMSG_n_r =Float(iotype='in', desc='Number of Rotor Spokes of the permanent magnet excited synchronous generator')
	PMSG_b_r = Float(iotype='in', desc='Circumferential arm dimension of the rotor spoke')
	PMSG_d_r = Float(iotype='in', desc='Rotor arm depth')
	PMSG_d_s= Float(iotype='in', desc='Stator arm depth ')
	PMSG_t_wr =Float(iotype='in', desc='Rotor arm thickness')
	PMSG_t_ws =Float(iotype='in', desc='Stator arm thickness')
	PMSG_R_o =Float(iotype='in', desc='Main shaft radius')
	
	C_Cu=Float( iotype='in', desc='Specific cost of copper')
	C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
	C_PM=Float(iotype='in', desc='Specific cost of Magnet')
	
	rho_Fes=Float(iotype='in', desc='Structural Steel density kg/m^3')
	rho_Fe=Float(iotype='in', desc='Magnetic Steel density kg/m^3')
	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
	rho_PM=Float(iotype='in', desc='Magnet density kg/m^3')
 	
 	
  
	
	def __init__(self,Optimiser='',Objective_function=''):
		
		super(Drive_PMSG,self).__init__()
		self.Optimiser=Optimiser
		self.Objective_function=Objective_function
		""" Creates a new Assembly containing PMSG and an optimizer"""
		self.add('PMSG',PMSG())
		self.connect('PMSG_r_s','PMSG.r_s')
		self.connect('PMSG_l_s','PMSG.l_s')
		self.connect('PMSG_h_s','PMSG.h_s')
		self.connect('PMSG_tau_p','PMSG.tau_p')
		self.connect('PMSG_h_m','PMSG.h_m')
		self.connect('PMSG_h_ys','PMSG.h_ys')
		self.connect('PMSG_h_yr','PMSG.h_yr')
		self.connect('PMSG_n_s','PMSG.n_s')
		self.connect('PMSG_b_st','PMSG.b_st')
		self.connect('PMSG_d_s','PMSG.d_s')
		self.connect('PMSG_t_ws','PMSG.t_ws')
		self.connect('PMSG_n_r','PMSG.n_r')
		self.connect('PMSG_b_r','PMSG.b_r')
		self.connect('PMSG_d_r','PMSG.d_r')
		self.connect('PMSG_t_wr','PMSG.t_wr')
		self.connect('PMSG_R_o','PMSG.R_o')
		self.connect('P_rated','PMSG.machine_rating')
		self.connect('N','PMSG.n_nom')
		self.connect('main_shaft_cm','PMSG.main_shaft_cm')
		self.connect('main_shaft_length','PMSG.main_shaft_length')
		self.connect('T_rated','PMSG.Torque')
		self.connect('PMSG.M_actual','Mass')
		self.connect('PMSG.gen_eff','Efficiency')
		self.connect('PMSG.r_s','r_s')
		self.connect('PMSG.l_s','l_s')
		self.connect('PMSG.I','I')
		self.connect('PMSG.cm','cm')
		
		self.connect('C_Fe','PMSG.C_Fe')
		self.connect('C_Fes','PMSG.C_Fes')
		self.connect('C_Cu','PMSG.C_Cu')
		self.connect('C_PM','PMSG.C_PM')
		
		self.connect('rho_PM','PMSG.rho_PM')
		self.connect('rho_Fe','PMSG.rho_Fe')
		self.connect('rho_Fes','PMSG.rho_Fes')
		self.connect('rho_Copper','PMSG.rho_Copper')

				
		
		
		Opt1=globals()[self.Optimiser]
		self.add('driver',Opt1())
		self.driver.iprint = 1 
		if (Opt1=='CONMINdriver'):
			#Create Optimizer instance
			self.driver.itmax = 100
			self.driver.fdch = 0.01
			self.driver.fdchm = 0.01
			self.driver.ctlmin = 0.01
			self.driver.delfun = 0.001
			self.driver.conmin_diff = True
		elif (Opt1=='COBYLAdriver'):
			# COBYLA-specific Settings
			self.driver.rhobeg=1.0
			self.driver.rhoend = 1.0e-4
			self.driver.maxfun = 1000
		elif (Opt1=='SLSQPdriver'):
			# SLSQP-specific Settings
			self.driver.accuracy = 1.0e-6
			self.driver.maxiter = 50
		elif (Opt1=='Genetic'):
			# Genetic-specific Settings
			self.driver.population_size = 90
			self.driver.crossover_rate = 0.9
			self.driver.mutation_rate = 0.02
			self.selection_method = 'rank'
		else:
			# NEWSUMT-specific Settings
			self.driver.itmax = 10 
		
			
		Obj1='PMSG'+'.'+self.Objective_function
		self.driver.add_objective(Obj1)
		self.driver.design_vars=['PMSG_r_s','PMSG_l_s','PMSG_h_s','PMSG_tau_p','PMSG_h_m','PMSG_h_ys','PMSG_h_yr','PMSG_n_r','PMSG_n_s','PMSG_b_r','PMSG_b_st','PMSG_d_r','PMSG_d_st','PMSG_t_wr','PMSG_t_ws']
		self.driver.add_parameter('PMSG_r_s', low=0.5, high=9)
		self.driver.add_parameter('PMSG_l_s', low=0.5, high=2.5)
		self.driver.add_parameter('PMSG_h_s', low=0.04, high=0.1)
		self.driver.add_parameter('PMSG_tau_p', low=0.04, high=0.1)
		self.driver.add_parameter('PMSG_h_m', low=0.005, high=0.1)
		self.driver.add_parameter('PMSG_n_r', low=5., high=15.)
		self.driver.add_parameter('PMSG_h_yr', low=0.045, high=0.25)
		self.driver.add_parameter('PMSG_h_ys', low=0.045, high=0.25)
		self.driver.add_parameter('PMSG_b_r', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_d_r', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_t_wr', low=0.001, high=0.2)
		self.driver.add_parameter('PMSG_n_s', low=5., high=15.)
		self.driver.add_parameter('PMSG_b_st', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_d_s', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_t_ws', low=0.001, high=0.2)
		
		self.driver.add_constraint('PMSG.B_symax<2')										  #1
		self.driver.add_constraint('PMSG.B_rymax<2')										  #2
		self.driver.add_constraint('PMSG.B_tmax<2')									      #3
		self.driver.add_constraint('PMSG.B_smax<PMSG.B_g') 								#4
		self.driver.add_constraint('PMSG.B_g>=0.7')  											#5                
		self.driver.add_constraint('PMSG.B_g<=1.2') 											#6
		self.driver.add_constraint('PMSG.E_p>=500')											  #7
		self.driver.add_constraint('PMSG.E_p<=5000')											#8
		self.driver.add_constraint('PMSG.u_As<PMSG.u_all_s')							#9
		self.driver.add_constraint('PMSG.z_A_s<PMSG.z_all_s')							#10
		self.driver.add_constraint('PMSG.y_As<PMSG.y_all')  							#11
		self.driver.add_constraint('PMSG.u_Ar<PMSG.u_all_r')							#12
		self.driver.add_constraint('PMSG.z_A_r<PMSG.z_all_r')							#13
		self.driver.add_constraint('PMSG.y_Ar<PMSG.y_all') 								#14
		self.driver.add_constraint('PMSG.TC1<PMSG.TC2')    								#15
		self.driver.add_constraint('PMSG.TC1<PMSG.TC3')    								#16
		self.driver.add_constraint('PMSG.b_r<PMSG.b_all_r')								#17
		self.driver.add_constraint('PMSG.b_st<PMSG.b_all_s')							#18
		self.driver.add_constraint('PMSG.A_1<60000')											#19
		self.driver.add_constraint('PMSG.J_s<=6') 												#20
		self.driver.add_constraint('PMSG.A_Cuscalc>=5') 									#21
		self.driver.add_constraint('PMSG.K_rad>0.2')											#22
		self.driver.add_constraint('PMSG.K_rad<=0.27')										#23 
		self.driver.add_constraint('PMSG.Slot_aspect_ratio>=4')						#24
		self.driver.add_constraint('PMSG.Slot_aspect_ratio<=10')					#25	
		self.driver.add_constraint('PMSG.gen_eff>=Eta_target')			#26
		
				
def optim_PMSG():
	opt_problem = Drive_PMSG('CONMINdriver','Costs')
#	opt_problem.Eta_target = 93
#	# Initial design variables for a DD PMSG designed for a 5MW turbine
#	opt_problem.P_rated=5.0e6
#	opt_problem.T_rated=4.143289e6
#	opt_problem.N=12.1
#	opt_problem.PMSG_r_s= 3.26               
#	opt_problem.PMSG_l_s= 1.6
#	opt_problem.PMSG_h_s = 0.07
#	opt_problem.PMSG_tau_p = 0.08
#	opt_problem.PMSG_h_m = 0.009
#	opt_problem.PMSG_h_ys = 0.0615
#	opt_problem.PMSG_h_yr = 0.057
#	opt_problem.PMSG_n_s = 5
#	opt_problem.PMSG_b_st = 0.480
#	opt_problem.PMSG_n_r =5
#	opt_problem.PMSG_b_r = 0.520
#	opt_problem.PMSG_d_r = 0.7
#	opt_problem.PMSG_d_s= 0.3
#	opt_problem.PMSG_t_wr =0.06
#	opt_problem.PMSG_t_ws =0.06
#	opt_problem.PMSG_R_o =0.43      #0.523950817  #0.43  #0.523950817 #0.17625 #0.2775 #0.363882632 ##0.35 #0.523950817 #0.43 #523950817 #0.43 #0.523950817 #0.523950817 #0.17625 #0.2775 #0.363882632 #0.43 #0.523950817 #0.43
#	opt_problem.run()
#	
	

	opt_problem.Eta_target = 93
	# Initial design variables for a DD PMSG designed for a 10MW turbine
	opt_problem.P_rated=0.75e6
	opt_problem.T_rated=250418.6168
	opt_problem.N=28.6
	opt_problem.PMSG_r_s=1.3             
	opt_problem.PMSG_l_s= 0.71
	opt_problem.PMSG_h_s = 0.045
	opt_problem.PMSG_tau_p = 0.06
	opt_problem.PMSG_h_m = 0.009
	opt_problem.PMSG_h_ys = 0.045
	opt_problem.PMSG_h_yr = 0.045
	opt_problem.PMSG_n_s = 5
	opt_problem.PMSG_b_st = 0.220
	opt_problem.PMSG_n_r =5
	opt_problem.PMSG_b_r = 0.220
	opt_problem.PMSG_d_r = 0.4
	opt_problem.PMSG_d_s= 0.18
	opt_problem.PMSG_t_wr =0.01
	opt_problem.PMSG_t_ws =0.01
	opt_problem.PMSG_R_o =0.17625                    # 1.5MW: 0.2775; 3MW: 0.363882632; 10 MW:0.523950817; 0.75MW :0.17625
	
	# Specific costs
	opt_problem.C_Cu   =4.786                  
	opt_problem.C_Fe	= 0.556                   
	opt_problem.C_Fes =0.50139                   
	opt_problem.C_PM  =95
	
	#Material properties
	opt_problem.rho_Fe = 7700                 #Steel density
	opt_problem.rho_Fes = 7850                 #Steel density
	opt_problem.rho_Copper =8900                  # Kg/m3 copper density
	opt_problem.rho_PM =7450
	
	
	
	
	
	opt_problem.run()
	raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor arms','Rotor Axial arm dimension','Rotor Circumferential arm dimension' ,'Rotor arm Thickness',' Stator Radial deflection', 'Stator Axial deflection','Stator circum deflection',' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Air gap diameter','Overall Outer diameter', 'Stator length', 'l/d ratio','Slot_aspect_ratio','Pole pitch', 'Stator slot height','Stator slotwidth','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental','Peak stator yoke flux density','Peak rotor yoke flux density','Flux density above magnet','Maximum Stator flux density','Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage', 'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns','Conductor cross-section','Stator Current density ','Specific current loading','Generator Efficiency ','Iron mass','Magnet mass','Copper mass','Mass of Arms', 'Total Mass', 'Stator Mass','Rotor Mass','Total Material Cost'],
			'Values': [opt_problem.PMSG.P_gennom/1000000,opt_problem.PMSG.n_s,opt_problem.PMSG.d_s*1000,opt_problem.PMSG.b_st*1000,opt_problem.PMSG.t_ws*1000,opt_problem.PMSG.n_r,opt_problem.PMSG.d_r*1000,opt_problem.PMSG.b_r*1000,opt_problem.PMSG.t_wr*1000,opt_problem.PMSG.Stator_delta_radial*1000,opt_problem.PMSG.Stator_delta_axial*1000,opt_problem.PMSG.Stator_circum*1000,opt_problem.PMSG.Rotor_delta_radial*1000,opt_problem.PMSG.Rotor_delta_axial*1000,opt_problem.PMSG.Rotor_circum*1000,2*opt_problem.PMSG.r_s,opt_problem.PMSG.R_out*2,opt_problem.PMSG.l_s,opt_problem.PMSG.K_rad,opt_problem.PMSG.Slot_aspect_ratio,opt_problem.PMSG.tau_p*1000,opt_problem.PMSG.h_s*1000,opt_problem.PMSG.b_s*1000,opt_problem.PMSG.b_t*1000,opt_problem.PMSG.t_s*1000,opt_problem.PMSG.t*1000,opt_problem.PMSG.h_m*1000,opt_problem.PMSG.b_m*1000,opt_problem.PMSG.B_g,opt_problem.PMSG.B_symax,opt_problem.PMSG.B_rymax,opt_problem.PMSG.B_pm1,opt_problem.PMSG.B_smax,opt_problem.PMSG.B_tmax,opt_problem.PMSG.p,opt_problem.PMSG.f,opt_problem.PMSG.E_p,opt_problem.PMSG.I_s,opt_problem.PMSG.R_s,opt_problem.PMSG.L_s,opt_problem.PMSG.S,opt_problem.PMSG.N_s,opt_problem.PMSG.A_Cuscalc,opt_problem.PMSG.J_s,opt_problem.PMSG.A_1/1000,opt_problem.PMSG.gen_eff,opt_problem.PMSG.Iron/1000,opt_problem.PMSG.mass_PM/1000,opt_problem.PMSG.M_Cus/1000,opt_problem.PMSG.Inactive/1000,opt_problem.PMSG.M_actual/1000,opt_problem.PMSG.Stator/1000,opt_problem.PMSG.Rotor/1000,opt_problem.PMSG.Costs/1000],
				'Limit': ['','','',opt_problem.PMSG.b_all_s*1000,'','','',opt_problem.PMSG.b_all_r*1000,'',opt_problem.PMSG.u_all_s*1000,opt_problem.PMSG.y_all*1000,opt_problem.PMSG.z_all_s*1000,opt_problem.PMSG.u_all_r*1000,opt_problem.PMSG.y_all*1000,opt_problem.PMSG.z_all_r*1000,'','','','(0.2-0.27)','(4-10)','','','','','','','','','','<2','<2','<2',opt_problem.PMSG.B_g,'','','','','>500','','','','','5','3-6','60','>93%','','','','','','','',''],
				'Units':['MW','unit','mm','mm','mm','mm','mm','','mm','mm','mm','mm','mm','mm','mm','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','A/mm^2','slots','turns','mm^2','kA/m','%','tons','tons','tons','tons','tons','ton','ton','k$']}
	df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print df
	df.to_excel('PMSG_'+str(opt_problem.P_rated/1e6)+'_arms_MW.xlsx')


		
if __name__=="__main__":
=======
"""PMDD.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.main.api import Assembly
from openmdao.lib.datatypes.api import Float,Array,Str
import numpy as np
from numpy import array
from numpy import float 
from numpy import min
from scipy.interpolate import interp1d
from openmdao.lib.drivers.api import COBYLAdriver, CONMINdriver,NEWSUMTdriver,SLSQPdriver,Genetic
import pandas as pd


class PMSG(Component):

 """ Evaluates the total cost """
 
 r_s = Float(iotype='in', desc='airgap radius r_s')
 l_s = Float(iotype='in', desc='Stator core length l_s')
 h_s = Float(iotype='in', desc='Yoke height h_s')
 tau_p =Float(iotype='in', desc='Pole pitch self.tau_p')
 B_g = Float(iotype='out', desc='Peak air gap flux density B_g')
 machine_rating=Float(iotype='in', desc='Machine rating')
 n_nom=Float(iotype='in', desc='rated speed')
 Torque=Float(iotype='in', desc='Rated torque ')
 h_m=Float(iotype='in', desc='magnet height')
 h_ys=Float(iotype='in', desc='Yoke height')
 h_yr=Float(iotype='in', desc='rotor yoke height')
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
 
 # Costs and material properties
 C_Cu=Float( iotype='in', desc='Specific cost of copper')
 C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
 C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
 C_PM=Float(iotype='in', desc='Specific cost of Magnet')
 
 rho_Fes=Float(iotype='in', desc='Structural Steel density kg/m^3')
 rho_Fe=Float(iotype='in', desc='Magnetic Steel density kg/m^3')
 rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
 rho_PM=Float(iotype='in', desc='Magnet density kg/m^3')
 
 # Magnetic loading
 B_symax = Float(iotype='out', desc='Peak Stator Yoke flux density B_ymax')
 B_tmax=Float(iotype='out', desc='Peak Teeth flux density')
 B_rymax=Float(iotype='out', desc='Peak Rotor yoke flux density')
 B_smax=Float(iotype='out', desc='Peak Stator flux density')
 B_pm1=Float(iotype='out', desc='Fundamental component of peak air gap flux density')
 
 #Stator design
 N_s =Float(iotype='out', desc='Number of turns in the stator winding')
 b_s=Float(iotype='out', desc='slot width')
 b_t=Float(iotype='out', desc='tooth width')
 A_Cuscalc=Float(iotype='out', desc='Conductor cross-section mm^2')
 
 #Rotor magnet dimension
 b_m=Float(iotype='out', desc='magnet width')
 p=Float(iotype='out', desc='No of pole pairs')
  
 # Electrical performance 
 E_p=Float(iotype='out', desc='Stator phase voltage')
 f=Float(iotype='out', desc='Generator output frequency')
 I_s=Float(iotype='out', desc='Generator output phase current')
 R_s=Float(iotype='out', desc='Stator resistance')
 L_s=Float(iotype='out', desc='Stator synchronising inductance')
 A_1 =Float(0,iotype='out', desc='Electrical loading')
 J_s=Float(iotype='out', desc='Current density')
 
 # Objective functions
 M_actual=Float(iotype='out', desc='Actual mass')
 Mass=Float(iotype='out', desc='Actual mass')
 Costs= Float(iotype='out', desc='Total cost')
 K_rad=Float(iotype='out', desc='K_rad')
 Losses=Float(iotype='out', desc='Total loss')
 
 gen_eff=Float(iotype='out', desc='Generator efficiency')
 Active=Float(iotype='out', desc='Generator efficiency')
 Stator=Float(iotype='out', desc='Generator efficiency')
 Rotor=Float(iotype='out', desc='Generator efficiency')
 mass_PM=Float(iotype='out', desc='Generator efficiency')
 M_Cus		=Float(iotype='out', desc='Generator efficiency')
 M_Fest	=Float(iotype='out', desc='Generator efficiency')
 M_Fesy	=Float(iotype='out', desc='Generator efficiency')
 M_Fery	=Float(iotype='out', desc='Generator efficiency')
 Iron=Float(iotype='out', desc='Electrical steel mass')
 
 # Structural performance
 Stator_delta_radial=Float(iotype='out', desc='Rotor radial deflection')
 Stator_delta_axial=Float(iotype='out', desc='Stator Axial deflection')
 Stator_circum=Float(iotype='out', desc='Rotor radial deflection')
 Rotor_delta_radial=Float(iotype='out', desc='Generator efficiency')
 Rotor_delta_axial=Float(iotype='out', desc='Rotor Axial deflection')
 Rotor_circum=Float(iotype='out', desc='Rotor circumferential deflection')
 
 P_gennom=Float(iotype='in', desc='Generator Power')
 u_Ar	=Float(iotype='out', desc='Rotor radial deflection')
 y_Ar =Float(iotype='out', desc='Rotor axial deflection')
 z_A_r=Float(iotype='out', desc='Rotor circumferential deflection')      # circumferential deflection
 u_As=Float(iotype='out', desc='Stator radial deflection')
 y_As =Float(iotype='out', desc='Stator axial deflection')
 z_A_s=Float(iotype='out', desc='Stator circumferential deflection')  
 u_all_r=Float(iotype='out', desc='Allowable radial rotor')
 u_all_s=Float(iotype='out', desc='Allowable radial stator')
 y_all=Float(iotype='out', desc='Allowable axial')
 z_all_s=Float(iotype='out', desc='Allowable circum stator')
 z_all_r=Float(iotype='out', desc='Allowable circum rotor')
 b_all_s=Float(iotype='out', desc='Allowable arm')
 b_all_r=Float(iotype='out', desc='Allowable arm dimensions')
 TC1	=Float(iotype='out', desc='Torque constraint')
 TC2=Float(iotype='out', desc='Torque constraint-rotor')
 TC3=Float(iotype='out', desc='Torque constraint-stator')
 R_out=Float(iotype='out', desc='Outer radius')
 N_s_max=Float(iotype='out', desc='Maximum number of turns per coil')
 S			=Float(iotype='out', desc='Stator slots')
 K_load=Float(iotype='out', desc='Load factor')
 Slot_aspect_ratio=Float(iotype='out', desc='Slot aspect ratio')
 
 main_shaft_cm = Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc='Main Shaft CM')
 main_shaft_length=Float(iotype='in', desc='main shaft length')
 cm=Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='COM [x,y,z]')
 I=Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
 
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  tau_p =self.tau_p
  B_g = self.B_g
  n_r =self.n_r
  n_s=self.n_s
  t= self.t
  t_s=self.t_s
  b_r = self.b_r
  d_r = self.d_r
  t_wr =self.t_wr
  b_st = self.b_st
  d_s = self.d_s
  t_ws =self.t_ws
  b_s =self.b_s
  b_t= self.b_t
  h_ys = self.h_ys
  h_yr  = self.h_yr
  h_m =self.h_m
  b_m =self.b_m
  f=self.f
  E_p=self.E_p
  I_s=self.I_s
  R_s=self.R_s
  L_s=self.L_s
  J_s=self.J_s
  M_actual=self.M_actual
  Mass=self.Mass
  gen_eff =self.gen_eff
  Losses=self.Losses
  A_1=self.A_1
  K_rad=self.K_rad
  N_s=self.N_s
  Active=self.Active
  Stator=self.Stator
  Rotor=self.Rotor
  mass_PM=self.mass_PM
  M_Cus		=self.M_Cus	
  M_Fest	=self.M_Fest
  M_Fesy	=self.M_Fesy
  M_Fery	=self.M_Fery
  Iron=self.Iron
  Stator_delta_radial=self.Stator_delta_radial
  Stator_delta_axial=self.Stator_delta_axial
  Stator_circum=self.Stator_circum
  Rotor_delta_radial=self.Rotor_delta_radial
  Rotor_delta_axial=self.Rotor_delta_axial
  Rotor_circum=self.Rotor_circum
  P_gennom=self.P_gennom
  u_As=self.u_As
  u_Ar=self.u_Ar
  y_As=self.y_As
  y_Ar=self.y_Ar
  z_A_s=self.z_A_s
  z_A_r=self.z_A_r
  z_all_s=self.z_all_s
  z_all_r=self.z_all_r
  R_out=self.R_out
  B_pm1=self.B_pm1
  B_tmax= self.B_tmax
  B_symax=self.B_symax
  B_rymax=self.B_rymax
  B_smax=self.B_smax
  B_pm1=self.B_pm1
  N_s_max=self.N_s_max
  A_Cuscalc=self.A_Cuscalc
  b_all_r=self.b_all_r
  b_all_s=self.b_all_s
  S=self.S
  Stator=self.Stator
  Rotor=self.Rotor
  K_load=self.K_load
  machine_rating=self.machine_rating
  Slot_aspect_ratio=self.Slot_aspect_ratio
  Torque=self.Torque
  main_shaft_cm = self.main_shaft_cm
  main_shaft_length=self.main_shaft_length
  cm=self.cm
  I=self.I
  TC1	=self.TC1
  TC2=self.TC2
  TC3=self.TC3
  R_o=self.R_o
  Costs=self.Costs
  n_nom = self.n_nom
  
  C_Cu=self.C_Cu
  C_Fe=self.C_Fe
  C_Fes=self.C_Fes
  C_PM =self.C_PM
  
  rho_Fe= self.rho_Fe
  rho_Fes=self.rho_Fes
  rho_Copper=self.rho_Copper
  rho_PM=self.rho_PM
  
	
  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign
  
  B_r    =1.2                 # Tesla remnant flux density 
  g1     =9.81                # m/s^2 acceleration due to gravity
  E      =2e11                # N/m^2 young's modulus
  sigma  =40e3                # shear stress assumed
  ratio  =0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
  mu_0   =pi*4e-7              # permeability of free space
  mu_r   =1.06								# relative permeability 
  phi    =90*2*pi/360         # tilt angle (rotor tilt -90 degrees during transportation)
  cofi   =0.85                 # power factor
  
  h_sy0  =0
  h_w    =0.005
  h_i    =0.001 										# coil insulation thickness
  h_s1   = 0.001
  h_s2   =0.004
  h_s3   =self.h_s-h_s1-h_s2
  h_cu   =(h_s3-4*h_i)*0.5
  self.t_s =self.h_ys
  self.t =self.h_yr
  y_tau_p=1
  self.K_rad=self.l_s/(2*self.r_s)
  m      =3                    # no of phases
  B_rmax=1.4                  
  q1     =1                    # no of slots per pole per phase
  b_s_tau_s=0.45
  k_sfil =0.65								 # Slot fill factor
  P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  rho_Cu=1.8*10**(-8)*1.4
  k_fes =0.9
  T =   self.Torque    #549296.586 #4143289.841 #9549296.586 #250418.6168  #698729.0185 #1.7904931e6 # #248679.5986 #9947183.943 #4143289.841 #9549296.586 #4143289.841 #9549296.586 #250418.6168 #698729.0185 #1.7904931e6 #4143289.841  #9947183.943 #for Todd 7.29e6 #9947183.943  #9549296.586 #9947183.943  #698729.0185 #1.7904931e6 #4143289.841 #9549296.586  #9947183.943 #4143289.841 #698729.0185 #9947183.943 #7.29e6 #4143289.841 #250418.6168 #1.7904931e6 #698729.0185 #1.7904931e6 # #9947183.943 #9549296.586 #4143289.841 #250418.6168 #698729.0185 #9549296.586 #4143289.841 #1.7904931e6
  self.P_gennom = self.machine_rating
  E_pnom   = 3000
   #10 #16 #10 #12.1 #10 #28.6 #20.5 #16 #12.1 #9.6 #7.2 #9.6 #12.1 #20.5 #9.6 #612.1 # # 16 #16 #12.1 #9.6 #10 #12.1 #28.6 #12.1 #12.1 #16
  
  gear      =1 
    
  self.K_load =1
  B_curve=(0.000,0.050,0.130,0.152,0.176,0.202,0.229,0.257,0.287,0.318,0.356,0.395,0.435,0.477,0.523,0.574,0.627,0.683,0.739,0.795,0.850,0.906,0.964,1.024,1.084,1.140,1.189,1.228,1.258,1.283,1.304,1.324,1.343,1.360,1.375,1.387,1.398,1.407,1.417,1.426,1.437,1.448,1.462,1.478,1.495,1.513,1.529,1.543,1.557,1.570,1.584,1.600,1.617,1.634,1.652,1.669,1.685,1.701,1.717,1.733,1.749,1.769,1.792,1.820,1.851,1.882,1.909,1.931,1.948,1.961,1.972,1.981)
  H_curve=(0,10,20,23,25,28,30,33,35,38,40,43,45,48,52,56,60,66,72,78,87,97,112,130,151,175,200,225,252,282,317,357,402,450,500,550,602,657,717,784,867,974,1117,1299,1515,1752,2000,2253,2521,2820,3167,3570,4021,4503,5000,5503,6021,6570,7167,7839,8667,9745,11167,12992,15146,17518,20000,22552,25417,28906,33333,38932)
  h_interp=interp1d(B_curve,H_curve,fill_value='extrapolate')
  U_Nrated=3.3e3
  # rotor structure      
  
  l					= self.l_s                          #l-stator core length
  l_u       =k_fes * self.l_s                   #useful iron stack length
  We				=self.tau_p
  l_b       = 2*self.tau_p  #end winding length
  l_e       =self.l_s+2*0.001*self.r_s     # equivalent core length
  self.b_m  =0.7*self.tau_p 
  constant=0.5
  a_r				= (self.b_r*self.d_r)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr))  # cross-sectional area of rotor armms
  a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws))
  A_r				= l*self.t 
  A_st      =l*self.t_s                     # cross-sectional area of rotor cylinder
  N_r				= round(self.n_r)
  N_st			= round(self.n_s)
  theta_r		=pi*1/N_r                             # half angle between spokes
  theta_s		=pi*1/N_st 
  I_r				=l*self.t**3/12                         # second moment of area of rotor cylinder
  I_st       =l*self.t_s**3/12 
  I_arm_axi_r	=((self.b_r*self.d_r**3)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr)**3))/12  # second moment of area of rotor arm
  I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
  I_arm_tor_r	= ((self.d_r*self.b_r**3)-((self.d_r-2*self.t_wr)*(self.b_r-2*self.t_wr)**3))/12  # second moment of area of rotot arm w.r.t torsion
  I_arm_tor_s	= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
  dia				=  2*self.r_s              # air gap diameter
  g         =  0.001*dia
  
  R					= self.r_s-g-self.h_m-0.5*self.t
  c					=R/500
  R_1				= R-self.t*0.5      
  K					=4*(sin(ratio*pi/2))/pi
  self.R_out=(R/0.995+self.h_s+self.h_ys)
  k_1				= sqrt(I_r/A_r)                               # radius of gyration
  k_2       = sqrt(I_st/A_st)
  m1				=(k_1/R)**2 
                                 
  l_ir			=R                                      # length of rotor arm beam at which rotor cylinder acts
  l_iir			=R_1 
  R_st 			=self.r_s+self.h_s+self.h_ys*0.5
  self.b_all_r		=2*pi*self.R_o/N_r 
  self.b_all_s		=2*pi*self.R_o/N_st                           
  
  K					=4*(sin(ratio*pi/2))/pi
 
  
  b_s1			=  0.003
  self.p		=  round(pi*dia/(2*self.tau_p))
  self.f    =  self.n_nom*self.p/60
  self.S				= 2*self.p*q1*m 
  N_conductors=self.S*2
  self.N_s=N_conductors/2/3
  tau_s=pi*dia/self.S
  
  self.b_s	=  b_s_tau_s*tau_s    #slot width 
  self.b_t	=  tau_s-(self.b_s)          #tooth width
  self.Slot_aspect_ratio=self.h_s/self.b_s
  k_t       =self.b_t/tau_s
  b_so			=  0.004
  b_cu			=  self.b_s -2*h_i         # conductor width
  gamma			=  4/pi*(b_so/2/(g+self.h_m/mu_r)*atan(b_so/2/(g+self.h_m/mu_r))-log(sqrt(1+(b_so/2/(g+self.h_m/mu_r))**2)))
  k_C				=  tau_s/(tau_s-gamma*(g+self.h_m/mu_r))   # carter coefficient
  g_eff			=  k_C*(g+self.h_m/mu_r)                   
  om_m			=  gear*2*pi*self.n_nom/60
  
  om_e			=  self.p*om_m/2
  
  alpha_p		=  pi/2*0.7
  self.B_pm1	 		=  B_r*self.h_m/mu_r/(g_eff)
  
  self.B_g=  B_r*self.h_m/mu_r/(g_eff)*(4/pi)*sin(alpha_p)
  self.B_symax=self.B_g*self.b_m*l_e/(2*self.h_ys*l_u)
  self.B_rymax=self.B_g*self.b_m*l_e/(2*self.h_yr*l)
  
  self.B_tmax	=self.B_g*tau_s/self.b_t
  q3					= self.B_g**2/2/mu_0   # normal stress
  m2        =(k_2/R_st)**2   
  c1        =R_st/500
  R_1s      = R_st-self.t_s*0.5
  d_se=dia+2*(self.h_ys+self.h_s+h_w)  # stator outer diameter
  
  self.mass_PM   =(2*pi*(R+0.5*self.t)*l*self.h_m*ratio*self.rho_PM)           # magnet mass
  
                                             
  w_r					=self.rho_Fes*g1*sin(phi)*a_r*N_r
  

  mass_st_lam=self.rho_Fe*2*pi*(R+0.5*self.t)*l*self.h_yr                                     # mass of rotor yoke steel  
  W				=g1*sin(phi)*(mass_st_lam/N_r+(self.mass_PM)/N_r)  # weight of rotor cylinder
  Numer=R**3*((0.25*(sin(theta_r)-(theta_r*cos(theta_r)))/(sin(theta_r))**2)-(0.5/sin(theta_r))+(0.5/theta_r))
  Pov=((theta_r/(sin(theta_r))**2)+1/tan(theta_r))*((0.25*R/A_r)+(0.25*R**3/I_r))
  Qov=R**3/(2*I_r*theta_r*(m1+1))
  Lov=(R_1-self.R_o)/a_r
  Denom=I_r*(Pov-Qov+Lov) # radial deflection % rotor
  
  
  self.u_Ar				=(q3*R**2/E/self.t)*(1+Numer/Denom)
  
  self.u_all_r    =c/20 # allowable radial deflection
  self.u_all_s    = c1/20
  y_a1=(W*l_ir**3/12/E/I_arm_axi_r)
  y_a2=(w_r*l_iir**4/24/E/I_arm_axi_r) 

  self.y_Ar       =y_a1+y_a2 # axial deflection
  self.y_all     =2*l/100    # allowable axial deflection

  
  self.z_all_s     =0.05*2*pi*R_st/360  # allowable torsional deflection
  self.z_all_r     =0.05*2*pi*R/360  # allowable torsional deflection
  
  self.z_A_r       =(2*pi*(R-0.5*self.t)*l/N_r)*sigma*(l_ir-0.5*self.t)**3/3/E/I_arm_tor_r       # circumferential deflection
  
  val_str_cost_rotor=self.C_PM*self.mass_PM+self.C_Fe*(mass_st_lam)+self.C_Fes*(N_r*(R_1-self.R_o)*a_r*self.rho_Fes)
  val_str_rotor		= self.mass_PM+((mass_st_lam)+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes))
  
  
  self.Rotor_delta_radial=self.u_Ar
  self.Rotor_delta_axial=self.y_Ar
  self.Rotor_circum=self.z_A_r

  
  r_m     	=  self.r_s+h_sy0+self.h_ys+self.h_s #magnet radius
  
  r_r				=  self.r_s-g             #rotor radius
  
  self.K_rad=self.l_s/dia 

  

  v_ys=constant*(self.tau_p+pi*(self.h_s+0.5*self.h_ys)/self.p)*h_interp(self.B_symax)
  v_d=h_interp(self.B_tmax)*(h_s3+0.5*h_s2)+h_interp(self.B_tmax)*(0.5*h_s2+h_s1)
  v_yr=constant*(self.tau_p+pi*(g+self.h_m+0.5*self.h_yr)/self.p)*h_interp(self.B_rymax)
  v_m=self.h_m*self.B_g/mu_r/mu_0
  v_g       =  g_eff*self.B_g/mu_0
  
  # stator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  k_wd			= sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
  beta_skew	= tau_s/self.r_s;
  k_wskew		= sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  L_t=self.l_s+2*self.tau_p
  
  self.E_p	= 2*(self.N_s)*L_t*self.r_s*k_wd*k_wskew*om_m*self.B_g/sqrt(2)
  
  l_Cus			= 2*(self.N_s)*(2*self.tau_p+L_t)
  A_s				= self.b_s*(self.h_s-h_w)*q1*self.p
  A_scalc   = self.b_s*1000*(self.h_s*1000-h_w*1000)*q1*self.p
  A_Cus			= A_s*k_sfil/(self.N_s)
  self.A_Cuscalc = A_scalc *k_sfil/(self.N_s)
  self.R_s	= l_Cus*rho_Cu/A_Cus

  L_m				= 2*m*k_wd**2*(self.N_s)**2*mu_0*self.tau_p*L_t/pi**2/g_eff/self.p
  L_ssigmas=2*mu_0*self.l_s*self.N_s**2/self.p/q1*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=(2*mu_0*self.l_s*self.N_s**2/self.p/q1)*0.34*g*(l_e-0.64*self.tau_p*y_tau_p)/self.l_s                                #end winding leakage inductance
  L_ssigmag=2*mu_0*self.l_s*self.N_s**2/self.p/q1*(5*(g*k_C/b_so)/(5+4*(g*k_C/b_so))) # tooth tip leakage inductance#tooth tip leakage inductance
  L_ssigma	= (L_ssigmas+L_ssigmaew+L_ssigmag)
  self.L_s  = L_m+L_ssigma
  Z=(self.P_gennom/(m*self.E_p))
  
  #self.I_s=Z
  G=(self.E_p**2-(om_e*self.L_s*Z)**2)
   
  self.I_s= sqrt(Z**2+(((self.E_p-G**0.5)/(om_e*self.L_s)**2)**2))
  
  self.J_s	= self.I_s/self.A_Cuscalc

  I_snom		=(self.P_gennom/m/self.E_p/cofi) #rated current

  I_qnom		=self.P_gennom/(m*self.E_p)
  X_snom		=om_e*(L_m+L_ssigma)
  
  self.B_smax=sqrt(2)*self.I_s*mu_0/g_eff
  pp=1  #number of parallel paths
  N_s_j=sqrt(3)*(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*self.J_s*pp*U_Nrated/self.P_gennom
  N_s_s=(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*0.5
  
  self.A_1 = 6*self.N_s*self.I_s/(pi*dia)
  N_s_EL=self.A_1*pi*(dia-g)/(q1*self.p*m)
  self.N_s_max=min([N_s_j,N_s_s,N_s_EL])
  

  #" masses %%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%"
  V_Cus 	=m*l_Cus*A_Cus     # copper volume
  V_Fest	=L_t*2*self.p*q1*m*self.b_t*self.h_s   # volume of iron in stator tooth
  V_Fesy	=L_t*pi*((self.r_s+self.h_s+self.h_ys+h_sy0)**2-(self.r_s+self.h_s)**2) # volume of iron in stator yoke
  V_Fery	=L_t*pi*((r_r-self.h_m)**2-(r_r-self.h_m-self.h_yr)**2)
  
  self.M_Cus		=V_Cus*self.rho_Copper
  self.M_Fest	=V_Fest*self.rho_Fe
  self.M_Fesy	=V_Fesy*self.rho_Fe
  self.M_Fery	=V_Fery*self.rho_Fe
  M_Fe		=self.M_Fest+self.M_Fesy+self.M_Fery
  M_gen		=(self.M_Cus)
  K_gen		=self.M_Cus*self.C_Cu
  
  mass_st_lam_s= self.M_Fest+pi*l*self.rho_Fe*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2) 
  W_is			=0.5*g1*sin(phi)*(self.rho_Fes*l*self.d_s**2) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
  W_iis     =g1*sin(phi)*(mass_st_lam_s+V_Cus*self.rho_Copper)/2/N_st
  w_s         =self.rho_Fes*g1*sin(phi)*a_s*N_st
  
  l_is      =R_st-self.R_o
  l_iis     =l_is 
  l_iiis    =l_is
    

  
  
  #stator structure deflection calculation
  mass_stru_steel  =2*(N_st*(R_1s-self.R_o)*a_s*self.rho_Fes)
  Numers=R_st**3*((0.25*(sin(theta_s)-(theta_s*cos(theta_s)))/(sin(theta_s))**2)-(0.5/sin(theta_s))+(0.5/theta_s))
  Povs=((theta_s/(sin(theta_s))**2)+1/tan(theta_s))*((0.25*R_st/A_st)+(0.25*R_st**3/I_st))
  Qovs=R_st**3/(2*I_st*theta_s*(m2+1))
  Lovs=(R_1s-self.R_o)*0.5/a_s
  Denoms=I_st*(Povs-Qovs+Lovs) 
 
  self.u_As				=(q3*R_st**2/E/self.t_s)*(1+Numers/Denoms)
  X_comp1 = (W_is*l_is**3/12/E/I_arm_axi_s)
  X_comp2 =(W_iis*l_iis**4/24/E/I_arm_axi_s)
  X_comp3 =w_s*l_iiis**4/24/E/I_arm_axi_s
  
  self.y_As       =X_comp1+X_comp2+X_comp3  # axial deflection
  self.z_A_s  =2*pi*(R_st+0.5*self.t_s)*l/(2*N_st)*sigma*(l_is+0.5*self.t_s)**3/3/E/I_arm_tor_s 
   #val_str_stator		= mass_stru_steel+mass_st_lam_s+((sign(u_As-u_all)+1)*(u_As-u_all)**3*1e100)+(((sign(y_As-y_all)+1)*(y_As-y_all)**3*1e100))+(((sign(z_As-z_all_s)+1)*(z_As-z_all_s)**3*1e100))+(((sign(self.b_st-b_all_s)+1)*(self.b_st-b_all_s)**3*1e50))+((sign((T/2/pi/sigma)-(R_st**2*l))+1)*((T/2/pi/sigma)-(R_st**2*l))**3*1e50)
  val_str_stator		= mass_stru_steel+mass_st_lam_s
  
  val_str_cost_stator =	self.C_Fes*mass_stru_steel+self.C_Fe*mass_st_lam_s
    
  val_str_mass=val_str_rotor+val_str_stator
  
  
  self.Stator_delta_radial=self.u_As
  self.Stator_delta_axial=self.y_As
  self.Stator_circum=self.z_A_s
  self.TC1=T/(2*pi*sigma)     # Desired shear stress 
  self.TC2=R**2*l              # Evaluating Torque constraint for rotor
  self.TC3=R_st**2*l           # Evaluating Torque constraint for stator
  
  self.Iron=mass_st_lam_s+(2*pi*t*l*(R+0.5*self.t)*self.rho_Fe)
  self.PM=self.mass_PM
  self.Copper=self.M_Cus
  self.Inactive=mass_stru_steel+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes)
  
  self.Stator=mass_st_lam_s+mass_stru_steel+self.Copper
  self.Rotor=((2*pi*t*l*(R+0.5*self.t)*self.rho_Fe)+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes))+self.PM
  self.M_actual	=self.Stator+self.Rotor
  self.Mass = self.M_actual
  self.Active=self.Iron+self.Copper+self.mass_PM
 #"% losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  P_Cu		=m*self.I_s**2*self.R_s
  
  K_R=1.2
  P_Sc=m*(self.R_s)*K_R*(I_snom)**2*1  #losses to include skin effect
  
  
  if (self.K_load==0):
  	P_Cusnom_total=P_Cu
  else:
    P_Cusnom_total=P_Sc*self.K_load**2
  
  
  
  
  #B_tmax	=B_pm*tau_s/self.b_t
  P_Hyys	=self.M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
  P_Ftys	=self.M_Fesy*((self.B_symax/1.5)**2)*(P_Fe0e*(om_e/(2*pi*60))**2)
  P_Fesynom=P_Hyys+P_Ftys
  P_Hyd=self.M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
  P_Ftd=self.M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)
  P_Festnom=P_Hyd+P_Ftd
  P_ad=0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd ) # additional stray losses due to leakage flux
  pFtm =300 # specific magnet loss
  P_Ftm=pFtm*2*self.p*self.b_m*self.l_s
  
  P_genlossnom=P_Cusnom_total+P_Festnom+P_Fesynom+P_ad+P_Ftm

  self.gen_eff=self.P_gennom*100/(self.P_gennom+P_genlossnom)
  
  self.Losses=P_genlossnom
  
  
  #"% cost of losses"
  val_str_cost=val_str_cost_rotor+val_str_cost_stator
  self.Costs=val_str_cost+K_gen
  
  self.I[0]   = (0.5*self.M_actual*self.R_out**2)
  self.I[1]   = (0.25*self.M_actual*self.R_out**2+(1/12)*self.M_actual*self.l_s**2) 
  self.I[2]   = self.I[1]
  cm[0]  = self.main_shaft_cm[0] + self.main_shaft_length/2. + self.l_s/2
  cm[1]  = self.main_shaft_cm[1]
  cm[2]  = self.main_shaft_cm[2]
  
  #print ((self.mass_PM))*g1,  (self.M_Fest+V_Cus*self.rho_Copper)*g1,q3
  
  
  

class Drive_PMSG_arms(Assembly):
	Eta_target=Float(iotype='in', desc='Target drivetrain efficiency')
	T_rated=Float(iotype='in', desc='Torque')
	N=Float(iotype='in', desc='rated speed')
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
	I = Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
	cm = Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc=' Center of mass [x, y,z]')
	PMSG_r_s= Float(iotype='in', units='m', desc='Air gap radius of a permanent magnet excited synchronous generator')
	PMSG_l_s= Float(iotype='in', units='m', desc='Core length of the permanent magnet excited synchronous generator')
	PMSG_h_s = Float(iotype='in', units='m', desc='Stator Slot height of the permanent magnet excited synchronous generator')
	PMSG_tau_p = Float(iotype='in', units='m', desc='Pole pitch of the permanent magnet excited synchronous generator')
	PMSG_h_m = Float(iotype='in', units='m', desc='Magnet height of the permanent magnet excited synchronous generator')
	PMSG_h_ys = Float(iotype='in', units='m', desc='Stator yoke height of the permanent magnet excited synchronous generator')
	PMSG_h_yr = Float(iotype='in', units='m', desc='Rotor yoke height of the permanent magnet excited synchronous generator')
	PMSG_n_s = Float(iotype='in', units='m', desc='Stator Spokes of the permanent magnet excited synchronous generator')
	PMSG_b_st = Float(iotype='in', units='m', desc='Circumferential arm dimension of stator spoke')
	PMSG_n_r =Float(iotype='in', desc='Number of Rotor Spokes of the permanent magnet excited synchronous generator')
	PMSG_b_r = Float(iotype='in', desc='Circumferential arm dimension of the rotor spoke')
	PMSG_d_r = Float(iotype='in', desc='Rotor arm depth')
	PMSG_d_s= Float(iotype='in', desc='Stator arm depth ')
	PMSG_t_wr =Float(iotype='in', desc='Rotor arm thickness')
	PMSG_t_ws =Float(iotype='in', desc='Stator arm thickness')
	PMSG_R_o =Float(iotype='in', desc='Main shaft radius')
	
	C_Cu=Float( iotype='in', desc='Specific cost of copper')
	C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
	C_PM=Float(iotype='in', desc='Specific cost of Magnet')
	
	rho_Fes=Float(iotype='in', desc='Structural Steel density kg/m^3')
	rho_Fe=Float(iotype='in', desc='Magnetic Steel density kg/m^3')
	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
	rho_PM=Float(iotype='in', desc='Magnet density kg/m^3')
 	
 	
  
	
	def __init__(self,Optimiser='',Objective_function=''):
		
		super(Drive_PMSG_arms,self).__init__()
		self.Optimiser=Optimiser
		self.Objective_function=Objective_function
		""" Creates a new Assembly containing PMSG and an optimizer"""
		self.add('PMSG',PMSG())
		self.connect('PMSG_r_s','PMSG.r_s')
		self.connect('PMSG_l_s','PMSG.l_s')
		self.connect('PMSG_h_s','PMSG.h_s')
		self.connect('PMSG_tau_p','PMSG.tau_p')
		self.connect('PMSG_h_m','PMSG.h_m')
		self.connect('PMSG_h_ys','PMSG.h_ys')
		self.connect('PMSG_h_yr','PMSG.h_yr')
		self.connect('PMSG_n_s','PMSG.n_s')
		self.connect('PMSG_b_st','PMSG.b_st')
		self.connect('PMSG_d_s','PMSG.d_s')
		self.connect('PMSG_t_ws','PMSG.t_ws')
		self.connect('PMSG_n_r','PMSG.n_r')
		self.connect('PMSG_b_r','PMSG.b_r')
		self.connect('PMSG_d_r','PMSG.d_r')
		self.connect('PMSG_t_wr','PMSG.t_wr')
		self.connect('PMSG_R_o','PMSG.R_o')
		self.connect('P_rated','PMSG.machine_rating')
		self.connect('N','PMSG.n_nom')
		self.connect('main_shaft_cm','PMSG.main_shaft_cm')
		self.connect('main_shaft_length','PMSG.main_shaft_length')
		self.connect('T_rated','PMSG.Torque')
		self.connect('PMSG.M_actual','Mass')
		self.connect('PMSG.gen_eff','Efficiency')
		self.connect('PMSG.r_s','r_s')
		self.connect('PMSG.l_s','l_s')
		self.connect('PMSG.I','I')
		self.connect('PMSG.cm','cm')
		
		self.connect('C_Fe','PMSG.C_Fe')
		self.connect('C_Fes','PMSG.C_Fes')
		self.connect('C_Cu','PMSG.C_Cu')
		self.connect('C_PM','PMSG.C_PM')
		
		self.connect('rho_PM','PMSG.rho_PM')
		self.connect('rho_Fe','PMSG.rho_Fe')
		self.connect('rho_Fes','PMSG.rho_Fes')
		self.connect('rho_Copper','PMSG.rho_Copper')

				
		
		
		Opt1=globals()[self.Optimiser]
		self.add('driver',Opt1())
		self.driver.iprint = 1 
		if (Opt1=='CONMINdriver'):
			#Create Optimizer instance
			self.driver.itmax = 100
			self.driver.fdch = 0.01
			self.driver.fdchm = 0.01
			self.driver.ctlmin = 0.01
			self.driver.delfun = 0.001
			self.driver.conmin_diff = True
		elif (Opt1=='COBYLAdriver'):
			# COBYLA-specific Settings
			self.driver.rhobeg=1.0
			self.driver.rhoend = 1.0e-4
			self.driver.maxfun = 1000
		elif (Opt1=='SLSQPdriver'):
			# SLSQP-specific Settings
			self.driver.accuracy = 1.0e-6
			self.driver.maxiter = 50
		elif (Opt1=='Genetic'):
			# Genetic-specific Settings
			self.driver.population_size = 90
			self.driver.crossover_rate = 0.9
			self.driver.mutation_rate = 0.02
			self.selection_method = 'rank'
		else:
			# NEWSUMT-specific Settings
			self.driver.itmax = 10 
		
			
		Obj1='PMSG'+'.'+self.Objective_function
		self.driver.add_objective(Obj1)
		self.driver.design_vars=['PMSG_r_s','PMSG_l_s','PMSG_h_s','PMSG_tau_p','PMSG_h_m','PMSG_h_ys','PMSG_h_yr','PMSG_n_r','PMSG_n_s','PMSG_b_r','PMSG_b_st','PMSG_d_r','PMSG_d_st','PMSG_t_wr','PMSG_t_ws']
		self.driver.add_parameter('PMSG_r_s', low=0.5, high=9)
		self.driver.add_parameter('PMSG_l_s', low=0.5, high=2.5)
		self.driver.add_parameter('PMSG_h_s', low=0.04, high=0.1)
		self.driver.add_parameter('PMSG_tau_p', low=0.04, high=0.1)
		self.driver.add_parameter('PMSG_h_m', low=0.005, high=0.1)
		self.driver.add_parameter('PMSG_n_r', low=5., high=15.)
		self.driver.add_parameter('PMSG_h_yr', low=0.045, high=0.25)
		self.driver.add_parameter('PMSG_h_ys', low=0.045, high=0.25)
		self.driver.add_parameter('PMSG_b_r', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_d_r', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_t_wr', low=0.001, high=0.2)
		self.driver.add_parameter('PMSG_n_s', low=5., high=15.)
		self.driver.add_parameter('PMSG_b_st', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_d_s', low=0.1, high=1.5)
		self.driver.add_parameter('PMSG_t_ws', low=0.001, high=0.2)
		
		self.driver.add_constraint('PMSG.B_symax<2')										  #1
		self.driver.add_constraint('PMSG.B_rymax<2')										  #2
		self.driver.add_constraint('PMSG.B_tmax<2')									      #3
		self.driver.add_constraint('PMSG.B_smax<PMSG.B_g') 								#4
		self.driver.add_constraint('PMSG.B_g>=0.7')  											#5                
		self.driver.add_constraint('PMSG.B_g<=1.2') 											#6
		self.driver.add_constraint('PMSG.E_p>=500')											  #7
		self.driver.add_constraint('PMSG.E_p<=5000')											#8
		self.driver.add_constraint('PMSG.u_As<PMSG.u_all_s')							#9
		self.driver.add_constraint('PMSG.z_A_s<PMSG.z_all_s')							#10
		self.driver.add_constraint('PMSG.y_As<PMSG.y_all')  							#11
		self.driver.add_constraint('PMSG.u_Ar<PMSG.u_all_r')							#12
		self.driver.add_constraint('PMSG.z_A_r<PMSG.z_all_r')							#13
		self.driver.add_constraint('PMSG.y_Ar<PMSG.y_all') 								#14
		self.driver.add_constraint('PMSG.TC1<PMSG.TC2')    								#15
		self.driver.add_constraint('PMSG.TC1<PMSG.TC3')    								#16
		self.driver.add_constraint('PMSG.b_r<PMSG.b_all_r')								#17
		self.driver.add_constraint('PMSG.b_st<PMSG.b_all_s')							#18
		self.driver.add_constraint('PMSG.A_1<60000')											#19
		self.driver.add_constraint('PMSG.J_s<=6') 												#20
		self.driver.add_constraint('PMSG.A_Cuscalc>=5') 									#21
		self.driver.add_constraint('PMSG.K_rad>0.2')											#22
		self.driver.add_constraint('PMSG.K_rad<=0.27')										#23 
		self.driver.add_constraint('PMSG.Slot_aspect_ratio>=4')						#24
		self.driver.add_constraint('PMSG.Slot_aspect_ratio<=10')					#25	
		self.driver.add_constraint('PMSG.gen_eff>=Eta_target')			#26
		
				
def optim_PMSG():
	opt_problem = Drive_PMSG_arms('CONMINdriver','Costs')
	opt_problem.Eta_target = 93
	# Initial design variables for a DD PMSG designed for a 5MW turbine
	opt_problem.P_rated=5.0e6
	opt_problem.T_rated=4.143289e6
	opt_problem.N=12.1
	opt_problem.PMSG_r_s= 3.26               
	opt_problem.PMSG_l_s= 1.6
	opt_problem.PMSG_h_s = 0.07
	opt_problem.PMSG_tau_p = 0.08
	opt_problem.PMSG_h_m = 0.009
	opt_problem.PMSG_h_ys = 0.0615
	opt_problem.PMSG_h_yr = 0.057
	opt_problem.PMSG_n_s = 5
	opt_problem.PMSG_b_st = 0.480
	opt_problem.PMSG_n_r =5
	opt_problem.PMSG_b_r = 0.520
	opt_problem.PMSG_d_r = 0.7
	opt_problem.PMSG_d_s= 0.3
	opt_problem.PMSG_t_wr =0.06
	opt_problem.PMSG_t_ws =0.06
	opt_problem.PMSG_R_o =0.43      #0.523950817  #0.43  #0.523950817 #0.17625 #0.2775 #0.363882632 ##0.35 #0.523950817 #0.43 #523950817 #0.43 #0.523950817 #0.523950817 #0.17625 #0.2775 #0.363882632 #0.43 #0.523950817 #0.43

#	
	

#	opt_problem.Eta_target = 93
#	# Initial design variables for a DD PMSG designed for a 10MW turbine
#	opt_problem.P_rated=0.75e6
#	opt_problem.T_rated=250418.6168
#	opt_problem.N=28.6
#	opt_problem.PMSG_r_s=1.3             
#	opt_problem.PMSG_l_s= 0.71
#	opt_problem.PMSG_h_s = 0.045
#	opt_problem.PMSG_tau_p = 0.06
#	opt_problem.PMSG_h_m = 0.009
#	opt_problem.PMSG_h_ys = 0.045
#	opt_problem.PMSG_h_yr = 0.045
#	opt_problem.PMSG_n_s = 5
#	opt_problem.PMSG_b_st = 0.220
#	opt_problem.PMSG_n_r =5
#	opt_problem.PMSG_b_r = 0.220
#	opt_problem.PMSG_d_r = 0.4
#	opt_problem.PMSG_d_s= 0.18
#	opt_problem.PMSG_t_wr =0.01
#	opt_problem.PMSG_t_ws =0.01
#	opt_problem.PMSG_R_o =0.17625                    # 1.5MW: 0.2775; 3MW: 0.363882632; 10 MW:0.523950817; 0.75MW :0.17625
	
	# Specific costs
	opt_problem.C_Cu   =4.786                  
	opt_problem.C_Fe	= 0.556                   
	opt_problem.C_Fes =0.50139                   
	opt_problem.C_PM  =95
	
	#Material properties
	opt_problem.rho_Fe = 7700                 #Steel density
	opt_problem.rho_Fes = 7850                 #Steel density
	opt_problem.rho_Copper =8900                  # Kg/m3 copper density
	opt_problem.rho_PM =7450
	
	
	
	
	
	opt_problem.run()
	raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor arms','Rotor Axial arm dimension','Rotor Circumferential arm dimension' ,'Rotor arm Thickness',' Stator Radial deflection', 'Stator Axial deflection','Stator circum deflection',' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Air gap diameter','Overall Outer diameter', 'Stator length', 'l/d ratio','Slot_aspect_ratio','Pole pitch', 'Stator slot height','Stator slotwidth','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental','Peak stator yoke flux density','Peak rotor yoke flux density','Flux density above magnet','Maximum Stator flux density','Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage', 'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns','Conductor cross-section','Stator Current density ','Specific current loading','Generator Efficiency ','Iron mass','Magnet mass','Copper mass','Mass of Arms', 'Total Mass', 'Stator Mass','Rotor Mass','Total Material Cost'],
			'Values': [opt_problem.PMSG.P_gennom/1000000,opt_problem.PMSG.n_s,opt_problem.PMSG.d_s*1000,opt_problem.PMSG.b_st*1000,opt_problem.PMSG.t_ws*1000,opt_problem.PMSG.n_r,opt_problem.PMSG.d_r*1000,opt_problem.PMSG.b_r*1000,opt_problem.PMSG.t_wr*1000,opt_problem.PMSG.Stator_delta_radial*1000,opt_problem.PMSG.Stator_delta_axial*1000,opt_problem.PMSG.Stator_circum*1000,opt_problem.PMSG.Rotor_delta_radial*1000,opt_problem.PMSG.Rotor_delta_axial*1000,opt_problem.PMSG.Rotor_circum*1000,2*opt_problem.PMSG.r_s,opt_problem.PMSG.R_out*2,opt_problem.PMSG.l_s,opt_problem.PMSG.K_rad,opt_problem.PMSG.Slot_aspect_ratio,opt_problem.PMSG.tau_p*1000,opt_problem.PMSG.h_s*1000,opt_problem.PMSG.b_s*1000,opt_problem.PMSG.b_t*1000,opt_problem.PMSG.t_s*1000,opt_problem.PMSG.t*1000,opt_problem.PMSG.h_m*1000,opt_problem.PMSG.b_m*1000,opt_problem.PMSG.B_g,opt_problem.PMSG.B_symax,opt_problem.PMSG.B_rymax,opt_problem.PMSG.B_pm1,opt_problem.PMSG.B_smax,opt_problem.PMSG.B_tmax,opt_problem.PMSG.p,opt_problem.PMSG.f,opt_problem.PMSG.E_p,opt_problem.PMSG.I_s,opt_problem.PMSG.R_s,opt_problem.PMSG.L_s,opt_problem.PMSG.S,opt_problem.PMSG.N_s,opt_problem.PMSG.A_Cuscalc,opt_problem.PMSG.J_s,opt_problem.PMSG.A_1/1000,opt_problem.PMSG.gen_eff,opt_problem.PMSG.Iron/1000,opt_problem.PMSG.mass_PM/1000,opt_problem.PMSG.M_Cus/1000,opt_problem.PMSG.Inactive/1000,opt_problem.PMSG.M_actual/1000,opt_problem.PMSG.Stator/1000,opt_problem.PMSG.Rotor/1000,opt_problem.PMSG.Costs/1000],
				'Limit': ['','','',opt_problem.PMSG.b_all_s*1000,'','','',opt_problem.PMSG.b_all_r*1000,'',opt_problem.PMSG.u_all_s*1000,opt_problem.PMSG.y_all*1000,opt_problem.PMSG.z_all_s*1000,opt_problem.PMSG.u_all_r*1000,opt_problem.PMSG.y_all*1000,opt_problem.PMSG.z_all_r*1000,'','','','(0.2-0.27)','(4-10)','','','','','','','','','','<2','<2','<2',opt_problem.PMSG.B_g,'','','','','>500','','','','','5','3-6','60','>93%','','','','','','','',''],
				'Units':['MW','unit','mm','mm','mm','mm','mm','','mm','mm','mm','mm','mm','mm','mm','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','A/mm^2','slots','turns','mm^2','kA/m','%','tons','tons','tons','tons','tons','ton','ton','k$']}
	df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print df
	df.to_excel('PMSG_'+str(opt_problem.P_rated/1e6)+'_arms_MW.xlsx')


		
if __name__=="__main__":
>>>>>>> develop
	optim_PMSG()