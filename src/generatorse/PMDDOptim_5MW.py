"""PMDD.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float
from openmdao.lib.datatypes.api import Array
from numpy import array
from numpy import float 
from scipy.interpolate import interp1d



class PMDDOptim_5MW(Component):

 """ Evaluates the total cost """
 
 r_s = Float(3.25, iotype='in', desc='airgap radius r_s')
 l_s = Float(1.6, iotype='in', desc='Stator core length l_s') #1.7
 h_s = Float(0.04, iotype='in', desc='Yoke height h_s') #0.06
 tau_p=Float(0.08, iotype='in', desc='Pole pitch self.tau_p') #0.09
 B_g = Float(0, iotype='out', desc='Peak air gap flux density B_g')
 B_symax = Float(1.2, iotype='out', desc='Peak Stator Yoke flux density B_ymax')
 B_tmax=Float(0.01, iotype='out', desc='Peak Teeth flux density')
 B_rymax=Float(0.01, iotype='out', desc='Peak Rotor yoke flux density')
 B_smax=Float(0.01, iotype='out', desc='Peak Stator flux density')
 B_pm1=Float(0.01, iotype='out', desc='Fundamental component of peak air gap flux density')
 n_r =Float(5, iotype='in', desc='number of arms n')
 h_m=Float(0.01449, iotype='in', desc='magnet height') #0.005
 n_r =Float(5, iotype='in', desc='number of arms n')
 t= Float(0.061, iotype='in', desc='thickness of arms t') #0.049
 t_s= Float(0.065, iotype='in', desc='thickness of arms t') #0.047
 b_r = Float(0.54, iotype='in', desc='arm width b') #0.53 cost
 d_r = Float(0.7, iotype='in', desc='arm depth d') #0.4 cost
 t_wr =Float(0.06, iotype='in', desc='arm depth thickness self.t_w')
 n_s =Float(5, iotype='in', desc='number of arms n')
 b_st = Float(0.48, iotype='in', desc='arm width b_r') #0.445 for cost
 d_s = Float(0.3, iotype='in', desc='arm depth d_r') #0.1 for cost
 t_ws =Float(0.06, iotype='in', desc='arm depth thickness self.t_wr') #0.2
 W_1a =Float(300, iotype='out', desc='Number of turns in the stator winding')
 A_1 =Float(0, iotype='out', desc='Electrical loading')
 M_actual=Float(0.01, iotype='out', desc='Actual mass')
 p=Float(190, iotype='out', desc='No of pole pairs')
 #f=Float(0.01, iotype='out', desc='Output frequency')
 E_p=Float(4000, iotype='out', desc='Stator phase voltage')
 J_s=Float(0.01, iotype='out', desc='Current density')
 #T_winding=Float(0.01, iotype='out', desc='Winding Temp')
 TC= Float(0.01, iotype='out', desc='Total cost')
 TM=Float(0.01, iotype='out', desc='Total Mass')
 K_rad=Float(0.01, iotype='out', desc='K_rad')  
 TL=Float(0.01, iotype='out', desc='Total loss')
 gen_eff=Float(0.01, iotype='out', desc='Generator efficiency')
 Active=Float(0.01, iotype='out', desc='Generator efficiency')
 Stator=Float(0.01, iotype='out', desc='Generator efficiency')
 Rotor=Float(0.01, iotype='out', desc='Generator efficiency')
 mass_PM=Float(0.01, iotype='out', desc='Generator efficiency')
 Electrical_Steel=Float(0.01, iotype='out', desc='Generator efficiency')
 A_Cuscalc=Float(0.01, iotype='out', desc='Conductor cross-section mm^2')
 u_Ar	=Float(0.1, iotype='out', desc='Radial deflection-rotor')
 u_As	=Float(0.1, iotype='out', desc='Radial deflection-stator')
 u_all_r=Float(0.1, iotype='out', desc='Allowable Radial deflection-rotor')
 u_all_s=Float(0.1, iotype='out', desc='Allowable Radial deflection-stator')
 y_Ar	=Float(0.1, iotype='out', desc='Axial deflection-rotor')
 y_As	=Float(0.1, iotype='out', desc='Axial deflection-stator')
 y_all=Float(0.1, iotype='out', desc='Allowable axial deflection')
 z_A_r	=Float(0.1, iotype='out', desc='Circum deflection-rotor')
 z_As	=Float(0.1, iotype='out', desc='Circum deflection-stator')
 z_all_r=Float(0.1, iotype='out', desc='Allowable Circum deflection-rotor')
 z_all_s=Float(0.1, iotype='out', desc='Allowable Circum deflection-stator')
 TC1	=Float(0.1, iotype='out', desc='Torque constraint')
 TC2=Float(0.1, iotype='out', desc='Torque constraint-rotor')
 TC3=Float(0.1, iotype='out', desc='Torque constraint-stator')
 b_all_r			=Float(0.1, iotype='out', desc='rotor arm constraint')
 b_all_s			=Float(0.1, iotype='out', desc='Stator arm constraint') 
 Cost			=Float(0.1, iotype='out', desc='Cost')
 N_s_max			=Float(0.1, iotype='out', desc='Maximum number of turns per coil')
 Slot_aspect_ratio=Float(0.1, iotype='out', desc='Slot aspect ratio')
 N_slots			=Float(0.1, iotype='out', desc='Stator slots')
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  tau_p =self.tau_p
  B_g = self.B_g
  B_ymax = self.B_symax
  n_r =self.n_r
  t= self.t
  t_s= self.t_s
  b_r = self.b_r
  d_r  = self.d_r
  t_wr =self.t_wr
  n_s =self.n_s
  b_st = self.b_st
  d_s  = self.d_s
  t_ws =self.t_ws
  TC =self.TC
  TM=self.TM
  E_p=self.E_p
  p=self.p
  #I_s=self.I_s
  #R_s=self.R_s
  #L_s=self.L_s
  J_s=self.J_s
  M_actual=self.M_actual
  Active=self.Active
  Stator=self.Stator
  Rotor=self.Rotor
  mass_PM=self.mass_PM
  gen_eff =self.gen_eff
  #P=self.P
  TL=self.TL
  K_rad=self.K_rad
  #P=self.P
  W_1a=self.W_1a
  Electrical_Steel=self.Electrical_Steel
  Cost=self.Cost
  u_Ar	=self.u_Ar
  u_As	=self.u_As
  z_As=self.z_As
  u_all_r=self.u_all_r
  u_all_s=self.u_all_s
  y_Ar	=self.y_Ar
  y_As	=self.y_As
  y_all=self.y_all
  z_A_r	=self.z_A_r
  z_all_r=self.z_all_r
  z_all_s=self.z_all_s
  TC1	=self.TC1
  TC2=self.TC2
  TC3=self.TC3
  b_all_r			=self.b_all_r
  b_all_s			=self.b_all_s
  B_tmax= self.B_tmax
  B_rymax=self.B_rymax
  B_smax=self.B_smax
  B_pm1=self.B_pm1
  N_s_max=self.N_s_max
  A_Cuscalc=self.A_Cuscalc
  Slot_aspect_ratio=self.Slot_aspect_ratio
  N_slots=self.N_slots
  
  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign

  
  
  
  rho    =7850                # Kg/m3 steel density
  rho_PM =7450                # Kg/m3 magnet density
  B_r    =1.2                 # Tesla remnant flux density 
  g1      =9.81                # m/s^2 acceleration due to gravity
  E      =2e11                # N/m^2 young's modulus
  sigma  =40e3                # shear stress assumed
  ratio  =0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
  mu_0   =pi*4e-7        # permeability of free space
  mu_r   =1.06
  phi    =90*2*pi/360 
  cofi   =0.85                 # power factor
  K_Cu   =15                  # Unit cost of Copper 
  K_Fe   =3                   # Unit cost of Iron 
  K_pm   =25                  # Unit cost of magnet
  K_elec =10                  # electricity price in c/kWH
  K_conv =1.1                  # Converter cost
  years  =8760                 # number of hours in a year
  h_sy0  =0
  #h_sy   =0.04
  h_w    =0.005
  h_i=0.001 # coil insulation thickness
  h_s1= 0.001
  b_s1=0.003
  h_s2=0.004
  h_s3=self.h_s-h_s1-h_s2
  h_cu=(h_s3-4*h_i)*0.5
  self.K_rad=self.l_s/(2*self.r_s)
  m      =3
  #B_rymax=1.4
  q1     =1
  b_s_tau_s=0.45
  k_sfil =0.65								 # Slot fill factor
  P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  #B_rymax=1.4
  rho_Cu=1.8*10**(-8)*1.4
  k_fes =0.9
  R_o = 0.43
  k_sfil =0.65								 # Slot fill factor
  T = 4143289.841 #1.7904931e6
  P_gennom = 5.0e6
  P_gridnom=3e6
  E_pnom   = 3000
  n_nom = 12.1
  P_convnom=0.03*P_gridnom
  y_tau_p=1
  gear      =1   
  self.K_load=1
  
  B_curve=(0.000,0.050,0.130,0.152,0.176,0.202,0.229,0.257,0.287,0.318,0.356,0.395,0.435,0.477,0.523,0.574,0.627,0.683,0.739,0.795,0.850,0.906,0.964,1.024,1.084,1.140,1.189,1.228,1.258,1.283,1.304,1.324,1.343,1.360,1.375,1.387,1.398,1.407,1.417,1.426,1.437,1.448,1.462,1.478,1.495,1.513,1.529,1.543,1.557,1.570,1.584,1.600,1.617,1.634,1.652,1.669,1.685,1.701,1.717,1.733,1.749,1.769,1.792,1.820,1.851,1.882,1.909,1.931,1.948,1.961,1.972,1.981)
  H_curve=(0,10,20,23,25,28,30,33,35,38,40,43,45,48,52,56,60,66,72,78,87,97,112,130,151,175,200,225,252,282,317,357,402,450,500,550,602,657,717,784,867,974,1117,1299,1515,1752,2000,2253,2521,2820,3167,3570,4021,4503,5000,5503,6021,6570,7167,7839,8667,9745,11167,12992,15146,17518,20000,22552,25417,28906,33333,38932)
  U_Nrated=3.3e3
  constant=0.5
  # rotor structure     
  h_yr =self.t
  
  h_ys=self.t_s
  
  l					= self.l_s                          #l-stator core length
  l_u       =k_fes * self.l_s                   #useful iron stack length
  We				=self.tau_p
  l_b       = 2*self.tau_p  #end winding length
  l_e       =self.l_s+2*0.001*self.r_s     # equivalent core length
  b_m  			=0.7*self.tau_p 
  #h_ys =self.B_g*b_m*l_e/(2*self.B_symax*l_u) 
  #h_yr =h_ys
  b_s1			=  0.003
  

  
  a_r				= (self.b_r*self.d_r)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr))  # cross-sectional area of rotor armms
  a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws))
  A_r				= l*self.t 
  A_st      =l*self.t_s                     # cross-sectional area of rotor cylinder
  N_r				= round(self.n_r)
  N_st			= round(self.n_s)
  theta_r		=pi/N_r                             # half angle between spokes
  theta_s		=pi/N_st 
  I_r				=l*self.t**3/12                         # second moment of area of rotor cylinder
  I_st      =l*self.t_s**3/12 
  I_arm_axi_r	=((self.b_r*self.d_r**3)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr)**3))/12  # second moment of area of rotor arm
  I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
  I_arm_tor_r	= ((self.d_r*self.b_r**3)-((self.d_r-2*self.t_wr)*(self.b_r-2*self.t_wr)**3))/12  # second moment of area of rotot arm w.r.t torsion
  I_arm_tor_s	= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
  
  dia				=  2*self.r_s              # air gap diameter
  g         =  0.001*dia
  
  R					= self.r_s-g-self.h_m-0.5*self.t
  c					=R/500
  R_1				= R-self.t*0.5                              # inner radius of rotor cylinder
  
  R_st 			=self.r_s+self.h_s+h_ys*0.5
  R_1s      = R_st-self.t_s*0.5
  k_1				= sqrt(I_r/A_r) 
  c1        =R_st/500                         # radius of gyration
  k_2       = sqrt(I_st/A_st)
  m1				=(k_1/R)**2 
  
  K					=4*(sin(ratio*pi/2))/pi
  #h_m=c/B_r/K/((1/self.B_g)-(1/(B_r*K)))         # magnet height
  #self.B_g  = B_r*(self.h_m/mu_r)/(self.h_m+c)*K
  self.p		=  round(pi*dia/(2*self.tau_p))
  self.f    =  n_nom*self.p/60
  self.N_slots				= 2*self.p*q1*m 
  N_conductors=self.N_slots*2
  self.W_1a=N_conductors/2/3

  
  tau_s=pi*dia/self.N_slots
  #tau_s			=  (self.tau_p)/3/q1    #slot pitch
  b_s	=  b_s_tau_s*tau_s    #slot width   
  self.Slot_aspect_ratio=self.h_s/b_s
  b_t	=  tau_s-(b_s)          #tooth width
  k_t=b_t/tau_s
  #self.B_tmax=self.B_g*self.tau_p*l_e/(l_u*b_t)
  b_so			=  0.004
  b_cu			=  b_s-2*h_i         # conductor width
  gamma			=  4/pi*(b_so/2/(g+self.h_m/mu_r)*atan(b_so/2/(g+self.h_m/mu_r))-log(sqrt(1+(b_so/2/(g+self.h_m/mu_r))**2)))
  k_C				=  tau_s/(tau_s-gamma*(g+self.h_m/mu_r))   # carter coefficient
  g_eff			=  k_C*(g+self.h_m/mu_r)                   
  om_m			=  gear*2*pi*n_nom/60
  
  om_e			=  self.p*om_m/2
  self.f=  n_nom*self.p/60
  
  h_interp=interp1d(B_curve,H_curve,fill_value='extrapolate')
  alpha_p		=  pi/2*0.7
  
  self.B_pm1	 		=  B_r*self.h_m/mu_r/(g_eff)
  self.B_g=  B_r*self.h_m/mu_r/(g_eff)*(4/pi)*sin(alpha_p)
  q					= self.B_g**2/2/mu_0   # normal stress
  self.B_symax=self.B_g*b_m*l_e/(2*h_ys*l_u)
  self.B_rymax=self.B_g*b_m*l_e/(2*h_yr*l)
  
  self.B_tmax	=self.B_g*tau_s/b_t
  
  
 
  m2        =(k_2/R_st)**2                                  
  l_ir			=R                                      # length of rotor arm beam at which rotor cylinder acts
  l_iir			=R_1 
  l_is      =R_st-R_o
  l_iis     =l_is 
  l_iiis    =l_is
  
  
  #H_c*self.h_m-0.5*v_ys-0.5*v_yr-v_d-v_g
  self.b_all_r			=2*pi*R_o/N_r 
  self.b_all_s			=2*pi*R_o/N_st                           
  self.mass_PM   =(2*pi*(R+0.5*self.t)*l*self.h_m*ratio*rho_PM)           # magnet mass
                                           
  w_r					=rho*g1*sin(phi)*a_r*N_r
  
  #self.h_yr =0.5*B_r/B_rymax*self.tau_p*ratio*(self.h_m/mu_r)/((self.h_m/mu_r)+c)  # rotor yoke height
  
  
  #print theta
  mass_st_lam=7700*2*pi*(R+0.5*self.t)*l*h_yr                                     # mass of rotor yoke steel  
  W				=g1*sin(phi)*(mass_st_lam/N_r+(self.mass_PM/N_r))  # weight of rotor cylinder
  self.TC1=T/(2*pi*sigma)
  self.TC2=R**2*l
  self.TC3=R_st**2*l
  Numer=R**3*((0.25*(sin(theta_r)-(theta_r*cos(theta_r)))/(sin(theta_r))**2)-(0.5/sin(theta_r))+(0.5/theta_r))
  Pov=((theta_r/(sin(theta_r))**2)+1/tan(theta_r))*((0.25*R/A_r)+(0.25*R**3/I_r))
  Qov=R**3/(2*I_r*theta_r*(m1+1))
  Lov=(R_1-R_o)/a_r
  Denom=I_r*(Pov-Qov+Lov) # radial deflection % rotor
  self.u_Ar				=(q*R**2/E/self.t)*(1+Numer/Denom)
  
  self.u_all_r     =c/20  # allowable radial deflection
  
  self.u_all_s     =c1/20
  
  y_a1=(W*l_ir**3/12/E/I_arm_axi_r)
  y_a2=(w_r*l_iir**4/24/E/I_arm_axi_r) 

  self.y_Ar       =y_a1+y_a2  # axial deflection
  self.y_all     =0.2*l/100    # allowable axial deflection

  self.z_all_s     =0.05*2*pi*R_st/360  # allowable torsional deflection
  self.z_all_r     =0.05*2*pi*R/360  # allowable torsional deflection
  self.z_A_r			=2*pi*(R-0.5*self.t)*l/N_r*sigma*(l_ir-0.5*self.t)**3/3/E/I_arm_tor_r       # circumferential deflection       # circumferential deflection
  

 
  #val_str_cost_rotor		= 35*self.mass_PM+10*((2*pi*self.t*l*R*rho)+(N_r*(R_1-R_o)*a_r*rho))+((sign(u_Ar-u_all)+1)*(u_Ar-u_all)**3*5e150)+(((sign(y_Ar-y_all)+1)*(y_Ar-y_all)**3*5e17))+(((sign(z_A_r-z_all_r)+1)*(z_A_r-z_all_r)**3*1e100))+(((sign(self.b_r-b_all_r)+1)*(self.b_r-b_all_r)**3*5e50))+((sign((T/2/pi/sigma)-(R**2*l))+1)*((T/2/pi/sigma)-(R**2*l))**3*5e50)
  val_str_cost_rotor=95*self.mass_PM+0.50139*((N_r*(R_1-R_o)*a_r*rho))+mass_st_lam*0.556
  val_str_rotor		= self.mass_PM+(mass_st_lam+(N_r*(R_1-R_o)*a_r*rho))
  #val_str_rotor		= self.mass_PM+((2*pi*self.t*l*R*rho)+(N_r*(R_1-R_o)*a_r*rho))+((sign(u_Ar-u_all)+1)*(u_Ar-u_all)**3*1e100)+(((sign(y_Ar-y_all)+1)*(y_Ar-y_all)**3*1e100))+(((sign(z_A_r-z_all_r)+1)*(z_A_r-z_all_r)**3*1e100))+(((sign(self.b_r-b_all_r)+1)*(self.b_r-b_all_r)**3*5e50))+((sign((T/2/pi/sigma)-(R**2*l))+1)*((T/2/pi/sigma)-(R**2*l))**3*5e50)
  

  
  
  r_m     	=  self.r_s+h_sy0+h_ys+self.h_s #magnet radius
  
  r_r				=  self.r_s-g              #rotor radius
    # number of pole pairs
  
  

 
  v_ys=constant*(self.tau_p+pi*(self.h_s+0.5*h_ys)/self.p)*h_interp(self.B_symax)
  v_d=h_interp(self.B_tmax)*(h_s3+0.5*h_s2)+h_interp(self.B_tmax)*(0.5*h_s2+h_s1)
  v_yr=constant*(self.tau_p+pi*(g+self.h_m+0.5*h_yr)/self.p)*h_interp(self.B_rymax)
  v_m				=self.h_m*self.B_g/mu_r/mu_0
  v_g       =  g_eff*self.B_g/mu_0
  # stator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  k_wd			= sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
  beta_skew	= tau_s/self.r_s;
  k_wskew		= sin(p*beta_skew/2)/(p*beta_skew/2)
  L_t       =self.l_s+2*self.tau_p
  #W_1a     =2*3*3000/(sqrt(3)*sqrt(2)*60*k_wd*B_pm1*2*L_t*self.tau_p) 
  #k_wskew	=1
  self.E_p	= 2*(self.W_1a)*L_t*self.r_s*k_wd*k_wskew*om_m*self.B_g/sqrt(2)
  
  l_Cus			= 2*(self.W_1a)*(2*self.tau_p+L_t)
  A_s				= b_s*(self.h_s-h_w)*q1*self.p
  A_scalc   = b_s*1000*(self.h_s*1000-h_w*1000)*q1*self.p
  A_Cus			= A_s*k_sfil/(self.W_1a)
  self.A_Cuscalc = A_scalc *k_sfil/(self.W_1a)
  R_s				= l_Cus*rho_Cu/A_Cus
  L_m				= 2*m*k_wd**2*(self.W_1a)**2*mu_0*self.tau_p*L_t/pi**2/g_eff
  L_ssigmas=2*mu_0*self.l_s*self.W_1a**2/self.p/q1*((self.h_s-h_w)/(3*b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=(2*mu_0*self.l_s*self.W_1a**2/self.p/q1)*0.34*g*(l_e-0.64*self.tau_p*y_tau_p)/self.l_s                                #end winding leakage inductance
  L_ssigmag=2*mu_0*self.l_s*self.W_1a**2/self.p/q1*(5*(g*k_C/b_so)/(5+4*(g*k_C/b_so))) # tooth tip leakage inductance#tooth tip leakage inductance
  L_ssigma	= (L_ssigmas+L_ssigmaew+L_ssigmag)   #stator leakage inductance
  om_e      = self.p*om_m/2
  L_s  		  = L_m+L_ssigma

  Z=(P_gennom/(m*self.E_p))
  G=((-((om_e*L_s)**2)*Z**2+self.E_p**2))
  
  #I_s= sqrt(Z**2+((self.E_p-G/(om_e*L_s)**2)**2))
  I_s	= Z
  self.J_s=I_s/self.A_Cuscalc
  
  I_snom		=(P_gennom/m/self.E_p/cofi) #rated current

  I_qnom		=P_gennom/(m*self.E_p)
  X_snom		=om_e*(L_m+L_ssigma)
  
  self.B_smax=sqrt(2)*I_s*mu_0/g_eff
  pp=1  #number of parallel paths
  N_s_j=sqrt(3)*(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*self.J_s*pp*U_Nrated/P_gennom
  N_s_s=(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*0.5
  
  self.A_1 = 6*self.W_1a*I_s/(pi*dia*l)
  N_s_EL=self.A_1*pi*(dia-g)/(q1*self.p*m)
  self.N_s_max=min([N_s_j,N_s_s,N_s_EL])

  #" masses %%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%"
  V_Cus 	=m*l_Cus*A_Cus     # copper volume
  V_Fest	=L_t*2*self.p*q1*m*b_t*self.h_s   # volume of iron in stator tooth
  V_Fesy	=L_t*pi*((self.r_s+self.h_s+h_ys+h_sy0)**2-(self.r_s+self.h_s)**2) # volume of iron in stator yoke
  V_Fery	=L_t*pi*((r_r-self.h_m)**2-(r_r-self.h_m-h_yr)**2)
  
  self.M_Cus=V_Cus*8900
  self.M_Fest	=V_Fest*7700
  self.M_Fesy	=V_Fesy*7700
  self.M_Fery	=V_Fery*7700
  M_Fe		=self.M_Fest+self.M_Fesy+self.M_Fery
  M_gen		=(self.M_Cus)
  K_gen		=self.M_Cus*4.786 
  
  mass_st_lam_s= self.M_Fest+pi*l*7700*((R_st+0.5*h_ys)**2-(R_st-0.5*h_ys)**2) 
  
  W_is			=g1*sin(phi)*(rho*l*self.d_s**2*0.5) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
  W_iis     =g1*sin(phi)*(mass_st_lam_s+V_Cus*8900)/2/N_st
  w_s         =rho*g1*sin(phi)*a_s*N_st
  
  #stator structure
  
  Numers=R_st**3*((0.25*(sin(theta_s)-(theta_s*cos(theta_s)))/(sin(theta_s))**2)-(0.5/sin(theta_s))+(0.5/theta_s))
  Povs=((theta_s/(sin(theta_s))**2)+1/tan(theta_s))*((0.25*R_st/A_st)+(0.25*R_st**3/I_st))
  Qovs=R_st**3/(2*I_st*theta_s*(m2+1))
  Lovs=(R_1s-R_o)*0.5/a_s
  Denoms=I_st*(Povs-Qovs+Lovs) 
 
  self.u_As				=(q*R_st**2/E/self.t_s)*(1+Numers/Denoms)
  X_comp1 = (W_is*l_is**3/12/E/I_arm_axi_s)
  X_comp2 =(W_iis*l_iis**4/24/E/I_arm_axi_s)
  X_comp3 =w_s*l_iiis**4/24/E/I_arm_axi_s
  
  self.y_As       =X_comp1+X_comp2+X_comp3  # axial deflection
  
  self.z_As  =2*pi*(R_st+0.5*self.t_s)*l/(2*N_st)*sigma*(l_is+0.5*self.t_s)**3/3/E/I_arm_tor_s 
  mass_stru_steel  =2*(N_st*(R_1s-R_o)*a_s*rho)
  #val_str_stator		= mass_stru_steel+mass_st_lam_s+((sign(u_As-u_all)+1)*(u_As-u_all)**3*1e100)+(((sign(y_As-y_all)+1)*(y_As-y_all)**3*1e100))+(((sign(z_As-z_all_s)+1)*(z_As-z_all_s)**3*1e100))+(((sign(self.b_st-b_all_s)+1)*(self.b_st-b_all_s)**3*1e50))+((sign((T/2/pi/sigma)-(R_st**2*l))+1)*((T/2/pi/sigma)-(R_st**2*l))**3*1e50)
  val_str_stator		= mass_stru_steel+mass_st_lam_s
  #val_str_cost_stator =	0.50139*mass_stru_steel+0.556*mass_st_lam_s+((sign(u_As-u_all)+1)*(u_As-u_all)**3*1e100)+(((sign(y_As-y_all)+1)*(y_As-y_all)**3*5e17))+(((sign(z_As-z_all_s)+1)*(z_As-z_all_s)**3*1e100))+(((sign(self.b_st-b_all_s)+1)*(self.b_st-b_all_s)**3*1e100))+((sign((T/2/pi/sigma)-(R_st**2*l))+1)*((T/2/pi/sigma)-(R_st**2*l))**3*1e100)
  val_str_cost_stator =	0.50139*mass_stru_steel+0.556*mass_st_lam_s
  #mass_steel  =pi*l*((R_st+h_ys*0.5)**2-(R_st-h_ys*0.5)**2)*rho+N_s*self.h_s*b_s*l*rho
  
  val_str_mass=val_str_rotor+val_str_stator
  
  
  self.Iron=mass_st_lam_s+mass_st_lam
  self.PM=self.mass_PM
  self.Copper=self.M_Cus
  self.Inactive=mass_stru_steel+(N_r*(R_1-R_o)*a_r*rho)
  #self.M_actual=self.Iron+self.PM+self.Copper+self.Inactive
   
  self.TM =val_str_mass+M_gen
  self.M_actual	=self.PM+self.Inactive+(self.M_Cus)+self.Iron
  #self.Stator=((2*pi*self.t_s*l*R_st*rho))+2*N_st*(R_1s-R_o)*a_s*rho
  #self.Rotor=((2*pi*t*l*R*rho)+(N_r*(R_1-R_o)*a_r*rho))
  self.Active=self.Iron+self.Copper+self.mass_PM
  #"% losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  P_Cu		=m*I_s**2*R_s
  
  K_R=1.2
  P_Sc=m*(R_s)*K_R*(I_snom)**2*1  #losses to include skin effect
  
  
  if (self.K_load==0):
  	P_Cusnom_total=P_Cu
  else:
    P_Cusnom_total=P_Sc*self.K_load**2
  
  P_Hyys	=self.M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftys	=self.M_Fesy*((self.B_symax/1.5)**2)*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Fesynom=P_Hyys+P_Ftys
  P_Hyd=self.M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftd=self.M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Festnom=P_Hyd+P_Ftd
  P_ad=0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd ) # additional stray losses due to leakage flux
  pFtm =300 # specific magnet loss
  P_Ftm=pFtm*2*self.p*b_m*self.l_s
  
  P_mech =0.005*P_gennom*0
  P_genlossnom=P_Cusnom_total+P_Festnom+P_Fesynom+P_ad+P_Ftm+P_mech

  self.gen_eff=P_gennom*100/(P_gennom+P_genlossnom)
  self.TL=P_genlossnom
  
  #"% cost of losses"
  val_str_cost=val_str_cost_rotor+val_str_cost_stator
  self.TC=val_str_cost+K_gen

  self.Cost=self.TC
  
  #Thermal model
  rho_c=1.1
  k_thc=1010
  h_air=100
  v_air=15
  tau_air=2
  alpha_1=60 #heat transfer coefficient at stator yoke back
  alpha_2=40 #heat transfer coefficient in air gap
  alpha_3=40 #heat transfer coefficient at end shields
  alpha_4=40 #heat transfer coefficient at the end windings
  alpha_5=40 #heat transfer coefficient at rotor yoke back
  k_fes=0.97
  L_Fe= 38  #thermal conductivity of iron
  L_cu=400 #thermal conductivity of cu
  L_ins =0.2 #thermal conductivity of insulation
  L_coil=1.5 #thermal conductivity of coil
  L_mag=9 #thermal conductivity of magnets
  L_glue=9 #thermal conductivity of magnet glue
  L_GRP=0.7 #thermal conductivity of GRP magnet protection
  h_m0 = 0.1e-3 #thickness of glue
  k_cu=0.8
  h_m1 =0.5e-3
  
  l_u = k_fes*self.l_s
  d_se=dia+2*self.h_s+2*h_ys
  N_Air=pi*d_se/tau_air
  q_vair=v_air*self.l_s*h_air
  q_vc=N_Air*q_vair
  R_eq =1/(q_vc*rho_c*k_thc)
  N_Air=pi*d_se/tau_air
  R_0=1/(q_vc*rho_c*k_thc)
  R_1r=1/(3*self.l_s*b_t*alpha_1)
  R_2=1/(3*self.l_s*b_s*alpha_1)
  R_3=0.5*h_ys/(l_u*b_t*L_Fe)
  R_4 =0.5*h_ys/(l_u*b_s*L_Fe)
  R_5=0.5*b_t/(l_u*h_ys*L_Fe)
  R_6=0.5*b_s/(l_u*h_ys*L_Fe)
  R_7=-R_5/3
  R_8=-R_6/3
  R_9=-R_3/3
  R_10=R_4/3
  R_11=h_i/(self.l_s*b_cu*L_ins)
  R_12 =0.5*(h_cu+2*h_i)/(l_u*b_t*L_Fe)
  R_13=0.5*h_cu/(self.l_s*b_cu*L_coil)
  R_14=0.5*b_t/(l_u*(h_cu+2*h_i)*L_Fe)
  R_15=h_i/(self.l_s*(h_cu)*L_ins)
  R_16=0.5*b_cu/(self.l_s*h_cu*L_coil)
  R_17=-R_14/3
  R_18=-R_16/3
  R_19=-R_12/3
  R_20=-R_13/3
  R_21=(h_s1+h_s2)/(l_u*(0.5*tau_s+0.5*b_t)*L_Fe)
  R_22=1/(self.l_s*(tau_s-b_s1)*alpha_2)
  R_23= (1/(self.l_s*b_m*alpha_2))+(h_m1/(self.l_s*b_m*L_GRP))
  R_24=0.5*self.h_m/(self.l_s*b_m*L_mag)
  R_25=-R_24/3
  R_26=h_m0/(self.l_s*b_m*L_glue)
  R_27=h_yr/(self.l_s*self.tau_p*L_Fe)
  R_28=1/(self.l_s*self.tau_p*alpha_5)
  R_29=1/(pi*(0.5*dia+self.h_s+h_ys)**2*alpha_3)
 
  R_31=0.5*self.l_s/(h_cu*b_cu*k_cu*L_cu)
  R_30= -R_31/3
  R_32=0.5*l_b/(h_cu*b_cu*k_cu*L_cu)
  R_33=-R_32/3
  R_35=2/(l_b*b_cu*L_coil)
  R_36=0.5*h_cu/(l_b*b_cu*L_coil)
  R_34=-R_36/3

  R_38=2/(l_b*h_cu*alpha_4)
  R_39=0.5*b_cu/(l_b*h_cu*L_coil)
  R_37=-R_39/3
  Q=self.N_slots
  R_50=R_0
  R_51=(R_1+R_3)/Q
  R_52=(R_2+R_4)/Q
  R_53=(R_7+R_8+R_9+R_10+0.5*(R_5+R_6))/Q
  R_54=(R_3+R_12)/Q
  R_55=(R_4+R_11+R_13)/Q
  R_56=(R_19+R_17+R_18)/Q+(0.5*(R_14+R_15+R_16)/Q)
  R_57=R_20/Q
  R_58=2*R_12/Q
  R_59=(2*R_13+2*R_11+R_20)/Q
  R_60=(R_12+R_21+R_22)/Q+(0.5*(R_23+R_24)/self.p)
  R_61=(R-24+R_26+R_27+R_28)*0.5/self.p
  R_62=R_29
  R_63=(R_30+0.5*(R_31+R_32))/Q
  R_64a=0.5*(R_34+0.5*(R_36+R_35))/Q
  R_64b=0.5*(R_37+0.5*(R_39+R_38))/Q
  R_64=R_64a*R_64b/(R_64a+R_64b)
  R_65=0.5*R_33/Q
  
  P_0=0
  P_1=b_t*(P_Hyys+P_Ftys)/tau_s
  P_2=b_s*(P_Hyys+P_Ftys)/tau_s
  P_3=0.5*(P_Hyd+P_Ftd)/tau_s
  P_4=0.5*(self.l_s/(self.l_s+l_b))*P_Cu
  P_5=0
  P_6=0
  P_7=(l_b/(self.l_s+l_b))*P_Cu
  P_8=0
  
  P_9=P_3+P_ad
  P_10 =P_4
  P_11=P_Ftm

  P=[[P_0],[P_1],[P_2],[P_3],[P_4],[P_5],[P_6],[P_7],[P_8],[P_9],[P_10],[P_1]]
   
  G=[[(1/(R_50+R_62)+1/R_51+1/R_52),                     -1/R_51,                      -1/R_52,                   0,                    0,                      0,                  			0,              0,                 -1/(R_50+R_62),                0,                          0,                      0],
                           [-1/R_51,                    (1/R_51+1/R_53+1/R_54),        -1/R_53,                -1/R_54,                 0,                      0,                 				0,              0,                          0,                      0,                      0,                       0],
                           [-1/R_52,                     -1/R_53,                 (1/R_52+1/R_53+1/R_55),         0,                    0,                  -1/R_55,               				0,              0,                          0,                      0,                      0,                       0],
             							 [	 0,                        -1/R_54,                         0,               (1/R_54+1/R_56+1/R_58),    -1/R_56,                   0,                				0,              0,                          0,                   -1/R_58,                   0,                       0],
                           [   0,                           0,                            0,                     -1/R_56,         (1/R_56+1/R_57+1/R_63),     -1/R_57,            		-1/R_63,            0,                          0,                     0,                       0,                       0],
                           [   0,                           0,                          -1/R_55,                  0,                -1/R_57,             (1/R_55+1/R_57+1/R_59),				  0,              0,                          0,                     0,                      -1/R_59,                  0],
                           [   0,                           0,                            0,                      0,                 -1/R_63,                   0,               (1/R_63+1/R_63+1/R_65), -1/R_65,                     0,                     0,                      -1/R_63,                  0],
                           [   0,                           0,                            0,                       0,                    0,                      0,                    -1/R_65,         (1/R_64+1/R_65),         -1/R_64,                         0,                       0,                       0],
                           [-1/(R_50+R_62),                 0,                            0,                       0,                    0,                      0,                        0,               -1/R_64,        1/(R_62+R_50)+1/R_61 ,            0,                       0,                   -1/R_61],
                           [   0,                           0,                            0,                      -1/R_58,               0,                      0,                        0,              0,                         0,                   (1/R_58+1/R_56+1/R_60),    -1/R_56,                -1/R_60],
                           [    0,                          0,                            0,                       0,                    0,                      -1/R_59,                -1/R_63,          0,                         0,                    -1/R_56,               (1/R_56+1/R_59+1/R_63),        0],
                           [    0,                          0,                            0,                       0,                     0,                      0,                        0,             0,                      -1/R_61,                   -1/R_60,                     0,                   (1/R_60+1/R_61)]]
  #k_wd,k_wskew,E_p,self.W_1a,B_pm1 
  delta_t= np.linalg.inv(G)*P 
  T_winding=delta_t[7,1]
#Dimensions=["Air gap diameter", "Stator length", "Pole pitch", "Stator slot height","Stator slot width","Stator tooth width", "Stator yoke height", "Rotor yoke height", "Magnet height", "Magnet width", "Peak air gap flux density","Peak stator yoke flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current", "Stator resistance", "Synchronous inductance", "Current density ","Generator Efficiency ", "Active mass", " Stator mass", "Rotor mass","Actual Mass","Stator Turns"]
	#Units=["m","m","mm",",mm","mm","mm","mm","mm","mm","mm","T","T","1","Hz","V","A","p.u","p.u","A/mm^2","%","tons","Turns"]
  #print 2*self.r_s,self.l_s,self.tau_p*1000,self.h_s*1000,b_s*1000,b_t*1000,h_ys*1000,h_yr*1000,h_m*1000,b_m*1000,self.B_g,self.B_symax,p,f,self.E_p,I_s,R_s,L_s,self.J_s,self.gen_eff,self.Active/1000,self.mass_PM/1000,self.Stator/1000,self.Rotor/1000,self.M_actual/1000,self.W_1a,self.M_Cus/1000,self.M_Fesy/1000,self.M_Fest/1000,self.M_Fery/1000
#op=pandas.DataFrame(Data,Dimensions)
  #print u_all,y_all,z_all_s,z_all_r
  print (self.M_Fest+V_Cus*8900)*g1 ,self.mass_PM*g1 # print self.d_s,self.t_ws
  #print delta_t
  Coefficient=pi**2*k_wd*self.A_1*self.B_g/sqrt(2)
  f_t=0.5*Coefficient/pi**2
  A_2=2*m*self.p*q**2*I_s*m/self.tau_p
  