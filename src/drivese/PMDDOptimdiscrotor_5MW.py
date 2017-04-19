"""PMDD.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float
from openmdao.lib.datatypes.api import Array
from numpy import array
from numpy import float 
from scipy.interpolate import interp1d



class PMDDOptimdiscrotor_5MW(Component):

 """ Evaluates the total cost """
 
 r_s = Float(3.49, iotype='in', desc='airgap radius r_s') #mass 3.4 , #cost 3
 l_s = Float(1.5, iotype='in', desc='Stator core length l_s') #mass 2 #cost 1.5
 h_s = Float(0.06, iotype='in', desc='Yoke height h_s') #mass 0.0525
 tau_p =Float(0.1, iotype='in', desc='Pole pitch self.tau_p') #0.065
 B_g = Float(0.9, iotype='out', desc='Peak air gap flux density B_g')
 B_symax = Float(0.9, iotype='out', desc='Peak Yoke flux density B_symax')
 B_tmax=Float(0.1, iotype='out', desc='Peak Teeth flux density')
 B_rymax=Float(0.01, iotype='out', desc='Peak Rotor yoke flux density')
 B_smax=Float(0.01, iotype='out', desc='Peak Stator flux density')
 B_pm1=Float(0.01, iotype='out', desc='Fundamental component of peak air gap flux density')
 h_m =Float(0.0205, iotype='in', desc='Magnet height')
 h_yr= Float(0.045, iotype='in', desc='thickness of arms t') #nn  0.15
 h_ys= Float(0.087, iotype='in', desc='thickness of arms t') #mass 0.15
 t_d =Float(0.16, iotype='in', desc='disc thickness self.t_d') #mass 0.1
 n_s =Float(5, iotype='in', desc='number of arms n') #mass 5
 t_s= Float(0.0, iotype='out', desc='stator back iron t') #mass 0
 b_st = Float(0.46, iotype='in', desc='arm width b_r') #mass 0.5
 d_s = Float(0.32, iotype='in', desc='arm depth d_r') #mass 0.43
 t_ws =Float(0.15, iotype='in', desc='arm depth thickness self.t_wr' ) #mass 0.015
 W_1a =Float(200, iotype='out', desc='Number of turns in the stator winding') #mass 250
 A_1 =Float(0, iotype='out', desc='Electrical loading')
 A_Cuscalc=Float(0.01, iotype='out', desc='Conductor cross-section mm^2')
 M_actual=Float(0.01, iotype='out', desc='Actual mass')
 #p=Float(0.01, iotype='out', desc='No of pole pairs')
 #f=Float(0.01, iotype='out', desc='Output frequency')
 E_p=Float(0.01, iotype='out', desc='Stator phase voltage')
 f=Float(0.01, iotype='out', desc='Generator output frequency')
 #I_s=Float(0.01, iotype='out', desc='Generator output phase current')
 #R_s=Float(0.01, iotype='out', desc='Stator resistance')
 #L_s=Float(0.01, iotype='out', desc='Stator synchronising inductance')
 J_s=Float(2, iotype='out', desc='Current density')
 #T_winding=Float(0.01, iotype='out', desc='Winding Temp')
 TC= Float(0.01, iotype='out', desc='Total cost')
 TM=Float(0.01, iotype='out', desc='Total Mass')
 K_rad=Float(0.01, iotype='out', desc='K_rad')  
 TL=Float(0.0, iotype='out', desc='Total loss')
 gen_eff=Float(0.01, iotype='out', desc='Generator efficiency')
 A=Float(0.01, iotype='out', desc='Generator efficiency')
 p= Float(0.0, iotype='out', desc='pole pairs')
 u_Ar	=Float(0.1, iotype='out', desc='Radial deflection-rotor')
 u_As	=Float(0.1, iotype='out', desc='Radial deflection-stator')
 u_all_r=Float(0.1, iotype='out', desc='Allowable Radial deflection-rotor')
 u_all_s=Float(0.1, iotype='out', desc='Allowable Radial deflection-stator')
 y_Ar	=Float(0.1, iotype='out', desc='Axial deflection-rotor')
 y_As	=Float(0.1, iotype='out', desc='Axial deflection-stator')
 y_all=Float(0.1, iotype='out', desc='Allowable axial deflection')
 z_As	=Float(0.1, iotype='out', desc='Circum deflection-stator')
 z_all_s=Float(0.1, iotype='out', desc='Allowable Circum deflection-stator')
 TC1	=Float(0.1, iotype='out', desc='Torque constraint')
 TC2=Float(0.1, iotype='out', desc='Torque constraint-rotor')
 TC3=Float(0.1, iotype='out', desc='Torque constraint-stator')
 b_all_r			=Float(0.1, iotype='out', desc='rotor arm constraint')
 b_all_s			=Float(0.1, iotype='out', desc='Stator arm constraint') 
 Cost			=Float(0.1, iotype='out', desc='Cost')
 #P = Array(array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]), dtype=float, shape=(12,1), iotype='out')
 N_s_max			=Float(0.1, iotype='out', desc='Maximum number of turns per coil')
 Slot_aspect_ratio=Float(0.1, iotype='out', desc='Slot aspect ratio')
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  tau_p =self.tau_p
  B_g = self.B_g
  B_symax = self.B_symax
  t_s=self.t_s
  h_yr= self.h_yr
  h_ys = self.h_ys
  t_d=self.t_d
  n_s =self.n_s
  b_st = self.b_st
  d_s  = self.d_s
  t_ws =self.t_ws
  TC =self.TC
  TM=self.TM
  #b_s =self.b_s
  #b_t= self.b_t
  
  #h_m =self.h_m
  #b_m =self.b_m
  f=self.f
  E_p=self.E_p
  #I_s=self.I_s
  #R_s=self.R_s
  #L_s=self.L_s
  J_s=self.J_s
  M_actual=self.M_actual
  gen_eff =self.gen_eff
  #P=self.P
  TL=self.TL
  K_rad=self.K_rad
  #P=self.P
  W_1a=self.W_1a
  A=self.A
  p=self.p
  Cost=self.Cost
  u_Ar	=self.u_Ar
  u_As	=self.u_As
  z_As=self.z_As
  u_all_r=self.u_all_r
  u_all_s=self.u_all_s
  y_Ar	=self.y_Ar
  y_As	=self.y_As
  y_all=self.y_all
  z_all_s=self.z_all_s
  TC1	=self.TC1
  TC2=self.TC2
  TC3=self.TC3
  b_all_s			=self.b_all_s
  B_tmax= self.B_tmax
  B_rymax=self.B_rymax
  B_smax=self.B_smax
  B_pm1=self.B_pm1
  N_s_max=self.N_s_max
  A_Cuscalc=self.A_Cuscalc
  Slot_aspect_ratio=self.Slot_aspect_ratio
  
  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan, cosh,sinh
  import numpy as np
  from numpy import sign

  rho    =7850                # Kg/m3 steel density
  rho_PM =7450                # Kg/m3 magnet density
  B_r    =1.2                 # Tesla remnant flux density 
  g1     =9.81                # m/s^2 acceleration due to gravity
  E      =2e11                # N/m^2 young's modulus
  sigma  =40e3                # shear stress assumed
  ratio  =0.7                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
  mu_0   =pi*4e-7        # permeability of free space
  mu_r   =1.06
  phi    =90*2*pi/360 
  cofi   =0.9                 # power factor
  K_Cu   =4.7856                  # Unit cost of Copper 
  K_Fe   =0.556                   # Unit cost of Iron 
  K_pm   =25                  # Unit cost of magnet
  K_elec =10                  # electricity price in c/kWH
  K_conv =1.1                  # Converter cost
  years  =8760                 # number of hours in a year
  h_sy0  =0
  #h_sy   =0.04
  h_w    =0.005
  h_i=0.001 # coil insulation thickness
  h_s1= 0.001
  h_s2=0.004
  h_s3=self.h_s-h_s1-h_s2
  h_cu=(h_s3-4*h_i)*0.5
  m      =3
  q1     =1
  b_s_tau_s=0.45
  k_sfil =0.65								 # Slot fill factor
  P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  B_rymax=1.4
  rho_Cu=1.8*10**(-8)*1.4
  k_fes =0.9
  R_o = 0.43
  k_sfil =0.65								 # Slot fill factor
  T = 4143289.8
  P_gennom = 5.0e6
  P_gridnom=5e6
  E_pnom   = 3000
  n_nom = 12.1
  P_convnom=0.03*P_gridnom
  
  gear      =1   
  B_curve=(0.000,0.050,0.130,0.152,0.176,0.202,0.229,0.257,0.287,0.318,0.356,0.395,0.435,0.477,0.523,0.574,0.627,0.683,0.739,0.795,0.850,0.906,0.964,1.024,1.084,1.140,1.189,1.228,1.258,1.283,1.304,1.324,1.343,1.360,1.375,1.387,1.398,1.407,1.417,1.426,1.437,1.448,1.462,1.478,1.495,1.513,1.529,1.543,1.557,1.570,1.584,1.600,1.617,1.634,1.652,1.669,1.685,1.701,1.717,1.733,1.749,1.769,1.792,1.820,1.851,1.882,1.909,1.931,1.948,1.961,1.972,1.981)
  H_curve=(0,10,20,23,25,28,30,33,35,38,40,43,45,48,52,56,60,66,72,78,87,97,112,130,151,175,200,225,252,282,317,357,402,450,500,550,602,657,717,784,867,974,1117,1299,1515,1752,2000,2253,2521,2820,3167,3570,4021,4503,5000,5503,6021,6570,7167,7839,8667,9745,11167,12992,15146,17518,20000,22552,25417,28906,33333,38932)
  U_Nrated=3.3e3
  constant=0.5
  
       
  #q					= self.B_g**2/2/mu_0   # normal stress
  l					= self.l_s                          #l-stator core length
  l_u       =k_fes * self.l_s                   #useful iron stack length
  We				=self.tau_p
  l_b       = 2*self.tau_p  #end winding length
  l_e       =self.l_s+2*0.001*self.r_s     # equivalent core length
  b_m  =0.7*self.tau_p 
 
  
  t=self.h_yr
  self.t_s=self.h_ys
  
  # stator with arms moments
  a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws))
  A_st      =l*self.t_s                    # cross-sectional area of rotor cylinder
  N_st			= round(self.n_s)
  theta_s		=pi/N_st                           # half angle between spokes
  I_st      =l*self.t_s**3/12 
  I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
  I_arm_tor_s= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
  			
                               # inner radius of rotor cylinder
  dia				=  2*self.r_s              # air gap diameter
  g         =  0.001*dia
  R					= self.r_s-g-self.h_m-t
  R_1				= R-t*0.5  
  f_d=40000
  self.K_rad=self.l_s/(2*self.r_s)
  D_g_anal=(2*T/pi/f_d/self.K_rad)**(1/3)
  c					=R/500
  R_st 			=self.r_s+self.h_s+self.h_ys*0.5
  K					=4*(sin(ratio*pi/2))/pi
  b_s1			=  0.003
  tau_s		  =  (self.tau_p)/3/q1    #slot pitch
  b_s				=  b_s_tau_s*tau_s    #slot width 
  self.Slot_aspect_ratio=self.h_s/b_s
  b_t	      =  tau_s-(b_s)          #tooth width
  k_t       =  b_t/tau_s
  b_so			=  b_s
  b_cu			=  b_s -2*h_i         # conductor width
  gamma			=  4/pi*(b_so/2/(g+self.h_m/mu_r)*atan(b_so/2/(g+self.h_m/mu_r))-log(sqrt(1+(b_so/2/(g+self.h_m/mu_r))**2)))
  k_C				=  tau_s/(tau_s-gamma*(g+self.h_m/mu_r))   # carter coefficient
  g_eff			=  k_C*(g+self.h_m/mu_r)
  alpha_p		=  (pi/2)*0.7
  
  #self.B_g  = B_r*(self.h_m/mu_r)/(self.h_m+c)*K
  self.B_pm		=  B_r*self.h_m/mu_r/(g_eff)
  self.B_g=  B_r*self.h_m/mu_r/(g_eff)*(4/pi)*sin(alpha_p)
  self.B_symax=self.B_g*b_m*l_e/(2*self.h_ys*l_u)
  self.B_rymax=self.B_g*b_m*l_e/(2*self.h_yr*l)
  self.B_tmax	=self.B_g*tau_s/b_t
  q					= self.B_g**2/2/mu_0   # normal stress
  h_interp=interp1d(B_curve,H_curve,fill_value='extrapolate')
  
  #h_m  =c/B_r/K/((1/self.B_g)-(1/(B_r*K)))  
  
  R_1s      = R_st-self.t_s*0.5
             
  k_2       = sqrt(I_st/A_st) # radius of gyration
  
  m2        =(k_2/R_st)**2                                  
  l_is      =R_st-R_o
  l_iis     =l_is 
  l_iiis    =l_is
  b=R_o
  R_b=R
  R_a=R+self.h_yr
  a=R
  a_1=R_b
  
  self.b_all_s			=2*pi*R_o/N_st                           
  
  mass_PM   =(2*pi*R_a*l*self.h_m*ratio*rho_PM)           # magnet mass
                                            
  #w_r					=rho*g1*sin(phi)*a_r
 
  #self.h_yr =0.5*B_r/B_rymax*self.tau_p*ratio*(self.h_m/mu_r)/((self.h_m/mu_r)+c)  # rotor yoke height
  v = 0.3
  r_r				=  self.r_s-g              #rotor radius
  self.p		=  round(pi*dia/(2*self.tau_p))   # number of pole pairs
  N_slot		=  2
  self.N_slots				= 2*self.p*q1*m 
  N_conductors=self.N_slots*2
  self.W_1a=N_conductors/2/3
  
  
 
  lamb   =(3*(1-v**2)/R_a**2/self.h_yr**2)**0.25

  # cylindrical shell function and circular plate parameters for disc rotor
  #C_1=0.5*(1+v)*(b/a)*log(a/b)+(1-v)*0.25*((a/b)-(b/a))
  C_2=cosh(lamb*self.l_s)*sin(lamb*self.l_s)+sinh(lamb*self.l_s)*cos(lamb*self.l_s)
  C_3=sinh(lamb*self.l_s)*sin(lamb*self.l_s)
  C_4=cosh(lamb*self.l_s)*sin(lamb*self.l_s)-sinh(lamb*self.l_s)*cos(lamb*self.l_s)
  C_11=sinh(lamb*self.l_s)**2-(sin(lamb*self.l_s)**2)
  C_13=cosh(lamb*self.l_s)*sinh(lamb*self.l_s)-cos(lamb*self.l_s)*sin(lamb*self.l_s)
  C_14=(sinh(lamb*self.l_s)**2+sin(lamb*self.l_s)**2)
  C_a1=cosh(lamb*self.l_s*0.5)*cos(lamb*self.l_s*0.5) 
  C_a2=cosh(lamb*self.l_s*0.5)*sin(lamb*self.l_s*0.5)+sinh(lamb*self.l_s*0.5)*cos(lamb*self.l_s*0.5) 
  F_1_x0=cosh(lamb*0)*cos(lamb*0)
  F_1_ls2=cosh(lamb*0.5*self.l_s)*cos(lamb*0.5*self.l_s)
  F_2_x0=cosh(lamb*0)*sin(lamb*0)+sinh(lamb*0)*cos(lamb*0)
  F_2_ls2=cosh(lamb*self.l_s/2)*sin(lamb*self.l_s/2)+sinh(lamb*self.l_s/2)*cos(lamb*self.l_s/2)
  if (self.l_s<2*a):
  	a=self.l_s/2
  else:
  	a=self.l_s*0.5-1
  F_a4_x0=cosh(lamb*(0))*sin(lamb*(0))-sinh(lamb*(0))*cos(lamb*(0))
  
  F_a4_ls2=cosh(lamb*(0.5*self.l_s-a))*sin(lamb*(0.5*self.l_s-a))-sinh(lamb*(0.5*self.l_s-a))*cos(lamb*(0.5*self.l_s-a))
  D_r=E*self.h_yr**3/12/(1-v**2)  
  D_ax=E*self.t_d**3/12/(1-v**2)
  
  
  Part_1 =R_b*((1-v)*R_b**2+(1+v)*R_o**2)/(R_b**2-R_o**2)/E
  
  Part_2 =(C_2*C_a2-2*C_3*C_a1)/2/C_11
 
  
  Part_3 = (C_3*C_a2-C_4*C_a1)/C_11
  
  Part_4 =((0.25/D_r/lamb**3))
  
  Part_5=q*R_b**2/(E*(R_a-R_b))
  
  f_d = Part_5/(Part_1-self.t_d*(Part_4*Part_2*F_2_ls2-Part_3*2*Part_4*F_1_ls2-Part_4*F_a4_ls2))
  fr=f_d*self.t_d
 
  phi=90*pi/180  
  W=0.5*g1*sin(phi)*((self.l_s-self.t_d)*self.h_yr*rho)
  w=rho*g1*sin(phi)*self.t_d
  a_i=R_o
 
  self.u_Ar				=Part_5+fr/2/D_r/lamb**3*((-F_1_x0/C_11)*(C_3*C_a2-C_4*C_a1)+(F_2_x0/2/C_11)*(C_2*C_a2-2*C_3*C_a1)-F_a4_x0/2)
  
  self.TC1=T/(2*pi*sigma)
  self.TC2=R**2*l
  self.TC3=R_st**2*l
  
  
  self.u_all_r    =0.05*R  # allowable radial deflection
  self.u_all_s    =0.05*R_st 
  self.y_all     =2*l/100    # allowable axial deflection

  C_2p= 0.25*(1-(((R_o/R)**2)*(1+(2*log(R/R_o)))))
  C_3p=(R_o/4/R)*((1+(R_o/R)**2)*log(R/R_o)+(R_o/R)**2-1)
  C_6= (R_o/4/R_a)*((R_o/R_a)**2-1+2*log(R_a/R_o))
  C_5=0.5*(1-(R_o/R)**2)
  C_8= 0.5*(1+v+(1-v)*((R_o/R)**2));
  C_9=(R_o/R)*(0.5*(1+v)*log(R/R_o) + (1-v)/4*(1-(R_o/R)**2));
  L_17=0.25*(1 - (1-v)*(1-(R_o/a_1)**4)/4 - ((R_o/a_1)**2)*(1 + (1-v)*log(a_1/R_o)));
  L_11=(1 + 4*(R_o/a_1)**2 - 5*(R_o/a_1)**4 - 4*((R_o/a_1)**2)*log(a_1/R_o)*(2+(R_o/a_1)**2))/64
  L_14=(1/16)*(1-(R_o/R_b)**4-4*(R_o/R_b)**2*log(R_b/R_o))
  #y_ai=-(W*R_b**4/(R_o*D_ax))*(C_2p*C_6/C_5-C_3p)
  y_ai=-W*(a_1**3)*(C_2p*(C_6*a_1/R_o - C_6)/C_5 - a_1*C_3p/R_o +C_3p)/D_ax;
  M_rb=-w*R**2*(C_9*(R**2-R_o**2)*0.5/R/R_o-L_17)/C_8
  Q_b=w*0.5*(R**2-R_o**2)/R_o
  y_aii=M_rb*R_a**2*C_2p/D_ax+Q_b*R_a**3*C_3p/D_ax-w*R_a**4*L_11/D_ax
  self.y_Ar=abs(y_ai+y_aii)
  z_all     =0.05*2*pi*R_st/360  # allowable torsional deflection
  #z_A_r       =(2*pi*R*l/N_r)*sigma*l_ir**3/3/E/I_arm_tor       # circumferential deflection
  
 
 
  val_str_cost_rotor	= 95*mass_PM+0.556*(pi*l*(R_a**2-R**2)*7700)+0.50139*pi*self.t_d*(R**2-R_o**2)*rho
  val_str_rotor		= mass_PM+(pi*l*(R_a**2-R**2)*7700)+pi*self.t_d*(R**2-R_o**2)*rho
  
  
  self.K_rad=self.l_s/dia 

                     
  om_m			=  gear*2*pi*n_nom/60
  om_e			=  self.p*om_m/2
  self.f    			=  n_nom*self.p/60
 
  
  
  v_ys=constant*(self.tau_p+pi*(self.h_s+0.5*self.h_ys)/self.p)*h_interp(self.B_symax)
  v_d=h_interp(self.B_tmax)*(h_s3+0.5*h_s2)+h_interp(self.B_tmax)*(0.5*h_s2+h_s1)
  v_yr=constant*(self.tau_p+pi*(g+self.h_m+0.5*self.h_yr)/self.p)*h_interp(self.B_rymax)
  v_m=self.h_m*self.B_g/mu_r/mu_0
  v_g       =  g_eff*self.B_g/mu_0
  
  
  # stator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  k_wd			= sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
  beta_skew	= tau_s/self.r_s;
  k_wskew		= sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  L_t       =self.l_s+2*self.tau_p
  #W_1a      =2*3*3000/(sqrt(3)*sqrt(2)*60*k_wd*B_pm1*2*L_t*self.tau_p) 
  #k_wskew	=1
  self.E_p	= 2*(self.W_1a)*L_t*self.r_s*k_wd*k_wskew*om_m*self.B_g/sqrt(2)
  
  l_Cus			= 2*(self.W_1a)*(2*self.tau_p+L_t)
  A_s				= b_s*(self.h_s-h_w)*q1*self.p
  A_scalc   = b_s*1000*(self.h_s*1000-h_w*1000)*q1*self.p
  A_Cus			= A_s*k_sfil/(self.W_1a)
  self.A_Cuscalc = A_scalc *k_sfil/(self.W_1a)
  R_s	= l_Cus*rho_Cu/A_Cus

  L_m				= 2*m*k_wd**2*(self.W_1a)**2*mu_0*self.r_s*L_t/pi/g_eff/self.p**2
  L_ssigmas	= 2*mu_0*L_t*(self.W_1a)**2/self.p/q1*((self.h_s-0.004)/(3*b_s)+0.004/b_so)  #slot leakage inductance
  L_ssigmaew= mu_0*(self.W_1a)**2/self.p*1.2*(2/3*self.tau_p+0.01)                    #end winding leakage inductance
  L_ssigmag	= 2*mu_0*L_t*(self.W_1a)**2/self.p/q1*(5*(g+self.h_m)/b_so)/(5+4*(g+self.h_m)/b_so)  #tooth tip leakage inductance
  L_ssigma	= (L_ssigmas+L_ssigmaew+L_ssigmag)   #stator leakage inductance
  
  L_s  = L_m+L_ssigma

  Z=(P_gennom/(m*self.E_p))
  G=((-((om_e*L_s)**2)*Z**2+self.E_p**2))
  
  #I_s= sqrt(Z**2+((self.E_p-G/(om_e*L_s)**2)**2))
  I_s	= Z
  self.J_s	= I_s/self.A_Cuscalc
  
  I_snom		=P_gennom/m/self.E_p/cofi 

  I_qnom		=P_gennom/(m*self.E_p)
  X_snom		=om_e*(L_m+L_ssigma)
  self.B_smax=sqrt(2)*I_s*mu_0/g_eff
  pp=1  #number of parallel paths
  N_s_j=sqrt(3)*(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*self.J_s*pp*U_Nrated/P_gennom
  N_s_s=(self.h_s-h_w)*tau_s*(1-k_t)*k_sfil*0.5
  N_s_EL=self.A_1*pi*(dia-g)/(q*self.p*m)
  self.N_s_max=min([N_s_j,N_s_s,N_s_EL])
  self.A_1 = 6*self.W_1a*I_s/(pi*dia*l)
  

  #" masses %%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%"
  V_Cus 	=m*l_Cus*A_Cus     # copper volume
  V_Fest	=L_t*2*self.p*q1*m*b_t*self.h_s   # volume of iron in stator tooth
  V_Fesy	=L_t*pi*((self.r_s+self.h_s+self.h_ys+h_sy0)**2-(self.r_s+self.h_s)**2) # volume of iron in stator yoke
  V_Fery	=L_t*pi*((r_r-self.h_m)**2-(r_r-self.h_m-self.h_yr)**2)
  
  M_Cus		=V_Cus*8900
  M_Fest	=V_Fest*7700
  M_Fesy	=V_Fesy*7700
  M_Fery	=V_Fery*7700
  M_Fe		=M_Fest+M_Fery
  M_gen		=(M_Cus)
  K_gen		=M_Cus*4.7856
  
  mass_st_lam_s=M_Fest+pi*l*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2)*7700                                     # mass of rotor yoke steel  
  #W				=g1*sin(phi)*((rho*(2*pi*R/N_r)*l*t)+(mass_PM/N_r)+(mass_st_lam/N_r))  # weight of rotor cylinder
  
  
  W_is			=g1*sin(phi)*(rho*l*self.d_s**2*0.5) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
  W_iis     =g1*sin(phi)*(mass_st_lam_s+M_Cus)/N_st/2
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
  val_str_stator		= mass_stru_steel+mass_st_lam_s
  val_str_cost_stator =	0.50139*mass_stru_steel+0.556*mass_st_lam_s
  
  val_str_mass=val_str_rotor+val_str_stator
  
  self.TM =val_str_mass+M_gen
 
  self.TC=val_str_cost_rotor+val_str_cost_stator+K_gen
  self.Cost=self.TC
  
  
  
  r_m     	=  self.r_s+h_sy0+self.h_ys+self.h_s #magnet radius
  
  
  K_R=1.2
 #"% losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  P_Sc=m*(R_s)*K_R*(I_snom)**2*1  #losses to include skin effect
  P_Cusnom_total=P_Sc
  P_Hyys	=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftys	=M_Fesy*((self.B_symax/1.5)**2)*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Fesynom=P_Hyys+P_Ftys
  P_Hyd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Festnom=P_Hyd+P_Ftd
  P_ad=0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd ) # additional stray losses due to leakage flux
  pFtm =300 # specific magnet loss
  P_Ftm=pFtm*2*self.p*b_m*self.l_s
  P_mech=0.005*P_gennom*0
  self.A= self.N_slots*q1*m*I_s/self.tau_p
  P_genlossnom=P_Cusnom_total+P_Festnom+P_Fesynom+P_ad+P_Ftm
  #P_conv0=P_convnom/31
  #P_conv1=K_conv*P_convnom/31*(10*I_snomex/I_snom+5*((I_snomex)/I_snom)**2)
  #P_conv2=K_conv*P_convnom/31*(10*I_snomex/I_snom*self.E_p/E_pnom+5*(I_snomex/I_snom*E_p/E_pnom)**2)
  #P_conv=P_conv0+P_conv1+P_conv2
  self.gen_eff=P_gennom*100/(P_gennom+P_genlossnom)
  self.TL=P_genlossnom
  #"% cost of losses"

  
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
  d_se=dia+2*self.h_s+2*self.h_ys
  N_Air=pi*d_se/tau_air
  q_vair=v_air*self.l_s*h_air
  q_vc=N_Air*q_vair
  R_eq =1/(q_vc*rho_c*k_thc)
  N_Air=pi*d_se/tau_air
  R_0=1/(q_vc*rho_c*k_thc)
  R_1=1/(3*self.l_s*b_t*alpha_1)
  R_2=1/(3*self.l_s*b_s*alpha_1)
  R_3=0.5*self.h_ys/(l_u*b_t*L_Fe)
  R_4 =0.5*self.h_ys/(l_u*b_s*L_Fe)
  R_5=0.5*b_t/(l_u*self.h_ys*L_Fe)
  R_6=0.5*b_s/(l_u*self.h_ys*L_Fe)
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
  R_27=self.h_yr/(self.l_s*self.tau_p*L_Fe)
  R_28=1/(self.l_s*self.tau_p*alpha_5)
  R_29=1/(pi*(0.5*dia+self.h_s+self.h_ys)**2*alpha_3)
 
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
  Q=N_slot
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
  P_4=0.5*(self.l_s/(self.l_s+l_b))*P_Sc
  P_5=0
  P_6=0
  P_7=(l_b/(self.l_s+l_b))*P_Sc
  P_8=0
  
  P_9=P_3+P_ad
  P_10 =P_4
  P_11=P_Ftm

  #P=[[P_0],[P_1],[P_2],[P_3],[P_4],[P_5],[P_6],[P_7],[P_8],[P_9],[P_10],[P_1]]
   
  #G=[[(1/(R_50+R_62)+1/R_51+1/R_52),                     -1/R_51,              a        -1/R_52,                   0,                    0,                      0,                  			0,              0,                 -1/(R_50+R_62),                0,                          0,                      0],
  #                         [-1/R_51,                    (1/R_51+1/R_53+1/R_54),        -1/R_53,                -1/R_54,                 0,                      0,                 				0,              0,                          0,                      0,                      0,                       0],
  #                         [-1/R_52,                     -1/R_53,                 (1/R_52+1/R_53+1/R_55),         0,                    0,                  -1/R_55,               				0,              0,                          0,                      0,                      0,                       0],
  #           							 [	 0,                        -1/R_54,                         0,               (1/R_54+1/R_56+1/R_58),    -1/R_56,                   0,                				0,              0,                          0,                   -1/R_58,                   0,                       0],
  #                         [   0,                           0,                            0,                     -1/R_56,         (1/R_56+1/R_57+1/R_63),     -1/R_57,            		-1/R_63,            0,                          0,                     0,                       0,                       0],
  #                         [   0,                           0,                          -1/R_55,                  0,                -1/R_57,             (1/R_55+1/R_57+1/R_59),				  0,              0,                          0,                     0,                      -1/R_59,                  0],
  #                         [   0,                           0,                            0,                      0,                 -1/R_63,                   0,               (1/R_63+1/R_63+1/R_65), -1/R_65,                     0,                     0,                      -1/R_63,                  0],
  #                         [   0,                           0,                            0,                       0,                    0,                      0,                    -1/R_65,         (1/R_64+1/R_65),         -1/R_64,                         0,                       0,                       0],
  #                         [-1/(R_50+R_62),                 0,                            0,                       0,                    0,                      0,                        0,               -1/R_64,        1/(R_62+R_50)+1/R_61 ,            0,                       0,                   -1/R_61],
  #                         [   0,                           0,                            0,                      -1/R_58,               0,                      0,                        0,              0,                         0,                   (1/R_58+1/R_56+1/R_60),    -1/R_56,                -1/R_60],
  #                         [    0,                          0,                            0,                       0,                    0,                      -1/R_59,                -1/R_63,          0,                         0,                    -1/R_56,               (1/R_56+1/R_59+1/R_63),        0],
  #                         [    0,                          0,                            0,                       0,                     0,                      0,                        0,             0,                      -1/R_61,                   -1/R_60,                     0,                   (1/R_60+1/R_61)]]
  
  #delta_t= np.linalg.inv(G)*P 
  #self.T_winding=delta_t[7,1]
  
  Active=M_gen+mass_PM
  Stator=mass_stru_steel+mass_st_lam_s+M_Cus
  Rotor=(pi*l*(R_a**2-R**2)*7700)+pi*self.t_d*(R**2-R_o**2)*rho+mass_PM
  #print w,R,C_9,R_o,L_17,C_8
  print mass_PM*g1,(M_Fest+M_Cus)*g1,self.TM,mass_stru_steel+(pi*(R**2-R_o**2)*self.t_d*rho)
  