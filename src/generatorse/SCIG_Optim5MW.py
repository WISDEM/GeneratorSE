"""DDSG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float
from scipy.interpolate import interp1d
import numpy as np


class SCIG_Optim5MW(Component):
 """ Evaluates the total cost """
 r_s = Float(0.55, iotype='in', desc='airgap radius r_s')
 l_s = Float(1.3, iotype='in', desc='Stator core length l_s')
 h_s = Float(0.09, iotype='in', desc='Yoke height h_s')
 h_r = Float(0.05, iotype='in', desc='Rotor slot height')
 tau_p =Float(0.2, iotype='out', desc='Pole pitch self.tau_p')
 I_0 = Float(140, iotype='in', desc='No load current')
 S_N = Float(-0.002, iotype='out', desc='Slip')
 h_ys=Float(0.0, iotype='out', desc='Back iron thickness')
 h_yr=Float(0.0, iotype='out', desc='Back iron thickness')
 B_g = Float(0., iotype='out', desc='Peak air gap flux density B_g')
 B_symax = Float(1.4, iotype='in', desc='Peak Yoke flux density B_ymax')
 B_rymax = Float(0, iotype='out', desc='Peak Rotor Yoke flux density B_ymax')
 B_tsmax = Float(0, iotype='out', desc='Peak stator tooth flux density B_ymax')
 B_trmax = Float(0, iotype='out', desc='Peak rotor tooth flux density B_ymax')
 W_1a = Float(0, iotype='out', desc='Peak Yoke flux density B_ymax')
 f=Float(0, iotype='out', desc='output frequency')
 M_actual=Float(0.0, iotype='out', desc='Actual mass')
 Q_r = Float(0, iotype='out', desc='Peak Yoke flux density B_ymax')
 p=Float(0, iotype='out', desc='Output frequency')
 E_p=Float(0, iotype='out', desc='Stator phase voltage')
 J_s=Float(0.0, iotype='out', desc='Current density')
 TC= Float(0.0, iotype='out', desc='Total cost')
 TL=Float(0.0, iotype='out', desc='Total Loss')
 K_rad=Float(0.0, iotype='out', desc='K_rad')
 TM=Float(0.01, iotype='out', desc='Total loss')
 M_gen=Float(0.01, iotype='out', desc='Active mass')
 M_Str=Float(0.01, iotype='out', desc='Structural mass')
 gen_eff=Float(0.01, iotype='out', desc='Generator efficiency')
 
 J_r=Float(0.01, iotype='out', desc='Rotor current density')
 TC1=Float(0.01, iotype='out', desc='Torque constraint')
 TC2=Float(0.01, iotype='out', desc='Torque constraint')
 A_1=Float(0.01, iotype='out', desc='Specific current loading')
 J_s=Float(0.01, iotype='out', desc='Stator current density')
 J_r=Float(0.01, iotype='out', desc='Rotor current density')
 Lambda_ratio=Float(0.01, iotype='out', desc='Stack length ratio')
 D_ratio=Float(0.01, iotype='out', desc='Stator diameter ratio')
 A_Cuscalc=Float(0.01, iotype='out', desc='Stator conductor cross-section')
 A_Curcalc=Float(0.01, iotype='out', desc='Rotor conductor cross-section')
 N_slots=Float(0.01, iotype='out', desc='Stator slots')
 Slot_aspect_ratio1=Float(0.01, iotype='out', desc='Slot apsect ratio')
 Slot_aspect_ratio2=Float(0.01, iotype='out', desc='Slot apsect ratio')
 D_ratio_UL=Float(0.01, iotype='out', desc='Dia ratio upper limit')
 D_ratio_LL=Float(0.01, iotype='out', desc='Dia ratio Lower limit')
 Lambda_ratio_UL=Float(0.01, iotype='out', desc='Aspect ratio upper limit')
 Lambda_ratio_LL=Float(0.01, iotype='out', desc='Aspect ratio Lower limit')
 r_rmax=Float(0.01, iotype='out', desc='Maximum rotor radius')
 r_r=Float(0.01, iotype='out', desc='Maximum rotor radius')
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  h_r=self.h_r
  tau_p =self.tau_p
  B_g = self.B_g
  h_ys=self.h_ys
  h_yr=self.h_yr
  B_symax = self.B_symax
  B_rymax=self.B_rymax
  B_tsmax=self.B_tsmax
  B_trmax=self.B_trmax
  S_N=self.S_N
  TC=self.TC
  TL=self.TL
  M_actual=self.M_actual
  E_p=self.E_p
  W_1a=self.W_1a
  p=self.p
  J_s=self.J_s
  gen_eff =self.gen_eff
  TM=self.TM
  K_rad=self.K_rad
  M_gen=self.M_gen
  M_Str=self.M_Str
  Q_r=self.Q_r
  I_0=self.I_0
  W_1a=self.W_1a
  TC1=self.TC1
  TC2=self.TC2
  N_slots=self.N_slots
  Q_r=self.Q_r
  A_1=self.A_1
  
  J_r=self.J_r
  Lambda_ratio=self.Lambda_ratio
  D_ratio=self.D_ratio
  D_ratio_UL=self.D_ratio_UL
  D_ratio_LL=self.D_ratio_LL
  f=self.f
  A_Curcalc=self.A_Curcalc
  A_Cuscalc=self.A_Cuscalc
  Slot_aspect_ratio1=self.Slot_aspect_ratio1
  Slot_aspect_ratio2=self.Slot_aspect_ratio2
  Lambda_ratio_LL= self.Lambda_ratio_LL 
  Lambda_ratio_UL= self.Lambda_ratio_UL
  r_rmax=self.r_rmax 
  r_r=self.r_r
  

 
 

  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign
  
  g1     			=9.81                	# m/s^2 acceleration due to gravity
  sigma  			=21.5e3                # shear stress [chapter Design of Rotating Electrical Machines
  mu_0   			=pi*4e-7        				# permeability of free space
  cofi   			=0.9                 	# power factor
  K_Cu   			=4.786                  # Unit cost of Copper 
  K_Fe   			=0.556                    # Unit cost of Iron 
  cstr   			=0.50139                   # specific cost of a reference structure
  h_w    			= 0.005
  m      			=3                     # Number of phases
  
  q1     			=6										 # Number of slots per pole per phase
  b_s_tau_s   =0.45										#
  b_r_tau_r   =0.45										#
  self.S_N    =-0.002									#
  P_gennom    = 5e6								#
  y_tau_p     =12./15										#
  k_y1        =sin(pi*0.5*y_tau_p)						#
  k_q1        =sin(pi/6)/q1/sin(pi/6/q1)# 
  k_wd        =k_y1*k_q1							#
  P_Fe0h      =4			               	#specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e      =1			               	#specific hysteresis losses W/kg @ 1.5 T @50 Hz
  rho_Cu      =1.8*10**(-8)*1.4
  
  n_nom       = 1200
  n_1					=n_nom/(1-self.S_N)
  gear        =1   
  
  
  freq        =60
  
  dia         =2*self.r_s             # air gap diameter
  self.p			= 3  # number of pole pairs
  self.Lambda_ratio=self.l_s/dia
  #if (2*self.p==2) :
  	#self.Lambda_ratio_LL =0.6
  	#self.Lambda_ratio_UL =1
  #elif (2*self.p==4) :
  #	self.Lambda_ratio_LL =1.2
  #	self.Lambda_ratio_UL =1.8
  #elif (2*self.p==6) :
  #	self.Lambda_ratio_LL =1.6
  #	self.Lambda_ratio_UL =2.2
  #else:
  self.Lambda_ratio_LL=0.5
  self.Lambda_ratio_UL =1.5
  g=(0.1+0.012*(P_gennom)**(1./3))*0.001
  self.r_r					=self.r_s-g             #rotor radius
  
  #tau_s				=self.tau_p/(m*q1)        #slot pitch
  
  self.tau_p=pi*dia/2/self.p
  
  self.N_slots=2*m*self.p*q1  
  #tau_s=self.tau_p/m/q1
  
  tau_s=self.tau_p/m/q1
  #self.N_slots=pi*dia/tau_s
  
  N_slots_pp	=self.N_slots/(m*self.p*2)
  
  
  b_s					=b_s_tau_s*tau_s        #slot width
  b_so				=0.004;
  b_ro				=0.004;
  b_t					=tau_s-b_s              #tooth width
  q2					=4
  self.Q_r		=2*self.p*m*q2
  tau_r				=pi*(dia-2*g)/self.Q_r
  b_r					=b_r_tau_r*tau_r
  b_tr=tau_r-b_r
  
  tau_r_min=pi*(dia-2*(g+self.h_r))/self.Q_r
  self.b_trmin=tau_r_min-b_r_tau_r*tau_r_min
  
  
  
  mu_rs				=3;
  mu_rr				=5;
  W_s					=b_s/mu_rs;
  W_r					=b_r/mu_rr;
  self.Slot_aspect_ratio1=self.h_s/b_s
  self.Slot_aspect_ratio2=self.h_r/b_r
  gamma_s			= (2*W_s/g)**2/(5+2*W_s/g)
  K_Cs				=(tau_s)/(tau_s-g*gamma_s*0.5)  #page 3-13
  gamma_r			= (2*W_r/g)**2/(5+2*W_r/g)
  K_Cr				=(tau_r)/(tau_r-g*gamma_r*0.5)  #page 3-13
  K_C					=K_Cs*K_Cr
  g_eff				=K_C*g
  
  om_m				=gear*2*pi*n_nom/60
  om_e				=self.p*om_m
  self.f			=n_nom*self.p/60
  K_s					=0.3
  n_c					=2    #number of conductors per coil
  a1 					=2    # number of parallel paths
  self.W_1a		=round(2*self.p*N_slots_pp*n_c/a1)
  self.B_g1		=mu_0*3*self.W_1a*self.I_0*sqrt(2)*k_y1*k_q1/(pi*self.p*g_eff*(1+K_s))
  self.B_g		=self.B_g1*K_C
  self.h_ys= self.B_g*self.tau_p/(self.B_symax*pi)
  self.B_rymax=self.B_symax
  self.h_yr= self.h_ys
  d_se				=dia+2*(self.h_ys+self.h_s+h_w)  # stator outer diameter
  self.D_ratio=d_se/dia 
  
  if (2*self.p==2) :
  	self.D_ratio_LL =1.65
  	self.D_ratio_UL =1.69 
  elif (2*self.p==4) :
  	self.D_ratio_LL =1.46
  	self.D_ratio_UL =1.49
  elif (2*self.p==6) :
  	self.D_ratio_LL =1.37
  	self.D_ratio_UL =1.4
  elif (2*self.p==8): 
  	self.D_ratio_LL =1.27
  	self.D_ratio_UL =1.3
  else:
  	self.D_ratio_LL =1.2
  	self.D_ratio_UL =1.24
  
  
  f = n_nom*self.p/60
  if (2*self.r_s>2):
      K_fills=0.65
  else:
  	  K_fills=0.4    
  beta_skew = tau_s/self.r_s
  k_wskew=sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  l_fs=2*(0.015+y_tau_p*self.tau_p/2/cos(40*pi/180))+pi*(self.h_s)
  l_Cus = 2*self.W_1a*(l_fs+self.l_s)/a1               #shortpitch
  A_s = b_s*(self.h_s-h_w)
  A_scalc=b_s*1000*(self.h_s*1000-h_w*1000)
  A_Cus = A_s*q1*self.p*K_fills/self.W_1a
  self.A_Cuscalc = A_scalc*q1*self.p*K_fills/self.W_1a
  
  R_s					=l_Cus*rho_Cu/A_Cus
  om_s				=(1200)*2*pi/60
  P_e					=P_gennom/(1-self.S_N)
  self.E_p		=om_s*self.W_1a*k_wd*self.r_s*self.l_s*self.B_g1
  k_e=0.98-0.005*self.p
  S_GN=(P_gennom-self.S_N*P_gennom)
  T_e         =self.p *(S_GN)/(2*pi*freq*(1-self.S_N))
  I_t					= T_e*2*pi*freq/(3*k_e*self.E_p*self.p)
  #I_srated		=sqrt(self.I_0**2+I_t**2)
  I_srated		=P_gennom/3/self.E_p/cofi
  
  #Field winding
  k_fillr 		= 0.7
  diff				= self.h_r-h_w
  self.A_bar			= b_r*diff
  Beta_skin		= sqrt(pi*mu_0*freq/2/rho_Cu)
  k_rm				= Beta_skin*self.h_r                    #equivalent winding co-efficient, for skin effect correction
  J_b					= 6e+06
  K_i					= 0.864
  I_b					= 2*m*self.W_1a*k_wd*I_srated/self.Q_r
  #self.A_bar	= I_b/J_b
  
  R_rb				=rho_Cu*k_rm*(self.l_s)/(self.A_bar)
  
  I_er				=I_b/(2*sin(pi*self.p/self.Q_r))
  J_er 				= 0.8*J_b
  A_er 				=I_er/J_er
  b						=self.h_r
  a						=A_er/b
  D_er=(self.r_s*2-2*g)-0.003
  l_er=pi*(D_er-b)/self.Q_r
  
  #R_re=rho_Cu*k_rm*D_er/(2*pi*self.p*A_er)
  R_re=rho_Cu*l_er/(2*A_er*(sin(pi*self.p/self.Q_r))**2)
  R_r=(R_rb+R_re)*4*m*(k_wd*self.W_1a)**2/self.Q_r
  
    
  K_fe=0.95
  
  self.B_trmax = self.B_g*tau_r/self.b_trmin 
  
  
  
  self.B_tsmax=tau_s*self.B_g/b_t
  
  
  l_r=self.l_s+(4)*g  # for axial cooling
  l_se =self.l_s+(2/3)*g
 
  L_e=l_se *K_fe   # radial cooling
  
  
  
  L_ssigmas=(2*mu_0*self.l_s*self.W_1a**2/self.p/q1)*((self.h_s-h_w)/(3*b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=(2*mu_0*self.l_s*self.W_1a**2/self.p/q1)*0.34*q1*(l_fs-0.64*self.tau_p*y_tau_p)/self.l_s                   #end winding leakage inductance
  L_ssigmag=2*mu_0*self.l_s*self.W_1a**2/self.p/q1*(5*(g*K_C/b_so)/(5+4*(g*K_C/b_so))) # tooth tip leakage inductance
  L_ssigma=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
  
  L_sm =6*mu_0*self.l_s*self.tau_p*(k_wd*self.W_1a)**2/(pi**2*(self.p)*g_eff*(1+K_s))
  
 
  lambda_ei=2.3*D_er/(4*self.Q_r*self.l_s*(sin(pi*self.p/self.Q_r)**2))*log(4.7*dia/(a+2*b))
  lambda_b=self.h_r/3/b_r+h_w/b_ro
  L_i=pi*dia/self.Q_r
  L_rsl=(mu_0*self.l_s)*((self.h_r-h_w)/(3*b_r)+h_w/b_ro)  #slot leakage inductance
  
  L_rel=mu_0*(self.l_s*lambda_b+2*lambda_ei*L_i)                  #end winding leakage inductance
  L_rtl=(mu_0*self.l_s)*(0.9*tau_r*0.09/g_eff) # tooth tip leakage inductance
  L_rsigma=(L_rsl+L_rtl+L_rel)*4*m*(k_wd*self.W_1a)**2/self.Q_r  # rotor leakage inductance
  
  
  I_r=sqrt(-self.S_N*P_e/m/R_r)
  
  I_sm=self.E_p/(2*pi*freq*L_sm)
  
  I_s=sqrt((I_r**2+I_sm**2))
  
  self.A_1=2*m*self.W_1a*I_s/pi/(2*self.r_s) 
  
  V_Cuss=m*l_Cus*A_Cus
  V_Cusr=(self.Q_r*self.l_s*self.A_bar+pi*(D_er*A_er-A_er*b))
  V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-2*m*q1*self.p*b_s*self.h_s*self.l_s)
  V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)
  r_r=self.r_s-g
  V_Fert=pi*self.l_s*(r_r**2-(r_r-self.h_r)**2)-2*m*q2*self.p*b_r*self.h_r*self.l_s
  V_Fery=self.l_s*pi*((r_r-self.h_r)**2-(r_r-self.h_r-self.h_yr)**2)
  M_Cus=(V_Cuss+V_Cusr)*8900
  M_Fest=V_Fest*7700
  M_Fesy=V_Fesy*7700
  M_Fert=V_Fert*7700
  M_Fery=V_Fery*7700
  M_Fe=M_Fest+M_Fesy+M_Fert+M_Fery
  self.M_gen=(M_Cus+M_Fe)
  L_tot=self.l_s
  self.M_Str=0.0001*self.M_gen**2+0.8841*self.M_gen-132.5
  K_gen=M_Cus*K_Cu+(M_Fe)*K_Fe #%M_pm*K_pm;
  C_str=cstr*self.M_Str
  self.TM=self.M_gen+self.M_Str
  self.M_actual=self.M_gen+self.M_Str
 
  self.TC=K_gen+C_str
  # losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P_Cuss=m*I_s**2*R_s
  P_Cusr=m*I_r**2*R_r
  P_Cusnom=P_Cuss+P_Cusr
  
  P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Hyd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Hyyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0h*abs(self.S_N)*om_e/(2*pi*50))
  P_Ftyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0e*(abs(self.S_N)*om_e/(2*pi*50))**2)
  P_Hydr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0h*abs(self.S_N)*om_e/(2*pi*50))
  P_Ftdr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0e*(abs(self.S_N)*om_e/(2*pi*50))**2)
  P_add=0.5*P_gennom/100  
  P_Fesnom=P_Hyys+P_Ftys+P_Hyd+P_Ftd+P_Hyyr+P_Ftyr+P_Hydr+P_Ftdr
  self.TL=P_Cusnom+P_Fesnom+P_add;
  self.gen_eff=(P_e-self.TL)*100/P_e
  self.J_s=I_s/self.A_Cuscalc
  
  self.J_r=I_r/(self.A_bar)/1e6
  #self.A_1=2*m*self.W_1a*I_srated/pi/(2*self.r_s)
  
  
  self.TC1=T_e/(2*pi*sigma)
  self.TC2=self.r_s**2*self.l_s
  
  
  sigma_yield=300e6
  C_1=0.823
  self.r_rmax=(sigma_yield/(C_1*7850*(2*pi*n_nom/60/self.S_N)**2))**0.5
  
  print self.B_symax,self.TM
  
   