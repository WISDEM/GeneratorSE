"""DDSG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component,Assembly
from openmdao.lib.datatypes.api import Float
from scipy.interpolate import interp1d
from openmdao.main.datatypes.api import Float, Bool, Int, Str, Array, Enum
from openmdao.lib.drivers.api import COBYLAdriver, CONMINdriver,NEWSUMTdriver,SLSQPdriver,Genetic
import numpy as np
import pandas


class SCIG(Component):
 """ Evaluates the total cost """
 r_s = Float(0.0,iotype='in', desc='airgap radius r_s')
 l_s = Float(0.0,iotype='in', desc='Stator core length l_s')
 h_s = Float(0.0,iotype='in', desc='Yoke height h_s')
 h_r = Float(0.0,iotype='in', desc='Rotor slot height')
 I_0 = Float(0.0,iotype='in', desc='No load current- Excitation')
 machine_rating=Float(0.0,iotype='in', desc='Machine rating')
 n_nom=Float(0.0,iotype='in', desc='Rated speed')
 B_symax = Float(0.0,iotype='in', desc='Peak Yoke flux density B_ymax')
 
 # Costs and material properties
 C_Cu=Float( iotype='in', desc='Specific cost of copper')
 C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
 C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
 
 rho_Fe=Float(iotype='in', desc='Steel density kg/m^3')
 rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
 
 highSpeedSide_cm =Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc=' high speed sidde COM [x, y, z]')
 highSpeedSide_length =Float(0.0,iotype='in', desc='high speed side length')
 Gearbox_efficiency=Float(0.0,iotype='in', desc='Gearbox efficiency')
 tau_p =Float(0.0,iotype='out', desc='Pole pitch self.tau_p')
 S_N = Float(0.0,iotype='out', desc='Slip')
 h_ys=Float(0.0,iotype='out', desc='stator back iron thickness')
 h_yr=Float(0.0,iotype='out', desc='rotor back iron thickness')
 B_g = Float(0.0,iotype='out', desc='Peak air gap flux density B_g')
 b_s=Float(0.0,iotype='out', desc='stator slot width')
 b_r=Float(0.0,iotype='out', desc='rotor slot width')
 b_t=Float(0.0,iotype='out', desc='stator tooth width')
 b_trmin=Float(0.0,iotype='out', desc='minimum rotor tooth width')
 b_tr=Float(0.0,iotype='out', desc='rotor tooth width')
 P_gennom=Float(0.0,iotype='out', desc='Generator input')
 B_rymax = Float(0.0,iotype='out', desc='Peak Rotor Yoke flux density B_ymax')
 B_tsmax = Float(0.0,iotype='out', desc='Peak stator tooth flux density B_ymax')
 B_trmax = Float(0.0,iotype='out', desc='Peak rotor tooth flux density B_ymax')
 N_s = Float(0.0,iotype='out', desc='Stator winding turns')
 f=Float(0.0,iotype='out', desc='output frequency')
 Q_r = Float(0.0,iotype='out', desc='Rotor slots')
 p=Float(0.0,iotype='out', desc='pole pairs')
 E_p=Float(0.0,iotype='out', desc='Stator phase voltage')
 I_s=Float(0.0,iotype='out', desc='Generator output phase current')
 Costs= Float(0.0,iotype='out', desc='Total cost')
 Losses=Float(0.0,iotype='out', desc='Total Loss')
 M_gen=Float(0.0,iotype='out', desc='Active mass')
 M_Str=Float(0.0,iotype='out', desc='Structural mass')
 gen_eff=Float(0.0,iotype='out', desc='Generator efficiency')
 q1=Float(0.0,iotype='out', desc='Stator slots per pole per phase')
 J_r=Float(0.0,iotype='out', desc='Rotor current density')
 TC1=Float(0.0,iotype='out', desc='Torque constraint')
 TC2=Float(0.0,iotype='out', desc='Torque constraint')
 A_1=Float(0.0,iotype='out', desc='Specific current loading')
 J_s=Float(0.0,iotype='out', desc='Stator current density')
 J_r=Float(0.0,iotype='out', desc='Rotor current density')
 K_rad=Float(0.0,iotype='out', desc='Stack length ratio')
 D_ratio=Float(0.0,iotype='out', desc='Stator diameter ratio')
 A_Cuscalc=Float(0.0,iotype='out', desc='Stator conductor cross-section')
 A_Curcalc=Float(0.0,iotype='out', desc='Rotor conductor cross-section')
 S=Float(0.0,iotype='out', desc='Stator slots')
 Slot_aspect_ratio1=Float(0.0,iotype='out', desc='Slot apsect ratio-stator')
 Slot_aspect_ratio2=Float(0.0,iotype='out', desc='Slot apsect ratio-rotor')
 D_ratio_UL=Float(0.0,iotype='out', desc='Dia ratio upper limit')
 D_ratio_LL=Float(0.0,iotype='out', desc='Dia ratio Lower limit')
 K_rad_UL=Float(0.0,iotype='out', desc='Aspect ratio upper limit')
 K_rad_LL=Float(0.0,iotype='out', desc='Aspect ratio Lower limit')
 r_r=Float(0.0,iotype='out', desc='rotor radius')
 Overall_eff=Float(0.0,iotype='out', desc='Overall efficiency')
 R_s=Float(0.0,iotype='out', desc='Stator resistance')
 L_s=Float(0.0,iotype='out', desc='Stator inductance')
 L_sm=Float(0.0,iotype='out', desc='mutual inductance')
 R_R=Float(0.0,iotype='out', desc='Rotor resistance')
 Copper=Float(0.0,iotype='out', desc='Copper mass')
 Iron=Float(0.0,iotype='out', desc='Iron mass')
 
 cm  =Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='COM [x,y,z]')
 Mass=Float(0.01, iotype='out', desc='Generator mass')
 I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  h_r=self.h_r
  b_s=self.b_s
  b_t=self.b_t
  b_r=self.b_r
  b_tr=self.b_tr
  b_trmin=self.b_trmin
  tau_p =self.tau_p
  B_g = self.B_g
  h_ys=self.h_ys
  h_yr=self.h_yr
  B_symax = self.B_symax
  B_rymax=self.B_rymax
  B_tsmax=self.B_tsmax
  B_trmax=self.B_trmax
  S_N=self.S_N
  Costs=self.Costs
  Losses=self.Losses
  E_p=self.E_p
  N_s=self.N_s
  p=self.p
  J_s=self.J_s
  gen_eff =self.gen_eff
  Mass=self.Mass
  M_gen=self.M_gen
  M_Str=self.M_Str
  Q_r=self.Q_r
  I_0=self.I_0
  TC1=self.TC1
  TC2=self.TC2
  S=self.S
  n_nom=self.n_nom
  Q_r=self.Q_r
  A_1=self.A_1
  q1=self.q1
  J_r=self.J_r
  K_rad=self.K_rad
  D_ratio=self.D_ratio
  D_ratio_UL=self.D_ratio_UL
  D_ratio_LL=self.D_ratio_LL
  f=self.f
  A_Curcalc=self.A_Curcalc
  A_Cuscalc=self.A_Cuscalc
  Slot_aspect_ratio1=self.Slot_aspect_ratio1
  Slot_aspect_ratio2=self.Slot_aspect_ratio2
  K_rad_LL= self.K_rad_LL 
  K_rad_UL= self.K_rad_UL
  
  r_r=self.r_r
  highSpeedSide_cm=self.highSpeedSide_cm
  highSpeedSide_length=self.highSpeedSide_length
  machine_rating=self.machine_rating
  Gearbox_efficiency=self.Gearbox_efficiency
  Overall_eff=self.Overall_eff
  cm = self.cm
  I=self.I
  I_s=self.I_s
  R_s=self.R_s
  L_sm=self.L_sm
  R_R=self.R_R
  L_s=self.L_s
  Copper=self.Copper
  Iron=self.Iron
  
  C_Cu=self.C_Cu
  C_Fe=self.C_Fe
  C_Fes=self.C_Fes
  
  rho_Fe= self.rho_Fe
  rho_Copper=self.rho_Copper
 

  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign
  
  g1     			=9.81                	# m/s^2 acceleration due to gravity
  sigma  			=21.5e3                # shear stress [chapter Design of Rotating Electrical Machines
  mu_0   			=pi*4e-7        				# permeability of free space
  cofi   			=0.9                 	# power factor
  h_w    			= 0.005
  m      			=3                     # Number of phases
  
  self.q1     =6										 # Number of slots per pole per phase
  b_s_tau_s   =0.45										#
  b_r_tau_r   =0.45										#
  self.S_N    =-0.002									#
  y_tau_p     =12./15										#
  k_y1        =sin(pi*0.5*y_tau_p)						#
  k_q1        =sin(pi/6)/self.q1/sin(pi/6/self.q1)# 
  k_wd        =k_y1*k_q1							#
  P_Fe0h      =4			               	#specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e      =1			               	#specific hysteresis losses W/kg @ 1.5 T @50 Hz
  rho_Cu      =1.8*10**(-8)*1.4
  
  n_1					=self.n_nom/(1-self.S_N)
  gear        =1   
  
  
  freq        =60
  
  dia         =2*self.r_s             # air gap diameter
  self.p			= 3  # number of pole pairs
  self.K_rad=self.l_s/dia

  self.K_rad_LL=0.5   # lower limit
  self.K_rad_UL =1.5   # upper limit
  g=(0.1+0.012*(self.machine_rating)**(1./3))*1e-3
  
  self.r_r					=self.r_s-g             #rotor radius
  
 
  self.tau_p=pi*dia/2/self.p
  
  self.S=2*m*self.p*self.q1  

  
  tau_s=self.tau_p/m/self.q1

  
  N_slots_pp	=self.S/(m*self.p*2)
  
  
  self.b_s					=b_s_tau_s*tau_s        #slot width
  b_so				=0.004;
  b_ro				=0.004;
  self.b_t				=tau_s-self.b_s              #tooth width
  q2					=4
  self.Q_r		=2*self.p*m*q2
  tau_r				=pi*(dia-2*g)/self.Q_r
  self.b_r					=b_r_tau_r*tau_r
  self.b_tr=tau_r-self.b_r
  tau_r_min=pi*(dia-2*(g+self.h_r))/self.Q_r
  self.b_trmin=tau_r_min-b_r_tau_r*tau_r_min
  
  
  
  mu_rs				=0.005;
  mu_rr				=0.005;
  W_s					=(self.b_s/mu_rs)*1e-3;
  W_r					=(self.b_r/mu_rr)*1e-3;
  
  self.Slot_aspect_ratio1=self.h_s/self.b_s
  self.Slot_aspect_ratio2=self.h_r/self.b_r
  gamma_s			= (2*W_s/g)**2/(5+2*W_s/g)
  K_Cs				=(tau_s)/(tau_s-g*gamma_s*0.5)  #page 3-13
  gamma_r			= (2*W_r/g)**2/(5+2*W_r/g)
  K_Cr				=(tau_r)/(tau_r-g*gamma_r*0.5)  #page 3-13
  K_C					=K_Cs*K_Cr
  g_eff				=K_C*g
 
  om_m				=gear*2*pi*self.n_nom/60
  om_e				=self.p*om_m
  self.f			=self.n_nom*self.p/60
  K_s					=0.3
  n_c					=2    #number of conductors per coil
  a1 					=2    # number of parallel paths
  self.N_s		=round(2*self.p*N_slots_pp*n_c/a1)
  self.B_g1		=mu_0*3*self.N_s*self.I_0*sqrt(2)*k_y1*k_q1/(pi*self.p*g_eff*(1+K_s))
  
  
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
  
  
  
  if (2*self.r_s>2):
      K_fills=0.65
  else:
  	  K_fills=0.4    
  beta_skew = tau_s/self.r_s
  k_wskew=sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  l_fs=2*(0.015+y_tau_p*self.tau_p/2/cos(40*pi/180))+pi*(self.h_s)
  l_Cus = 2*self.N_s*(l_fs+self.l_s)/a1               #shortpitch
  A_s = self.b_s*(self.h_s-h_w)
  A_scalc=self.b_s*1000*(self.h_s*1000-h_w*1000)
  A_Cus = A_s*self.q1*self.p*K_fills/self.N_s
  self.A_Cuscalc = A_scalc*self.q1*self.p*K_fills/self.N_s
  
  self.R_s					=l_Cus*rho_Cu/A_Cus
  om_s				=(self.n_nom)*2*pi/60
  P_e					=self.machine_rating/(1-self.S_N)
  self.E_p		=om_s*self.N_s*k_wd*self.r_s*self.l_s*self.B_g1
  
  k_e=0.98-0.005*self.p
  S_GN=(self.machine_rating-self.S_N*self.machine_rating)
  T_e         =self.p *(S_GN)/(2*pi*freq*(1-self.S_N))
  I_t					= T_e*2*pi*freq/(3*k_e*self.E_p*self.p)
  #I_srated		=sqrt(self.I_0**2+I_t**2)
  I_srated		=self.machine_rating/3/self.E_p/cofi
  
  #Field winding
  k_fillr 		= 0.7
  diff				= self.h_r-h_w
  
  self.A_bar			= self.b_r*diff
  Beta_skin		= sqrt(pi*mu_0*freq/2/rho_Cu)
  k_rm				= Beta_skin*self.h_r                    #equivalent winding co-efficient, for skin effect correction
  J_b					= 6e+06
  K_i					= 0.864
  I_b					= 2*m*self.N_s*k_wd*I_srated/self.Q_r

  
  R_rb				=rho_Cu*k_rm*(self.l_s)/(self.A_bar)
  
  I_er				=I_b/(2*sin(pi*self.p/self.Q_r))
  J_er 				= 0.8*J_b
  A_er 				=I_er/J_er
  b						=self.h_r
  a						=A_er/b
  D_er=(self.r_s*2-2*g)-0.003
  l_er=pi*(D_er-b)/self.Q_r
  

  R_re=rho_Cu*l_er/(2*A_er*(sin(pi*self.p/self.Q_r))**2)
  self.R_R=(R_rb+R_re)*4*m*(k_wd*self.N_s)**2/self.Q_r
  
    
  K_fe=0.95
  
  self.B_trmax = self.B_g*tau_r/self.b_trmin 
  
  
  
  self.B_tsmax=tau_s*self.B_g/self.b_t
  
  
  l_r=self.l_s+(4)*g  # for axial cooling
  l_se =self.l_s+(2/3)*g
 
  L_e=l_se *K_fe   # radial cooling
  
  
  
  L_ssigmas=(2*mu_0*self.l_s*self.N_s**2/self.p/self.q1)*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=(2*mu_0*self.l_s*self.N_s**2/self.p/self.q1)*0.34*self.q1*(l_fs-0.64*self.tau_p*y_tau_p)/self.l_s                   #end winding leakage inductance
  L_ssigmag=2*mu_0*self.l_s*self.N_s**2/self.p/self.q1*(5*(g*K_C/b_so)/(5+4*(g*K_C/b_so))) # tooth tip leakage inductance
  self.L_s=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
  
  self.L_sm =6*mu_0*self.l_s*self.tau_p*(k_wd*self.N_s)**2/(pi**2*(self.p)*g_eff*(1+K_s))
  
 
  lambda_ei=2.3*D_er/(4*self.Q_r*self.l_s*(sin(pi*self.p/self.Q_r)**2))*log(4.7*dia/(a+2*b))
  lambda_b=self.h_r/3/self.b_r+h_w/b_ro
  L_i=pi*dia/self.Q_r
  L_rsl=(mu_0*self.l_s)*((self.h_r-h_w)/(3*self.b_r)+h_w/b_ro)  #slot leakage inductance
  
  L_rel=mu_0*(self.l_s*lambda_b+2*lambda_ei*L_i)                  #end winding leakage inductance
  L_rtl=(mu_0*self.l_s)*(0.9*tau_r*0.09/g_eff) # tooth tip leakage inductance
  L_rsigma=(L_rsl+L_rtl+L_rel)*4*m*(k_wd*self.N_s)**2/self.Q_r  # rotor leakage inductance
  
  I_r=sqrt(-self.S_N*P_e/m/self.R_R)
 
  
  I_sm=self.E_p/(2*pi*freq*self.L_sm)
  
  self.I_s=sqrt((I_r**2+I_sm**2))
  
  self.A_1=2*m*self.N_s*self.I_s/pi/(2*self.r_s) 
  
  V_Cuss=m*l_Cus*A_Cus
  V_Cusr=(self.Q_r*self.l_s*self.A_bar+pi*(D_er*A_er-A_er*b))
  V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-2*m*self.q1*self.p*self.b_s*self.h_s*self.l_s)
  V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)
  r_r=self.r_s-g
  V_Fert=pi*self.l_s*(r_r**2-(r_r-self.h_r)**2)-2*m*q2*self.p*self.b_r*self.h_r*self.l_s
  V_Fery=self.l_s*pi*((r_r-self.h_r)**2-(r_r-self.h_r-self.h_yr)**2)
  self.Copper=(V_Cuss+V_Cusr)*self.rho_Copper
  M_Fest=V_Fest*self.rho_Fe
  M_Fesy=V_Fesy*self.rho_Fe
  M_Fert=V_Fert*self.rho_Fe
  M_Fery=V_Fery*self.rho_Fe
  self.Iron=M_Fest+M_Fesy+M_Fert+M_Fery
  self.M_gen=(self.Copper+self.Iron)
  L_tot=self.l_s
  self.M_Str=0.0001*self.M_gen**2+0.8841*self.M_gen-132.5
  K_gen=self.Copper*self.C_Cu+(self.Iron)*self.C_Fe #%M_pm*K_pm;
  Cost_str=self.C_Fes*self.M_Str
  
  self.Mass=self.M_gen+self.M_Str
  
  self.Costs=K_gen+Cost_str
  K_R=1.2 # skin effect correction coefficient
  # losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P_Cuss=m*self.I_s**2*self.R_s*K_R
  P_Cusr=m*I_r**2*self.R_R
  P_Cusnom=P_Cuss+P_Cusr
  
  P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
  P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)
  P_Hyd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
  P_Ftd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)
  P_Hyyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0h*abs(self.S_N)*om_e/(2*pi*60))
  P_Ftyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0e*(abs(self.S_N)*om_e/(2*pi*60))**2)
  P_Hydr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0h*abs(self.S_N)*om_e/(2*pi*60))
  P_Ftdr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0e*(abs(self.S_N)*om_e/(2*pi*60))**2)
  P_add=0.5*self.machine_rating/100  
  P_Fesnom=P_Hyys+P_Ftys+P_Hyd+P_Ftd+P_Hyyr+P_Ftyr+P_Hydr+P_Ftdr
  self.Losses=P_Cusnom+P_Fesnom+P_add;
  self.gen_eff=(P_e-self.Losses)*100/P_e
  self.Overall_eff=self.gen_eff*self.Gearbox_efficiency
  self.J_s=self.I_s/self.A_Cuscalc
  
  self.J_r=I_r/(self.A_bar)/1e6

  
  
  self.TC1=T_e/(2*pi*sigma)
  self.TC2=self.r_s**2*self.l_s
  
  

  
  r_out=d_se*0.5
  self.I[0]   = (0.5*self.Mass*r_out**2)
  self.I[1]   = (0.25*self.Mass*r_out**2+(1/12)*self.Mass*self.l_s**2)
  self.I[2]   = self.I[1]
  self.cm[0]  = self.highSpeedSide_cm[0] + self.highSpeedSide_length/2. + self.l_s/2.
  self.cm[1]  = self.highSpeedSide_cm[1]
  self.cm[2]  = self.highSpeedSide_cm[2]
  

  
class Drive_SCIG(Assembly):
	Eta_target=Float(0.0,iotype='in', desc='Target drivetrain efficiency')
	Gearbox_efficiency=Float(0.0,iotype='in', desc='Gearbox efficiency')
	highSpeedSide_cm = Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc='High speed side CM')
	highSpeedSide_length = Float(0.0,iotype='in', desc='High speed side length')
	Objective_function=Str(iotype='in')
	Optimiser=Str(desc = 'Optimiser', iotype = 'in')
	L=Float(0.0,iotype='out')
	Mass=Float(0.0,iotype='out')
	Efficiency=Float(0.0,iotype='out')
	r_s=Float(0.0,iotype='out', desc='Optimised radius')
	l_s=Float(0.0,iotype='out', desc='Optimised generator length')
	I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
	cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' Center of mass [x, y,z]')
	SCIG_r_s = Float(0.0,iotype='in', desc='airgap radius r_s')
	SCIG_l_s = Float(0.0,iotype='in', desc='Stator core length l_s')
	SCIG_h_s = Float(0.0,iotype='in', desc='Stator slot height h_s')
	SCIG_h_r = Float(0.0,iotype='in', desc='Rotor slot height ')
	SCIG_S_N =Float(0.0,iotype='in', desc='Slip ')
	SCIG_B_symax = Float(0.0,iotype='in', desc='Peak Yoke flux density B_ymax')
	SCIG_I_0= Float(0.0,iotype='in', desc='Rotor current at no-load')
	SCIG_P_rated=Float(0.0,iotype='in',desc='Rated power')
	SCIG_N_rated=Float(0.0,iotype='in',desc='Rated speed')
	
	C_Cu=Float( iotype='in', desc='Specific cost of copper')
	C_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	C_Fes=Float(iotype='in', desc='Specific cost of structural steel')
	
	rho_Fe=Float(iotype='in', desc='Steel density kg/m^3')
	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
	
	def __init__(self,Optimiser='',Objective_function=''):
		
				super(Drive_SCIG,self).__init__()
				""" Creates a new Assembly containing SCIG and an optimizer"""
				self.add('SCIG',SCIG())
				self.connect('SCIG_r_s','SCIG.r_s')
				self.connect('SCIG_l_s','SCIG.l_s')
				self.connect('SCIG_h_s','SCIG.h_s')
				self.connect('SCIG_h_r','SCIG.h_r')
				self.connect('SCIG_B_symax','SCIG.B_symax')
				self.connect('SCIG_I_0','SCIG.I_0')
				self.connect('SCIG_P_rated','SCIG.machine_rating')
				self.connect('SCIG_N_rated','SCIG.n_nom')
				self.connect('Gearbox_efficiency','SCIG.Gearbox_efficiency')
				self.connect('highSpeedSide_cm','SCIG.highSpeedSide_cm')
				self.connect('highSpeedSide_length','SCIG.highSpeedSide_length')
				self.connect('SCIG.Mass','Mass')
				self.connect('SCIG.Overall_eff','Efficiency')
				self.connect('SCIG.r_s','r_s')
				self.connect('SCIG.l_s','l_s')
				self.connect('SCIG.I','I')
				self.connect('SCIG.cm','cm')
				
				self.connect('C_Fe','SCIG.C_Fe')
				self.connect('C_Fes','SCIG.C_Fes')
				self.connect('C_Cu','SCIG.C_Cu')
				self.connect('rho_Fe','SCIG.rho_Fe')
				self.connect('rho_Copper','SCIG.rho_Copper')
				
				
				
				self.Optimiser=Optimiser
				self.Objective_function=Objective_function
				Opt1=globals()[self.Optimiser]
				self.add('driver',Opt1())
				Obj1='SCIG'+'.'+self.Objective_function
				self.driver.add_objective(Obj1)
				self.driver.design_vars=['SCIG_r_s','SCIG_l_s','SCIG_h_s','SCIG_h_r','SCIG_S_N','SCIG_B_symax','SCIG_I_0']
				self.driver.add_parameter('SCIG_r_s', low=0.2, high=1)
				self.driver.add_parameter('SCIG_l_s', low=0.4, high=2)
				self.driver.add_parameter('SCIG_h_s', low=0.04, high=0.1)
				self.driver.add_parameter('SCIG_h_r', low=0.04, high=0.1)
				self.driver.add_parameter('SCIG_B_symax', low=1, high=2)
				self.driver.add_parameter('SCIG_I_0', low=5, high=200)
				self.driver.iprint=1
				self.driver.add_constraint('SCIG.Overall_eff>=Eta_target')		  						#constraint 1
				self.driver.add_constraint('SCIG.E_p>500.0')															  #constraint 2
				self.driver.add_constraint('SCIG.E_p<5000.0')																#constraint 3
				self.driver.add_constraint('SCIG.TC1<SCIG.TC2')															#constraint 4
				self.driver.add_constraint('SCIG.B_g>=0.7')																	#constraint 5
				self.driver.add_constraint('SCIG.B_g<=1.2')																	#constraint 6
				self.driver.add_constraint('SCIG.B_rymax<2.')																#constraint 7
				self.driver.add_constraint('SCIG.B_trmax<2.')																#constraint 8
				self.driver.add_constraint('SCIG.B_tsmax<2.') 															#constraint 9
				self.driver.add_constraint('SCIG.A_1<60000')  															#constraint 10
				self.driver.add_constraint('SCIG.J_s<=6')											        			#constraint 11
				self.driver.add_constraint('SCIG.J_r<=6')  																	#constraint 12
				self.driver.add_constraint('SCIG.K_rad>=SCIG.K_rad_LL')  										#constraint 13 #boldea Chapter 3
				self.driver.add_constraint('SCIG.K_rad<=SCIG.K_rad_UL')											#constraint 14
				self.driver.add_constraint('SCIG.D_ratio>=SCIG.D_ratio_LL')  							  #constraint 15 #boldea Chapter 3
				self.driver.add_constraint('SCIG.D_ratio<=SCIG.D_ratio_UL')  								#constraint 16
				self.driver.add_constraint('SCIG.Slot_aspect_ratio1>=4')										#constraint 17
				self.driver.add_constraint('SCIG.Slot_aspect_ratio1<=10')										#constraint 18
				     	       	 							
def optim_SCIG():
	opt_problem = Drive_SCIG('CONMINdriver','Costs')
	 #Initial design variables for a SCIG designed for a 5MW turbine
	opt_problem.SCIG_r_s= 0.55 #meter
	opt_problem.SCIG_l_s= 1.3 #meter
	opt_problem.SCIG_h_s = 0.09 #meter
	opt_problem.SCIG_h_r = 0.050 #meter
	opt_problem.SCIG_I_0 = 140   #Ampere
	opt_problem.SCIG_B_symax = 1.4  #Tesla
	opt_problem.Eta_target=93
	opt_problem.SCIG_P_rated=5e6
	opt_problem.Gearbox_efficiency=0.955
	opt_problem.SCIG_N_rated=1200
	
	# Specific costs
	opt_problem.C_Cu   =4.786                  # Unit cost of Copper $/kg
	opt_problem.C_Fe	= 0.556                    # Unit cost of Iron $/kg
	opt_problem.C_Fes =0.50139                   # specific cost of structure
	
	#Material properties
	opt_problem.rho_Fe = 7700                 #Steel density
	opt_problem.rho_Copper =8900                  # Kg/m3 copper density
	
	
	opt_problem.run()
	
	
	
	# Initial design variables for a SCIG designed for a 10MW turbine
#	opt_problem.SCIG_P_rated=10e6
#	opt_problem.SCIG_r_s= 0.6 #meter
#	opt_problem.SCIG_l_s= 1.8 #meter
#	opt_problem.SCIG_h_s = 0.1 #meter
#	opt_problem.SCIG_h_r = 0.050 #meterf
#	opt_problem.SCIG_I_0 = 160 #Ampere
#	opt_problem.SCIG_B_symax = 1.3  #Tesla
#	opt_problem.Eta_target=93
#	opt_problem.Gearbox_efficiency=0.955
#	opt_problem.SCIG_N_rated=1200
#	opt_problem.run()
	raw_data = {'Parameters': ['Rating','Objective function',"Air gap diameter", "Stator length","Lambda ratio","Diameter ratio", "Pole pitch(tau_p)", " Number of Stator Slots","Stator slot height(h_s)","Stator slot width(b_s)","Stator Slot aspect ratio", "Stator tooth width(b_t)", "Stator yoke height(h_ys)","Rotor slots", "Rotor slot height(h_r)", "Rotor slot width(b_r)","Rotor tooth width(b_tr)","Rotor yoke height(h_yr)","Rotor Slot_aspect_ratio","Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak Rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading","Stator resistance", "Excited magnetic inductance","Magnetization current","Conductor cross-section"," Rotor Current density","Rotor resitance", "Generator Efficiency","Overall drivetrain Efficiency","Copper mass","Iron Mass", "Structural mass","Total Mass","Total Material Cost"],
		           'Values': [opt_problem.SCIG.machine_rating/1e6,opt_problem.Objective_function,2*opt_problem.SCIG.r_s,opt_problem.SCIG.l_s,opt_problem.SCIG.K_rad,opt_problem.SCIG.D_ratio,opt_problem.SCIG.tau_p*1000,opt_problem.SCIG.S,opt_problem.SCIG.h_s*1000,opt_problem.SCIG.b_s*1000,opt_problem.SCIG.Slot_aspect_ratio1,opt_problem.SCIG.b_t*1000,opt_problem.SCIG.h_ys*1000,opt_problem.SCIG.Q_r,opt_problem.SCIG.h_r*1000,opt_problem.SCIG.b_r*1000,opt_problem.SCIG.b_tr*1000,opt_problem.SCIG.h_yr*1000,opt_problem.SCIG.Slot_aspect_ratio2,opt_problem.SCIG.B_g,opt_problem.SCIG.B_g1,opt_problem.SCIG.B_symax,opt_problem.SCIG.B_rymax,opt_problem.SCIG.B_tsmax,opt_problem.SCIG.B_trmax,opt_problem.SCIG.p,opt_problem.SCIG.f,opt_problem.SCIG.E_p,opt_problem.SCIG.I_s,opt_problem.SCIG.S_N,opt_problem.SCIG.N_s,opt_problem.SCIG.A_Cuscalc,opt_problem.SCIG.J_s,opt_problem.SCIG.A_1/1000,opt_problem.SCIG.R_s,opt_problem.SCIG.L_sm,opt_problem.SCIG.I_0,opt_problem.SCIG.A_bar*1e6,opt_problem.SCIG.J_r,opt_problem.SCIG.R_R,opt_problem.SCIG.gen_eff,opt_problem.SCIG.Overall_eff,opt_problem.SCIG.Copper/1000,opt_problem.SCIG.Iron/1000,opt_problem.SCIG.M_Str/1000,opt_problem.SCIG.Mass/1000,opt_problem.SCIG.Costs/1000],
		           	'Limit': ['','','','',"("+str(opt_problem.SCIG.K_rad_LL)+"-"+str(opt_problem.SCIG.K_rad_UL)+")","("+str(opt_problem.SCIG.D_ratio_LL)+"-"+str(opt_problem.SCIG.D_ratio_UL)+")",'','','','','(4-10)','','','','','','','','(4-10)','(0.7-1.2)','','2','2','2','2','','(10-60)','(500-5000)','','(-30% to -0.2%)','','','(3-6)','<60','','','','','','','',opt_problem.Eta_target,'','','','',''],
		           	'Units':['MW','','m','m','-','-','mm','-','mm','mm','-','mm','mm','-','mm','mm','mm','mm','','T','T','T','T','T','T','-','Hz','V','A','%','turns','mm^2','A/mm^2','kA/m','ohms','p.u','A','mm^2','A/mm^2','ohms','%','%','Tons','Tons','Tons','Tons','$1000']} 
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print(df)
	df.to_excel('SCIG_'+str(opt_problem.SCIG_P_rated/1e6)+'_MW.xlsx')


		
if __name__=="__main__":
		optim_SCIG()