"""DDSG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float
from scipy.interpolate import interp1d
import numpy as np
from numpy import array

class DFIG_Optim5MW(Component):
 """ Evaluates the total cost """
 r_s = Float(0.5, iotype='in', desc='airgap radius r_s')
 l_s = Float(0.8, iotype='in', desc='Stator core length l_s')
 h_s = Float(0.1, iotype='in', desc='Yoke height h_s')
 h_r = Float(0.07, iotype='in', desc='Stator slot height ')
 S_N =Float(-0.3, iotype='in', desc='Stator slot height ')
 B_symax = Float(1.35, iotype='in', desc='Peak Yoke flux density B_ymax')
 I_f= Float(27, iotype='in', desc='Rotor current at no-load')
 tau_p=Float(0.460, iotype='out', desc='Pole pitch')
 tau_p_act=Float(0.01, iotype='out', desc='actual Pole pitch')
 p=Float(0, iotype='out', desc='Pole pairs')
 q1   =Float(5, iotype='out', desc='Slots per pole per phase')
 h_ys=Float(0.06, iotype='out', desc=' stator yoke height')
 B_g = Float(0.7, iotype='out', desc='Peak air gap flux density B_g')
 B_g1 = Float(0.7, iotype='out', desc='Peak air gap flux density B_g')
 B_rymax = Float(0.7, iotype='out', desc='Peak air gap flux density B_g')
 
 B_tsmax  =Float(0.9, iotype='out', desc='Peak Yoke flux density B_tmax')
 B_trmax = Float(0.9, iotype='out', desc='Peak Yoke flux density B_ymax')
 N_slots = Float(0.9, iotype='out', desc='Stator slots')
 Q_r = Float(0.9, iotype='out', desc='Rotor slots')
 h_yr=Float(0.9, iotype='out', desc=' rotor yoke height')
 W_1a= Float(0, iotype='out', desc='Stator turns')
 W_2= Float(80, iotype='out', desc='Rotor turns')
 
 M_actual=Float(0.0, iotype='out', desc='Actual mass')
 p=Float(0.0, iotype='out', desc='No of pole pairs')
 f=Float(0.0, iotype='out', desc='Output frequency')
 E_p=Float(0.0, iotype='out', desc='Stator phase voltage')
 I_s=Float(0.0, iotype='out', desc='Generator output phase current')
 b_trmin=Float(0.0, iotype='out', desc='minimum tooth width')
 b_tr=Float(0.0, iotype='out', desc='rotor tooth width')
 #R_R=Float(0.0, iotype='out', desc='Rotor resistance')
 #L_r=Float(0.0, iotype='out', desc='Rotor impedance')
 #L_s=Float(0.0, iotype='out', desc='Stator synchronising inductance')
 J_s=Float(0.0, iotype='out', desc='Current density')
 TC= Float(0.0, iotype='out', desc='Total cost')
 TL=Float(0.0, iotype='out', desc='Total Loss')
 K_rad=Float(0.0, iotype='out', desc='K_rad')
 TM=Float(0.01, iotype='out', desc='Total loss')
 gen_eff=Float(0.01, iotype='out', desc='Generator efficiency')
 M_actual=Float(0.01, iotype='out', desc='Total Mass')
 Structure=Float(0.01, iotype='out', desc='Structural mass')
 M_Fe=Float(0.01, iotype='out', desc='Structural mass')
 M_Cus=Float(0.01, iotype='out', desc='Structural mass')
 Active=Float(0.01, iotype='out', desc='Active Mass')
 TC1=Float(0.01, iotype='out', desc='Torque constraint')
 TC2=Float(0.01, iotype='out', desc='Torque constraint')
 A_1=Float(0.01, iotype='out', desc='Specific current loading')
 J_s=Float(0.01, iotype='out', desc='Stator current density')
 J_r=Float(0.01, iotype='out', desc='Rotor current density')
 lambda_ratio=Float(0.01, iotype='out', desc='Stack length ratio')
 D_ratio=Float(0.01, iotype='out', desc='Stator diameter ratio')
 A_Cuscalc=Float(0.01, iotype='out', desc='Stator conductor cross-section')
 A_Curcalc=Float(0.01, iotype='out', desc='Rotor conductor cross-section')
 Current_ratio=Float(0.01, iotype='out', desc='Rotor current ratio')
 Slot_aspect_ratio1=Float(0.01, iotype='out', desc='Slot apsect ratio')
 Slot_aspect_ratio2=Float(0.01, iotype='out', desc='Slot apsect ratio')
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  h_yr=self.h_yr
  h_ys=self.h_ys
  q1=self.q1
  tau_p =self.tau_p
  B_g = self.B_g
  B_g1 = self.B_g1
  B_symax = self.B_symax
  B_rymax = self.B_rymax
  B_tsmax = self.B_tsmax
  B_trmax = self.B_trmax
  TC=self.TC
  TL=self.TL
  M_actual=self.M_actual
  W_2=self.W_2
  I_f=self.I_f
  E_p=self.E_p
  J_s=self.J_s
  gen_eff =self.gen_eff
  TM=self.TM
  K_rad=self.K_rad
  M_actual=self.M_actual
  Active=self.Active
  Structure=self.Structure
  W_1a=self.W_1a
  TC1=self.TC1
  TC2=self.TC2
  N_slots=self.N_slots
  Q_r=self.Q_r
  A_1=self.A_1
  J_s=self.J_s
  J_r=self.J_r
  lambda_ratio=self.lambda_ratio
  D_ratio=self.D_ratio
  Current_ratio=self.Current_ratio
  Slot_aspect_ratio1=self.Slot_aspect_ratio1
  Slot_aspect_ratio2=self.Slot_aspect_ratio2
  S_N=self.S_N
  p=self.p
  A_Curcalc=self.A_Curcalc
  A_Cuscalc=self.A_Cuscalc
  b_trmin=self.b_trmin
  b_tr=self.b_tr
  M_Fe=self.M_Fe
  M_Cus=self.M_Cus
  #tau_p_act=self.tau_p_act

  from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign,abs
  
  rho    =7850                # Kg/m3 steel density
  g1     =9.81                # m/s^2 acceleration due to gravity
  sigma  =21.5e3                # shear stress
  ratio  =0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
  mu_0   =pi*4e-7        # permeability of free space
  mu_r   =1.06
  phi    =90*2*pi/360 
  cofi   =0.9                 # power factor
  K_Cu   =4.786                  # Unit cost of Copper 
  K_Fe   =0.556                    # Unit cost of Iron 
  cstr   =0.50139                   # specific cost of a reference structure
  h_sy0  =0
  #h_sy   =0.04
  h_w    = 0.005
  m      =3
  b_s_tau_s=0.45
  b_r_tau_r=0.45
  self.q1=5
  #self.S_N=-0.1
  #A=abs(self.S_N)
  
  #self.S_N=-1*str(round(A,2))
  K_rs     =1/(-1*self.S_N)
  P_gennom = 5e6
  
  I_SN   = P_gennom/(sqrt(3)*3000)
  I_SN_r =I_SN/K_rs
  I_m    =0.3*I_SN_r
  I_RN_r =sqrt((I_SN_r**2)+(I_m**2))
  k_y1=sin(pi*0.5)*12/15
  q2=self.q1-1
  y_tau_p=12./15
  y_tau_r=10./12
  
  k_y1=sin(pi*0.5*y_tau_p)
  k_q1=sin(pi/6)/(self.q1*sin(pi/(6*self.q1)))
  k_y2=sin(pi*0.5*y_tau_r)
  k_q2=sin(pi/6)/(q2*sin(pi/(6*q2))) 
  
  k_wd1=k_y1*k_q1
  k_wd2=k_q2*k_y2
  k_sfil =0.65								 # Slot fill factor
  P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  rho_Cu=1.8*10**(-8)*1.4
  k_fes =0.9
  self.p=3
 
  
  freq=60
  
  #T = 5968.310366  
  
  P_gridnom=3e6
  E_pnom   = 3000
  
  n_nom = 12
  
  n_1=n_nom/(1-self.S_N)
  P_convnom=0.03*P_gridnom
  gear      =100   
  
  #self.h_r=self.h_s
  
  
  
  
  
   
  alpha_p=pi/2*.7
  dia=2*self.r_s             # air gap diameter
  g=(0.1+0.012*(P_gennom)**(1./3))*0.001
  self.lambda_ratio=self.l_s/dia
  r_r=self.r_s-g             #rotor radius
  
  self.tau_p=(pi*dia/(2*self.p)) 
  #self.tau_p_act=(pi*dia/(2*self.p))  # number of pole pairs
  N_slot=2
  self.N_slots=N_slot*self.p*self.q1*m    
  N_slots_pp=self.N_slots/(m*self.p*2)
  n  = self.N_slots/2*self.p/self.q1        #no of slots per pole per phase
  tau_s=self.tau_p/(m*self.q1)        #slot pitch
  b_s=b_s_tau_s*tau_s;       #slot width
  b_so=0.004;
  b_ro=0.004;
  b_t=tau_s-b_s              #tooth width
  v=0.29
  rho_Fe=8760
  C_1=(3+v)/4
  rotor_rad_max=(300e6/(C_1*rho_Fe*(2*20*pi/(1))**2))*0.5
  
  self.Q_r=2*self.p*m*q2
  tau_r=pi*(dia-2*g)/self.Q_r
  
  b_r=b_r_tau_r*tau_r
  self.b_tr=tau_r-b_r
  mu_rs=3;
  mu_rr=5;
  
  mu_rs=3;
  mu_rr=5;

  W_s=b_s/mu_rs;
  W_r=b_r/mu_rr;
  self.Slot_aspect_ratio1=self.h_s/b_s
  self.Slot_aspect_ratio2=self.h_r/b_r
  gamma_s	= (2*W_s/g)**2/(5+2*W_s/g)
  K_Cs=(tau_s)/(tau_s-g*gamma_s*0.5)  #page 3-13
  gamma_r		= (2*W_r/g)**2/(5+2*W_r/g)
  K_Cr=(tau_r)/(tau_r-g*gamma_r*0.5)  #page 3-13
  K_C=K_Cs*K_Cr
  g_eff=K_C*g
  om_m=gear*2*pi*n_nom/60
  om_e=self.p*om_m
  K_s=0.3
  n_c1=2    #number of conductors per coil
  a1 =2    # number of parallel paths
  self.W_1a=round(2*self.p*N_slots_pp*n_c1/a1)
  self.W_2=round(self.W_1a*k_wd1*K_rs/k_wd2)
  n_c2=self.W_2/(self.Q_r/m)
  self.B_g1=mu_0*3*self.W_2*self.I_f*2**0.5*k_y2*k_q2/(pi*self.p*g_eff*(1+K_s))
  self.B_g=self.B_g1*K_C
  #W_1a=0.97*3000/(sqrt(3)*sqrt(2)*60*k_wd*B_g1*2*dia*self.tau_p) # chapter 14, 14.9 #Boldea
  
  self.h_ys = self.B_g*self.tau_p/(self.B_symax*pi)
  self.B_rymax = self.B_symax
  self.h_yr=self.h_ys
  d_se=dia+2*(self.h_ys+self.h_s)  # stator outer diameter
  self.D_ratio=d_se/dia
  

  skew_r =pi*self.p*1/self.Q_r
 


  
  
  f = n_nom*self.p/60
  if (2*self.r_s>2):
      K_fills=0.65
  else:
  	  K_fills=0.4    
  beta_skew = tau_s/self.r_s
  k_wskew=sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  l_fs=2*(0.015+y_tau_p*self.tau_p/2/cos(40))+pi*(self.h_s)
  l_Cus = 2*self.W_1a*(l_fs+self.l_s)/a1             #shortpitch
  A_s = b_s*(self.h_s-h_w)
  A_scalc=b_s*1000*(self.h_s*1000-h_w*1000)
  A_Cus = A_s*self.q1*self.p*K_fills/self.W_1a
  self.A_Cuscalc = A_scalc*self.q1*self.p*K_fills/self.W_1a
  R_s=l_Cus*rho_Cu/A_Cus
  
  
 
  
  
  tau_r_min=pi*(dia-2*(g+self.h_r))/self.Q_r
  self.b_trmin=tau_r_min-b_r_tau_r*tau_r_min
  self.B_trmax = self.B_g*tau_r/self.b_trmin
  
  
  
  
  
  K_01=1-0.033*(W_s**2/g/tau_s)
  sigma_ds=0.0042
  K_02=1-0.033*(W_r**2/g/tau_r)
  sigma_dr=0.0062
  
  L_ssigmas=(2*mu_0*self.l_s*n_c1**2*self.N_slots/m/a1**2)*((self.h_s-h_w)/(3*b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=(2*mu_0*self.l_s*n_c1**2*self.N_slots/m/a1**2)*0.34*self.q1*(l_fs-0.64*self.tau_p*y_tau_p)/self.l_s                   #end winding leakage inductance
  L_ssigmag=(2*mu_0*self.l_s*n_c1**2*self.N_slots/m/a1**2)*(0.9*tau_s*self.q1*k_wd1*K_01*sigma_ds/g_eff)
  L_ssigma=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
  
 
  
  L_sm =6*mu_0*self.l_s*self.tau_p*(k_wd1*self.W_1a)**2/(pi**2*(self.p)*g_eff*(1+K_s))
  L_s=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator  inductance
  
  l_fr=(0.015+y_tau_r*tau_r/2/cos(40*pi/180))+pi*(self.h_r)
  L_rsl=(mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*((self.h_r-h_w)/(3*b_r)+h_w/b_ro)  #slot leakage inductance
  L_rel= (mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*0.34*q2*(l_fr-0.64*tau_r*y_tau_r)/self.l_s                   #end winding leakage inductance                  #end winding leakage inductance
  L_rtl=(mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*(0.9*tau_s*q2*k_wd2*K_02*sigma_dr/g_eff) # tooth tip leakage inductance
  L_r=(L_rsl+L_rtl+L_rel)/K_rs**2  # rotor leakage inductance
  sigma1=1-(L_sm**2/L_s/L_r)
  
  #Field winding
  k_fillr = 0.55
  diff=self.h_r-h_w
  A_Cur=k_fillr*self.p*q2*b_r*diff/self.W_2
  self.A_Curcalc=A_Cur*1e6
  L_cur=2*self.W_2*(l_fr+self.l_s)
  R_r=rho_Cu*L_cur/A_Cur
  R_R=R_r/(K_rs**2)
  
  om_s=(120*freq/self.p)*2*pi/60
  P_e=P_gennom/(1-self.S_N)
  
 
  self.E_p=om_s*self.W_1a*k_wd1*self.r_s*self.l_s*self.B_g1
  
  I_r=P_e/m/self.E_p         # rotor active current
  
  I_sm=self.E_p/(2*pi*freq*(L_s+L_sm)) # stator reactive current
  
  I_s=sqrt((I_r**2+I_sm**2))
  #I_s=P_e/3/self.E_p/0.8
  I_srated=P_gennom/3/self.E_p/K_rs
  self.J_s=I_s/(self.A_Cuscalc)
  self.J_r=I_r/(self.A_Curcalc)
  self.A_1=2*m*W_1a*I_s/pi/(2*self.r_s)
  self.Current_ratio=self.I_f/I_srated
  
  V_Cuss=m*l_Cus*A_Cus
  V_Cusr=m*L_cur*A_Cur
  V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-(2*m*self.q1*self.p*b_s*self.h_s*self.l_s))
  V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)
  r_r=self.r_s-g
  V_Fert=pi*self.l_s*(r_r**2-(r_r-self.h_r)**2)-2*m*self.q1*self.p*b_r*self.h_r*self.l_s
  V_Fery=self.l_s*pi*((r_r-self.h_r)**2-(r_r-self.h_r-self.h_yr)**2)
  self.M_Cus=(V_Cuss+V_Cusr)*8900
  M_Fest=V_Fest*7700
  M_Fesy=V_Fesy*7700
  M_Fert=V_Fert*7700
  M_Fery=V_Fery*7700
  self.M_Fe=M_Fest+M_Fesy+M_Fert+M_Fery
  M_gen=(self.M_Cus)+(self.M_Fe)
  K_gen=self.M_Cus*K_Cu+(self.M_Fe)*K_Fe #%M_pm*K_pm;
  L_tot=self.l_s
  #Gen_mass   =(1.28,2.27,4.13,6.84,12.62)*1000
  #Struc_Mass=(1.66,3.04,6.32,12.86,35.74)*1000
  #Struc_interp=interp1d(Gen_mass,Struc_Mass,fill_value='extrapolate')
  
  #C_str=cstr*0.5*((d_se/2)**3+(L_tot)**3)
  #self.Structure=0.5*((d_se/2)**3+(L_tot)**3)*7700
  #A=Struc_interp(M_gen)
  self.Structure=0.0002*M_gen**2+0.6457*M_gen+645.24
  self.M_actual=M_gen+self.Structure
  
  
  
  
  
  self.B_tsmax=self.B_g*tau_s/(b_t)
  
  self.TM=self.M_actual
  
  self.TC=K_gen+cstr*self.Structure
  K_R=1.2
  # losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P_Cuss=m*I_s**2*R_s*K_R
  P_Cusr=m*I_r**2*R_R
  P_Cusnom=P_Cuss+P_Cusr
  # B_tmax=B_pm*tau_s/b_t
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
  delta_v=1
  p_b=3*delta_v*I_r
  
  self.TL=P_Cusnom+P_Fesnom+p_b+P_add;
  self.gen_eff=(P_e-self.TL)*100/P_e
  self.J_s=self.I_s/A_Cuscalc
  
  S_GN=(P_gennom-self.S_N*P_gennom)/self.gen_eff/0.01
  T_e=self.p *(P_gennom*1.002)/(2*pi*freq*(1-self.S_N))
  self.TC1=T_e/(2*pi*sigma)
  self.TC2=self.r_s**2*self.l_s
  P_total_loss=TL #+ P_conv;
  
  print self.TM,P_Cusnom