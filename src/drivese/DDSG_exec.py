"""DDSG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float


class DDSG_exec(Component):
 """ Evaluates the total cost """
 r_s = Float(0, iotype='in', desc='airgap radius r_s')
 l_s = Float(0., iotype='in', desc='Stator core length l_s')
 h_s = Float(0., iotype='in', desc='Yoke height h_s')
 tau_p =Float(0., iotype='in', desc='Pole pitch self.tau_p')
 B_g = Float(1.06, iotype='out', desc='Peak air gap flux density B_g')
 B_gfm=Float(1.06, iotype='out', desc='Average air gap flux density B_g')
 B_symax = Float(1.2, iotype='out', desc='Peak Stator Yoke flux density self.B_symax')
 B_rymax = Float(1.2, iotype='out', desc='Peak rotor Yoke flux density B_rymax')
 B_tmax= Float(1.2, iotype='out', desc='Peak tooth density self.B_tmax')
 B_pc=Float(1.2, iotype='out', desc='Pole core flux density')
 I_f=Float(0, iotype='in', desc='Input current')
 N_f=Float(0, iotype='in', desc='field turns')
 W_1a=Float(0, iotype='out', desc='Stator turns')
 b_s=Float(0.0, iotype='out', desc='slot width')
 b_t=Float(0.0, iotype='out', desc='tooth width')
 h_ys=Float(0.0, iotype='in', desc='Yoke height')
 h_yr=Float(0.0, iotype='in', desc='rotor yoke height')
 h_p=Float(0.0, iotype='out', desc='Yoke height')
 b_p=Float(0.0, iotype='out', desc='rotor yoke height')
 M_actual=Float(0.0, iotype='out', desc='Actual mass')
 p=Float(0.0, iotype='out', desc='No of pole pairs')
 N_slots=Float(0.0, iotype='out', desc='No of slots')
 f=Float(0.0, iotype='out', desc='Output frequency')
 E_s=Float(0.0, iotype='out', desc='Stator phase voltage')
 f=Float(0.0, iotype='out', desc='Generator output frequency')
 I_s=Float(0.0, iotype='out', desc='Generator output phase current')
 R_s=Float(0.0, iotype='out', desc='Stator resistance')
 R_r=Float(0.0, iotype='out', desc='Rotor resistance')
 L_m=Float(0.0, iotype='out', desc='Stator synchronising inductance')
 J_s=Float(0.0, iotype='out', desc='Current density')
 J_f=Float(0.0, iotype='out', desc='rotor Current density')
 n_brushes=Float(0.0, iotype='out', desc='number of brushes')
 TC= Float(0.0, iotype='out', desc='Total cost')
 TL=Float(0.0, iotype='out', desc='Total Loss')
 K_rad=Float(0.0, iotype='out', desc='K_rad')
 TM=Float(0.0, iotype='out', desc='Total loss')
 gen_eff=Float(0.01, iotype='out', desc='Generator efficiency')
 P_gennom=Float(0.01, iotype='out', desc='Generator efficiency')
 Iron=Float(0.01, iotype='out', desc='Iron mass')
 Copper=Float(0.01, iotype='out', desc='Copper mass')
 Inactive=Float(0.01, iotype='out', desc='Structure mass')
 u_all_r=Float(0.0, iotype='out', desc='Allowable radial deflection')
 u_all_s=Float(0.0, iotype='out', desc='Stator Allowable radial deflection')
 y_all=Float(0.0, iotype='out', desc='Allowable axial deflection')
 z_all_r=Float(0.0, iotype='out', desc='Allowable circum deflection')
 z_all_s=Float(0.0, iotype='out', desc='Allowable circum deflection')
 b_all_s=Float(0.0, iotype='out', desc='Allowable arm space')
 b_all_r=Float(0.0, iotype='out', desc='Allowable arm space')
 Axial_delta_stator=Float(0.0, iotype='out', desc='Stator axial deflection')
 Radial_delta_stator=Float(0.0, iotype='out', desc='Stator radial deflection')
 Circum_delta_stator=Float(0.0, iotype='out', desc='Stator circum deflection')
 Axial_delta_rotor=Float(0.0, iotype='out', desc='Rotor axial deflection')
 Radial_delta_rotor=Float(0.0, iotype='out', desc='Rotor radial deflection')
 Circum_delta_rotor=Float(0.0, iotype='out', desc='Stator circum deflection')
 N_slots=Float(0.0, iotype='out', desc='Rotor slots')
 A_1=Float(0.0, iotype='out', desc='Specific current loading')
 n_s =Float(0, iotype='in', desc='number of arms n')
 b_st = Float(0, iotype='in', desc='arm width b_r')
 d_s = Float(0, iotype='in', desc='arm depth d_r')
 t_ws =Float(0, iotype='in', desc='arm depth thickness self.t_wr')
 #rotor
 n_r =Float(0, iotype='in', desc='number of arms n')
 b_r = Float(0, iotype='in', desc='arm width b')
 d_r = Float(0., iotype='in', desc='arm depth d')
 t_wr =Float(0., iotype='in', desc='arm depth thickness self.t_w')
 Cost=Float(0, iotype='out', desc='Rotor Current density')
 P_gennom=Float(0, iotype='out', desc='Nominal power')
 A_Cuscalc=Float(0, iotype='out', desc='Stator Conductor cross-section')
 A_Curcalc=Float(0, iotype='out', desc='Rotor Conductor cross-section')
 Load_mmf_ratio=Float(0, iotype='out', desc='mmf_ratio')
 Power_ratio=Float(0, iotype='out', desc='Power_ratio')
 K_load=Float(0, iotype='out', desc='Power_ratio')
 Slot_aspect_ratio=Float(0, iotype='out', desc='Power_ratio')
 
 def execute(self):
  r_s = self.r_s
  l_s = self.l_s
  h_s = self.h_s
  tau_p =self.tau_p
  B_g = self.B_g
  B_symax = self.B_symax
  B_rymax=self.B_rymax
  B_tmax=self.B_tmax
  B_pc=self.B_pc
  B_gfm=self.B_gfm
  TC=self.TC
  TL=self.TL
  M_actual=self.M_actual
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
  TM=self.TM
  K_rad=self.K_rad
  N_f=self.N_f
  W_1a=self.W_1a
  p=self.p
  f=self.f
  A_Cuscalc=self.A_Cuscalc
  A_Curcalc=self.A_Curcalc
  A_1=self.A_1
  n_brushes=self.n_brushes
  
  Iron=self.Iron
  Copper=self.Copper
  Inactive=self.Inactive
  I_f=self.I_f
  R_r=self.R_r
  N_slots=self.N_slots
  Axial_delta_stator=self.Axial_delta_stator
  Radial_delta_stator=self.Radial_delta_stator
  Circum_delta_stator=self.Circum_delta_stator
  Radial_delta_rotor=self.Radial_delta_rotor
  Axial_delta_rotor=self.Axial_delta_rotor
  Circum_delta_rotor=self.Circum_delta_rotor
  u_all_r=self.u_all_r
  u_all_s=self.u_all_s
  y_all=self.y_all
  z_all_r=self.z_all_r
  z_all_s=self.z_all_s
  b_all_r=self.b_all_r
  b_all_s=self.b_all_s
  n_s=self.n_s
  d_s=self.d_s
  b_st=self.b_st
  t_ws=self.t_ws
  n_r=self.n_r
  d_r=self.d_r
  b_r=self.b_r
  t_wr=self.t_wr
  J_f=self.J_f
  Cost=self.Cost
  P_gennom=self.P_gennom
  Load_mmf_ratio=self.Load_mmf_ratio
  Power_ratio=self.Power_ratio
  K_load=self.K_load
  Slot_aspect_ratio=self.Slot_aspect_ratio
  from math import pi, cos,cosh,sinh, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
  import numpy as np
  from numpy import sign
  
  rho    =7850                # Kg/m3 steel density
  
  g1     =9.81                # m/s^2 acceleration due to gravity
  E      =2e11                # N/m^2 young's modulus
  sigma  =48.373e3                # shear stress
  ratio  =0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
  mu_0   =pi*4e-7        # permeability of free space
  mu_r   =1.06
  phi    =90*2*pi/360 
  cofi   =0.9                 # power factor
  K_Cu   =4.786                  # Unit cost of Copper 
  K_Fe   =0.556                   # Unit cost of Iron 
  h_sy0  =0
  #h_sy   =0.04
  h_w    =0.005
  b_so	=  0.004
  m      =3
  q1     =2
  b_s_tau_s=0.45
  k_sfil =0.65								 # Slot fill factor
  P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
  
  rho_Cu=1.8*10**(-8)*1.4
  k_fes =0.9
  R_o = 0.43 #0.523950817
  T = 4143289.841  #9549296.586 #4143289.841  #4143289.841 #1790493.11 #9947183.943 #250418.6168  # #4143289.841 #9549296.586 #1790493.11 #9947183.943 #8.32E+05  # #9549296.586  #4143289.841 #250418.6168 #698729.0185 #1790493.11
  self.P_gennom = 5e6 #280.3e3 #10e6
  P_gridnom=3e6
  E_pnom   = 3000
  n_nom = 12.1 #20.5 #12.1 #12.1 #16 #9.6 #9.6 #3.216 #9.6 #rpm #10
  P_convnom=0.03*P_gridnom
  
  gear=1

  self.K_rad=self.l_s/(2*self.r_s)
  #mass_st_lam=rho*2*pi.*R.*l.*h_ry;  % mass of rotor yoke steel
  #W=g*sin(phi).*((rho*(2*pi.*R./N).*l.*t)+(mass_PM./N)+(mass_st_lam./N)); % weight of rotor cylinder

  alpha_p=pi/2*.7
  dia=2*self.r_s             # air gap diameter
  g=0.001*dia
  if(g<0.005):
  	g=0.005
  r_r=self.r_s-g             #rotor radius
  d_se=dia+2*self.h_s+2*self.h_ys  # stator outer diameter
  self.p=round(pi*dia/(2*self.tau_p))  # number of pole pairs
  N_slot=2   #number of conductors per slot
  self.N_slots=2*self.p*q1*m    # number of slots of stator phase winding
  n  = self.N_slots/2*self.p/m/q1        #no of slots per pole per phase
  N_conductors=self.N_slots*N_slot
  self.W_1a=N_conductors/2/3
  alpha =180/self.N_slots/self.p    #electrical angle
  K_d =sin(n*alpha/2)/n*sin(alpha/2)
  #tau_s=self.tau_p/m/q1       #slot pitch
  tau_s=pi*dia/self.N_slots
  h_ps=0.1*self.tau_p
  b_pc=0.4*self.tau_p
  h_pc=0.6*self.tau_p
  self.h_p=0.7*tau_p
  self.b_s=tau_s * b_s_tau_s        #slot width
  self.Slot_aspect_ratio=self.h_s/self.b_s
  self.b_t=tau_s-self.b_s              #tooth width
  
  g_a=g
  b_r=self.b_s
  K_C1=(tau_s+10*g_a)/(tau_s-self.b_s+10*g_a)  # salient pole rotor
  g_1=K_C1*g
  om_m=gear*2*pi*n_nom/60
  #om_e=self.p*om_m/2
  om_e=60
  self.f    =  n_nom*self.p/60
  if (2*self.r_s>2):
      K_fills=0.65
  else:
  	  K_fills=0.4  
  y_tau_p=1  #fullpitch
  k_y1=sin(y_tau_p*pi/2) 
  k_q1=sin(pi/6)/q1/sin(pi/6/q1) 
  k_wd=k_y1*k_q1  
  beta_skew = tau_s/self.r_s
  k_wskew=sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
  #W_1a=0.97*3000/(sqrt(3)*sqrt(2)*60*k_wd*B_gfm*2*dia*self.tau_p) # chapter 14, 14.9 #Boldea

   
  #Field winding
  J_cur=10
  K_fill=0.62
  alpha_i=0.7
  self.b_p=self.tau_p*alpha_i
  W_p=0.5*tau_s
  W_c=(self.b_p-W_p)*0.5
  K_h=0.15
  
  
 
  k_s=0.2 #magnetic saturation factor 
    
  k_wd			=sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
  self.N_f=round(self.N_f)
  #B_gfm=0.85
  #self.N_f=0.85*(g_1*(1+k_s))/(self.I_f*mu_0)
  self.B_gfm=mu_0*self.N_f*self.I_f/(g_1*(1+k_s))  #No load air gap flux density
  self.B_g=self.B_gfm*4*sin(0.5*self.b_p*pi/self.tau_p)/pi  # fundamental component
  
  
  tau_r=0.5*(self.b_p-2*k_s*self.tau_p/(6*q1+1)) #damper bar pitch
  k_fillr = 0.7
  Wf_IF = J_cur*W_c*self.h_p*K_fill
  
  
  
  
  k_s=0.2 #magnetic saturation factor 
    
  k_wd			=sin(pi/6)/q1/sin(pi/6/q1)      # winding factor
  
  
  self.N_f=round(self.N_f)
  shortpitch=0
  l_Cus = 2*self.W_1a*(2*(self.tau_p-shortpitch/m/q1)+self.l_s)             
  A_s = self.b_s*(self.h_s-h_w)
  A_scalc=self.b_s*1000*(self.h_s*1000-h_w*1000)
  A_Cus = A_s*q1*self.p*K_fills/self.W_1a
  self.A_Cuscalc = A_scalc*q1*self.p*K_fills/self.W_1a
  self.R_s=l_Cus*rho_Cu/A_Cus
  
  k_fillr = 0.7
  Wf_IF = J_cur*W_c*self.h_p*K_fill
  cos_phi=0.85
  I_srated=self.P_gennom/(sqrt(3)*5000*cos_phi)
  l_pole=self.l_s-0.05+0.120  # 50mm smaller than stator and 120mm longer to accommodate end stack
  K_fe=0.95
  l_pfe=l_pole*K_fe
  #self.N_f=(B_pc*b_pc -(2*self.tau_p/pi)*self.B_g*cos(delta_m))/(2*mu_0*self.I_f*((2*h_ps/(self.tau_p-b_p))+(h_pc/(self.tau_p-b_pc))))
  l_Cur=4*self.p*self.N_f*(l_pfe+b_pc+pi/4*(pi*(r_r-h_pc-h_ps)/self.p-b_pc))
  A_Cur=k_fillr*h_pc*0.5/self.N_f*(pi*(r_r-h_pc-h_ps)/self.p-b_pc)
  self.A_Curcalc=k_fillr*h_pc*1000*0.5/self.N_f*(pi*(r_r-h_pc-h_ps)*1000/self.p-b_pc*1000)
  #A_Cur=2.45e-4 #1.22549e-4
  Slot_Area=A_Cur*2*self.N_f/k_fillr
  self.R_r=rho_Cu*l_Cur/A_Cur
  self.J_f=self.I_f/self.A_Curcalc
  
  
  
  
  
   #Stator teeth flux density
  self.B_symax=self.tau_p*self.B_g/pi/self.h_ys #stator yoke flux density
  L_fg=2*mu_0*self.p*self.l_s*4*self.N_f**2*((h_ps/(self.tau_p-self.b_p))+(h_pc/(3*pi*(r_r-h_pc-h_ps)/self.p-b_pc)))
  self.E_s=2*self.W_1a*self.l_s*self.r_s*k_wd*k_wskew*om_m*self.B_g/sqrt(2) #no load voltage
  
  self.I_s=(self.E_s-(self.E_s**2-4*self.R_s*self.P_gennom/m)**0.5)/(2*self.R_s)
  #self.I_s=self.P_gennom/m/self.E_s
  self.A_1 = 6*self.W_1a*self.I_s/(pi*dia*self.l_s)
  self.J_s=self.I_s/self.A_Cuscalc
  delta_m=0
  self.B_pc=(1/b_pc)*((2*self.tau_p/pi)*self.B_g*cos(delta_m)+(2*mu_0*self.I_f*self.N_f*((2*h_ps/(self.tau_p-self.b_p))+(h_pc/(self.tau_p-b_pc)))))
  self.B_rymax= 0.5*b_pc*self.B_pc/self.h_yr
  self.B_tmax=(self.B_gfm+self.B_g)*tau_s*0.5/self.b_t 
  L_ssigmas=2*mu_0*self.l_s*self.W_1a**2/self.p/q1*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
  L_ssigmaew=mu_0*1.2*self.W_1a**2/self.p*1.2*(2/3*self.tau_p+0.01)                    #end winding leakage inductance
  L_ssigmag=2*mu_0*self.l_s*self.W_1a**2/self.p/q1*(5*(g/b_so)/(5+4*(g/b_so))) # tooth tip leakage inductance
  L_ssigma=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
    
  
  At_g=g_1*self.B_gfm/mu_0
  At_t=self.h_s*(400*self.B_tmax+7*(self.B_tmax)**13)
  At_sy=self.tau_p*0.5*(400*self.B_symax+7*(self.B_symax)**13)
  At_pc=(h_pc+h_ps)*(400*self.B_pc+7*(self.B_pc)**13)
  At_ry=self.tau_p*0.5*(400*self.B_rymax+7*(self.B_rymax)**13)
  g_eff = (At_g+At_t+At_sy+At_pc+At_ry)*g_1/At_g
  self.L_m = 6*k_wd**2*self.W_1a**2*mu_0*self.r_s*self.l_s/pi/g_eff/self.p**2
  
  B_r1=(mu_0*self.I_f*self.N_f*4*sin(0.5*(b_p/self.tau_p)*pi))/g_eff/pi 
  
  L_dm= (self.b_p/self.tau_p +(1/pi)*sin(pi*self.b_p/self.tau_p))*self.L_m
  L_qm=(self.b_p/self.tau_p -(1/pi)*sin(pi*self.b_p/self.tau_p)+2/(3*pi)*cos(self.b_p*pi/2*self.tau_p))*self.L_m
  delta_m=(atan(om_e*L_qm*self.I_s/self.E_s))
  L_d=L_dm+L_ssigma
  L_q=L_qm+L_ssigma  
  I_sd=self.I_s*sin(delta_m)
  I_sq=self.I_s*cos(delta_m)  
  E_p=om_e*L_dm*I_sd+sqrt(self.E_s**2-(om_e*L_qm*I_sq)**2)
  M_sf =mu_0*8*self.r_s*self.l_s*k_wd*self.W_1a*self.N_f*sin(0.5*self.b_p/self.tau_p*pi)/(self.p*g_eff*pi)
  I_f1=sqrt(2)*(E_p)/(om_e*M_sf)
  I_f2=(E_p/self.E_s)*self.B_g*g_eff*pi/(4*self.N_f*mu_0*sin(pi*self.b_p/2/self.tau_p))
  phi_max_stator=k_wd*self.W_1a*pi*self.r_s*self.l_s*2*mu_0*self.N_f*self.I_f*4*sin(0.5*self.b_p/self.tau_p/pi)/(self.p*pi*g_eff*pi)
  #M_sf=mu_0*8*self.r_s*self.l_s*k_wd*self.W_1a*self.N_f*sin(0.5*b_p/self.tau_p/pi)/(self.p*g_eff*pi)
  
  #self.J_r=self.I_f/self.A_Curcalc
  
  
  L_tot=self.l_s+2*self.tau_p
  
 
  
  
  
  V_fn=500
  Power_excitation=V_fn*2*self.I_f   #total rated power in excitation winding
  self.Power_ratio =Power_excitation*100/self.P_gennom
  
  tau_r=0.5*(self.b_p-2*k_s*self.tau_p/(6*q1+1))
 
  
  #self.TC=K_gen+C_str
  # losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  t=self.h_yr
  t_s=self.h_ys

   
    
  L_tot=self.l_s+2*self.tau_p
  V_Cuss=m*l_Cus*A_Cus
  V_Cusr=l_Cur*A_Cur
  V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-2*m*q1*self.p*self.b_s*self.h_s*self.l_s)
  V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)
  V_Fert=2*self.p*l_pfe*(h_pc*b_pc+self.b_p*h_ps)
  
  V_Fery=l_pfe*pi*((r_r-h_ps-h_pc)**2-(r_r-h_ps-h_pc-self.h_yr)**2)
  M_Cus=(V_Cuss+V_Cusr)*8900
  M_Fest=V_Fest*7700
  M_Fesy=V_Fesy*7700
  M_Fert=V_Fert*7700
  M_Fery=V_Fery*7700
  M_Fe=M_Fest+M_Fesy+M_Fert+M_Fery
  
  K_gen=M_Cus*K_Cu+(M_Fe)*K_Fe #%M_pm*K_pm;
  #C_str=cstr*0.5*((d_se/2)**3+(L_tot)**3)*7700
  cos_phi=0.85
  I_snom=self.P_gennom/(3*self.E_s*cos_phi)
  
  F_1no_load=3*2**0.5*self.W_1a*k_wd*self.I_s/(pi*self.p)
  Nf_If_no_load=self.N_f*self.I_f
  SCR=0.7
  #F_1_rated=60e3*self.tau_p/2
  F_1_rated=(3*2**0.5*self.W_1a*k_wd*I_srated)/(pi*self.p)
  Nf_If_rated=2*Nf_If_no_load
  self.Load_mmf_ratio=Nf_If_rated/F_1_rated
  
  #self.TC=K_gen+C_str
  # losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P_Cuss=m*self.I_s**2*self.R_s
  P_Cusr=self.I_f**2*self.R_r
  K_R=1.2 
  P_Sc=m*(self.R_s)*K_R*(I_snom)**2*1  #losses to include skin effect
  P_Cusrnom=(self.I_f)**2*self.R_r
  P_Custotal=P_Cuss+P_Cusr
  P_Cusnom_total=P_Sc+P_Cusrnom
  P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*50))**2)
  P_Fesynom=P_Hyys+P_Ftys
  P_Hyd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0h*om_e/(2*pi*50))
  P_Ftd=M_Fest*(self.B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*50))**2)
  delta_v=1
  self.n_brushes=(self.I_f*2/120)
  if (self.n_brushes<0.5):
     self.n_brushes=1
  else:
  	 self.n_brushes=round(self.n_brushes)
  p_b=2*delta_v*(self.I_f)
  I_fn=2*self.I_f
  P_Festnom=P_Hyd+P_Ftd
  
  
  #P_mech =1.5e-3*(2*pi*n_nom/60)**3*2*r_r**5*(1+5*l_pole/(2*r_r)) #windage
  P_mech =0.005*self.P_gennom

  self.TL=P_Cusnom_total+P_Festnom+P_Fesynom+p_b
  self.gen_eff=self.P_gennom*100/(self.TL+self.P_gennom)
  t=self.h_yr
  t_s=self.h_ys

  q3				= self.B_g**2/2/mu_0   # normal stress
  l					= self.l_s                          #l-stator core length
  l_u       =k_fes * self.l_s                   #useful iron stack length
  We				=self.tau_p
  l_b       = 2*self.tau_p  #end winding length
  l_e       =self.l_s+2*0.001*self.r_s     # equivalent core length


  a_r				= (self.b_r*self.d_r)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr))  # cross-sectional area of rotor armms
  a_s       = (self.b_st*self.d_s)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws))
  A_r				= l*t 
  A_st      =l*t_s                     # cross-sectional area of rotor cylinder
  N_r				= round(self.n_r)
  N_st			= round(self.n_s)
  theta_r		=pi/N_r                             # half angle between spokes
  theta_s		=pi/N_st 
  I_r				=l*self.h_yr**3/12                         # second moment of area of rotor cylinder
  I_st      =l*self.h_ys**3/12 
  I_arm_axi_r	=((self.b_r*self.d_r**3)-((self.b_r-2*self.t_wr)*(self.d_r-2*self.t_wr)**3))/12  # second moment of area of rotor arm
  I_arm_axi_s	=((self.b_st*self.d_s**3)-((self.b_st-2*self.t_ws)*(self.d_s-2*self.t_ws)**3))/12  # second moment of area of stator arm
  I_arm_tor_r	= ((self.d_r*self.b_r**3)-((self.d_r-2*self.t_wr)*(self.b_r-2*self.t_wr)**3))/12  # second moment of area of rotot arm w.r.t torsion
  I_arm_tor_s	= ((self.d_s*self.b_st**3)-((self.d_s-2*self.t_ws)*(self.b_st-2*self.t_ws)**3))/12  # second moment of area of rotot arm w.r.t torsion
  R					= r_r-h_ps-h_pc-0.5*self.h_yr
  R_1				= R-self.h_yr*0.5                               # inner radius of rotor cylinder
  dia				=  2*self.r_s              # air gap diameter
  g         =  0.001*dia
  R_st 			=(self.r_s+self.h_s+self.h_ys*0.5)
  R_1s      = R_st-self.h_ys*0.5
  k_1				= sqrt(I_r/A_r)                               # radius of gyration
  k_2       = sqrt(I_st/A_st)
  m1				=(k_1/R)**2 
  c					=R/500
  
  
  m2        =(k_2/R_st)**2                                  
  l_ir			=R                                      # length of rotor arm beam at which rotor cylinder acts
  l_iir			=R_1 
  l_is      =R_st-R_o
  l_iis     =l_is 
  l_iiis    =l_is
  
  #self.b_1r =2*R_o*asin(b_r*0.5/R_o)
  #self.b_1s =2*R_o*asin(b_s*0.5/R_o)
  self.b_all_r			=2*pi*R_o/N_r 
  self.b_all_s			=2*pi*R_o/N_st                           
  
                                    
  w_r					=rho*g1*sin(phi)*a_r*N_r
  
  #self.h_yr =0.5*B_r/self.B_rymaxmax*self.tau_p*ratio*(h_m/mu_r)/((h_m/mu_r)+c)  # rotor yoke height

  #print theta
  mass_st_lam=7700*2*pi*(R+0.5*self.h_yr)*l*self.h_yr                                    # mass of rotor yoke steel  
  W				=g1*sin(phi)*(mass_st_lam+(V_Cusr*8900)+M_Fert)/N_r  # weight of rotor cylinder
  
  Numer=R**3*((0.25*(sin(theta_r)-(theta_r*cos(theta_r)))/(sin(theta_r))**2)-(0.5/sin(theta_r))+(0.5/theta_r))
  Pov=((theta_r/(sin(theta_r))**2)+1/tan(theta_r))*((0.25*R/A_r)+(0.25*R**3/I_r))
  Qov=R**3/(2*I_r*theta_r*(m1+1))
  Lov=(R_1-R_o)/a_r
  Denom=I_r*(Pov-Qov+Lov) # radial deflection % rotor
  self.u_Ar				=(q3*R**2/E/self.h_yr)*(1+Numer/Denom)
  
  self.u_all_r     =R/10000 # allowable radial deflection
  self.u_all_s    = R_st/10000

  self.y_Ar       =(W*l_ir**3/12/E/I_arm_axi_r)+(w_r*l_iir**4/24/E/I_arm_axi_r)  # axial deflection
  self.y_all     =2*l/100    # allowable axial deflection

  self.z_all_s     =0.05*2*pi*R_st/360  # allowable torsional deflection
  self.z_all_r     =0.05*2*pi*R/360  # allowable torsional deflection
  self.z_A_r       =(2*pi*(R-0.5*self.h_yr)*l/N_r)*sigma*(l_ir-0.5*self.h_yr)**3/3/E/I_arm_tor_r       # circumferential deflection
  #sigma=k_wd*self.A_1*self.B_g/(sqrt(2))
  self.Axial_delta_rotor=self.y_Ar
  self.Radial_delta_rotor=self.u_Ar
  self.Circum_delta_rotor=self.z_A_r 
  
  
  self.TC1=T/(2*pi*sigma)
  self.TC2=R**2*l
  self.TC3=R_st**2*l
 
  val_str_cost_rotor		= 0.50139*((N_r*(R_1-R_o)*a_r*rho))
  
  #val_str_rotor		= ((2*pi*t*l*R*rho)+(N_r*(R_1-R_o)*a_r*rho))+((sign(self.u_Ar-self.u_all_r)+1)*(u_Ar-self.u_all_r)**3*1e200)+(((sign(y_Ar-self.y_all)+1)*(y_Ar-self.y_all)**3*1e100))+(((sign(z_A_r-self.z_all_r)+1)*(z_A_r-self.z_all_r)**3*1e100))+(((sign(self.b_r-self.b_all_r)+1)*(self.b_r-self.b_all_r)**3*5e50))+((sign(self.TC1-self.TC2)+1)*((self.TC1-self.TC2))**3*5e50)
  val_str_rotor		= 7700*2*pi*(R+0.5*self.h_yr)*l*self.h_yr+(N_r*(R_1-R_o)*a_r*rho)
 
 
  
  
  mass_st_lam_s= M_Fest+pi*l*7700*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2) 
  
  W_is			=g1*sin(phi)*(rho*l*self.d_s**2*0.5) # weight of rotor cylinder                               # length of rotor arm beam at which self-weight acts
  W_iis     =g1*sin(phi)*(V_Cuss*8900+mass_st_lam_s)/2/N_st
  w_s         =rho*g1*sin(phi)*a_s*N_st
  
  #stator structure
  
  Numers=R_st**3*((0.25*(sin(theta_s)-(theta_s*cos(theta_s)))/(sin(theta_s))**2)-(0.5/sin(theta_s))+(0.5/theta_s))
  Povs=((theta_s/(sin(theta_s))**2)+1/tan(theta_s))*((0.25*R_st/A_st)+(0.25*R_st**3/I_st))
  Qovs=R_st**3/(2*I_st*theta_s*(m2+1))
  Lovs=(R_1s-R_o)*0.5/a_s
  Denoms=I_st*(Povs-Qovs+Lovs) 
 
  self.u_As				=(q3*R_st**2/E/t_s)*(1+Numers/Denoms)
  X_comp1 = (W_is*l_is**3/12/E/I_arm_axi_s)
  X_comp2 =(W_iis*l_iis**4/24/E/I_arm_axi_s)
  X_comp3 =w_s*l_iiis**4/24/E/I_arm_axi_s
  
  self.y_As       =X_comp1+X_comp2+X_comp3  # axial deflection
  
  self.z_As  =2*pi*(R_st+0.5*self.h_ys)*l/(2*N_st)*sigma*(l_is+0.5*self.h_ys)**3/3/E/I_arm_tor_s 
  self.Axial_delta_stator=self.y_As 
  self.Radial_delta_stator=self.u_As	
  self.Circum_delta_stator=self.z_As
  
  
  
  
  mass_stru_steel  =2*(N_st*(R_1s-R_o)*a_s*rho)
  #val_str_stator		= mass_stru_steel+mass_st_lam_s+((sign(u_As-u_all)+1)*(u_As-u_all)**3*1e100)+(((sign(y_As-y_all)+1)*(y_As-y_all)**3*1e100))+(((sign(z_As-z_all_s)+1)*(z_As-z_all_s)**3*1e100))+(((sign(self.b_st-b_all_s)+1)*(self.b_st-b_all_s)**3*1e50))+((sign((T/2/pi/sigma)-(R_st**2*l))+1)*((T/2/pi/sigma)-(R_st**2*l))**3*1e50)
  val_str_stator		= mass_stru_steel+mass_st_lam_s
  #val_str_cost_stator =	0.50139*mass_stru_steel+0.556*mass_st_lam_s+((sign(u_As-u_all)+1)*(u_As-u_all)**3*1e100)+(((sign(y_As-y_all)+1)*(y_As-y_all)**3*5e17))+(((sign(z_As-z_all_s)+1)*(z_As-z_all_s)**3*1e100))+(((sign(self.b_st-b_all_s)+1)*(self.b_st-b_all_s)**3*1e100))+((sign((T/2/pi/sigma)-(R_st**2*l))+1)*((T/2/pi/sigma)-(R_st**2*l))**3*1e100)
  val_str_cost_stator =	0.50139*mass_stru_steel
  
  val_str_mass=val_str_rotor+val_str_stator
  
    
  M_Fe=M_Fest+M_Fesy+M_Fert+M_Fery
  M_gen=M_Cus+M_Fert
  
   
  self.TM =val_str_mass+M_gen

  #"% cost of losses"
  val_str_cost=val_str_cost_rotor+val_str_cost_stator
  
  
  K_gen		=M_Cus*4.786+(M_Fest+pi*l*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2) +M_Fery+M_Fert)*0.556
  Stator=mass_stru_steel
  Rotor=(N_r*(R_1-R_o)*a_r*rho)
  
  #self.Inactive=((d_se/2)**3+(L_tot)**3)*7700*0.2
  self.Inactive=Stator+Rotor
  
  
  self.TC=val_str_cost+K_gen
  
  #self.Cost=K_gen+0.5013*mass_stru_steel+2*mass_st_lam_s+0.50139*((pi*l*(R_a**2-R**2)*rho)+pi*self.t_d*(R**2-R_o**2)*rho)
  
  
  #self.Inactive=((d_se/2)**3+(L_tot)**3)*7700*0.2
  self.Inactive=Stator+Rotor
 
  self.M_actual=(2*pi*t*l*(R+0.5*self.h_yr)*7700)+M_Fert+M_Cus+self.Inactive+M_Fest+pi*l*7700*((R_st+0.5*h_ys)**2-(R_st-0.5*h_ys)**2)
  self.Iron=M_Fest+7700*pi*l*((R_st+0.5*self.h_ys)**2-(R_st-0.5*self.h_ys)**2)+M_Fert+(2*pi*t*l*(R+0.5*self.h_yr)*7700)
  self.Copper=M_Cus
  self.Cost=self.Inactive*0.50139+K_gen
  
  #print ((V_Cusr*8900)+M_Fert)*g1,(V_Cuss*8900+M_Fest)*g1,V_Cuss*8900+mass_st_lam_s,W_is,l_is,E,I_arm_axi_s,W_iis,l_iis,w_s,l_iiis
  print self.gen_eff,self.TM,self.TC