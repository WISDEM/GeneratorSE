"""DFIG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.main.api import Component
from openmdao.main.api import set_as_top
from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Float, Bool, Int, Str, Array, Enum
from openmdao.lib.drivers.api import COBYLAdriver, CONMINdriver,NEWSUMTdriver,SLSQPdriver,Genetic
from openmdao.lib.datatypes.api import Float
from scipy.interpolate import interp1d
import numpy as np
from numpy import array
import pandas

class DFIG(Component):
	""" Evaluates the total cost """
	""" Evaluates the total cost """
	r_s = Float(iotype='in', desc='airgap radius r_s')
	l_s = Float(iotype='in', desc='Stator core length l_s')
	h_s = Float(iotype='in', desc='Yoke height h_s')
	h_r = Float(iotype='in', desc='Stator slot height ')
	B_symax = Float(iotype='in', desc='Peak Yoke flux density B_ymax')
	machine_rating=Float(iotype='in', desc='Machine rating')
	n_nom=Float(iotype='in', desc='rated speed ')
	highSpeedSide_cm =Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc=' high speed sidde COM [x, y, z]')
	highSpeedSide_length =Float(iotype='in', desc='high speed side length')
	Gearbox_efficiency=Float(iotype='in', desc='Gearbox efficiency')
	S_Nmax =Float(iotype='in', desc='Stator slot height ')
	I_0=Float(iotype='in', desc='Rotor current at no-load')
	
	K_Cu=Float(iotype='in', desc='Specific cost of copper')
	K_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	K_Fes=Float(iotype='in', desc='Specific cost of structural steel')
	rho_Fe=Float(iotype='in', desc='Steel density kg/m^3')
	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
	
	tau_p=Float(iotype='out', desc='Pole pitch')
	p=Float(0, iotype='out', desc='Pole pairs')
	B_g = Float(iotype='out', desc='Peak air gap flux density B_g')
	q1   =Float(iotype='out', desc='Slots per pole per phase')
	h_ys=Float(iotype='out', desc=' stator yoke height')
	h_yr=Float(iotype='out', desc=' rotor yoke height')
	B_g = Float(iotype='out', desc='Peak air gap flux density B_g')
	B_g1 = Float(iotype='out', desc='air gap flux density fundamental ')
	B_rymax = Float(iotype='out', desc='maximum flux density in rotor yoke')
	B_tsmax  =Float(iotype='out', desc='maximum tooth flux density in stator')
	B_trmax = Float(iotype='out', desc='maximum tooth flux density in rotor')
	N_slots = Float(iotype='out', desc='Stator slots')
	Q_r = Float(iotype='out', desc='Rotor slots')

	N_s= Float(0, iotype='out', desc='Stator turns')
	N_r= Float(iotype='out', desc='Rotor turns')
	M_actual=Float( iotype='out', desc='Actual mass')
	p=Float( iotype='out', desc='No of pole pairs')
	f=Float( iotype='out', desc='Output frequency')
	E_p=Float( iotype='out', desc='Stator phase voltage')
	I_s=Float( iotype='out', desc='Generator output phase current')
	b_s=Float( iotype='out', desc='stator slot width')
	b_r=Float( iotype='out', desc='rotor slot width')
	b_t=Float( iotype='out', desc='stator tooth width')
	b_trmin=Float( iotype='out', desc='minimum tooth width')
	b_tr=Float( iotype='out', desc='rotor tooth width')
	gen_eff=Float(iotype='out', desc='Generator efficiency')
	Structure=Float(iotype='out', desc='Structural mass')
	Active=Float(iotype='out', desc='Active Mass')
	TC1=Float(iotype='out', desc='Torque constraint')
	TC2=Float(iotype='out', desc='Torque constraint')
	A_1=Float(iotype='out', desc='Specific current loading')
	J_s=Float(iotype='out', desc='Stator winding current density')
	J_r=Float(iotype='out', desc='Rotor winding current density')
	K_rad=Float(iotype='out', desc='Stack length ratio')
	D_ratio=Float(iotype='out', desc='Stator diameter ratio')
	A_Cuscalc=Float(iotype='out', desc='Stator conductor cross-section')
	A_Curcalc=Float(iotype='out', desc='Rotor conductor cross-section')
	Current_ratio=Float(iotype='out', desc='Rotor current ratio')
	Slot_aspect_ratio1=Float(iotype='out', desc='Slot apsect ratio')
	Slot_aspect_ratio2=Float(iotype='out', desc='Slot apsect ratio')
	Costs= Float( iotype='out', desc='Total cost')
	Losses=Float( iotype='out', desc='Total power Loss')
	K_rad=Float( iotype='out', desc='Aspect ratio')
	Mass=Float(iotype='out', desc='Total mass')
	cm  =Array(np.array([0.0, 0.0, 0.0]),iotype='out', desc='COM [x,y,z]')
	
	I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
	gen_eff=Float(iotype='out', desc='Generator efficiency')
	Overall_eff=Float(iotype='out', desc='Overall drivetrain efficiency')
	R_s=Float(iotype='out', desc='Stator resistance')
	L_sm=Float( iotype='out', desc='mutual inductance')
	R_R=Float( iotype='out', desc='Rotor resistance')
	L_r=Float( iotype='out', desc='Rotor impedance')
	L_s=Float( iotype='out', desc='Stator synchronising inductance')
	Cu=Float(iotype='out', desc='Copper mass')
	Iron=Float(iotype='out', desc='Iron mass')
	

	
	def execute(self):
		r_s = self.r_s
		l_s = self.l_s
		h_s = self.h_s
		h_yr=self.h_yr
		h_ys=self.h_ys
		f=self.f
		q1=self.q1
		tau_p =self.tau_p
		B_g = self.B_g
		B_g1 = self.B_g1
		B_symax = self.B_symax
		B_rymax = self.B_rymax
		B_tsmax = self.B_tsmax
		B_trmax = self.B_trmax
		N_r=self.N_r
		I_0=self.I_0
		E_p=self.E_p
		J_s=self.J_s
		gen_eff =self.gen_eff
		K_rad=self.K_rad
		Active=self.Active
		Structure=self.Structure
		Cu=self.Cu
		N_s=self.N_s
		TC1=self.TC1
		TC2=self.TC2
		N_slots=self.N_slots
		Q_r=self.Q_r
		A_1=self.A_1
		J_s=self.J_s
		J_r=self.J_r
		K_rad=self.K_rad
		D_ratio=self.D_ratio
		Current_ratio=self.Current_ratio
		Slot_aspect_ratio1=self.Slot_aspect_ratio1
		Slot_aspect_ratio2=self.Slot_aspect_ratio2
		S_Nmax=self.S_Nmax
		p=self.p
		b_s=self.b_s
		A_Curcalc=self.A_Curcalc
		A_Cuscalc=self.A_Cuscalc
		b_trmin=self.b_trmin
		b_tr=self.b_tr
		b_r=self.b_r
		b_t=self.b_t
		I=self.I
		cm=self.cm
		Costs=self.Costs
		Losses=self.Losses
		Mass=self.Mass
		highSpeedSide_cm=self.highSpeedSide_cm
		highSpeedSide_length=self.highSpeedSide_length
		machine_rating=self.machine_rating
		Gearbox_efficiency=self.Gearbox_efficiency
	
		n_nom=self.n_nom
		I_s=self.I_s
		L_s=self.L_s
		L_r=self.L_r
		L_sm=self.L_sm
		R_R=self.R_R
		Iron=self.Iron
		
		K_Cu=self.K_Cu
		K_Fe=self.K_Fe
		K_Fes=self.K_Fes
		
		rho_Fe= self.rho_Fe
		rho_Copper=self.rho_Copper
		
		from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil, tan, atan
		import numpy as np
		from numpy import sign,abs
		
		g1     =9.81                # m/s^2 acceleration due to gravity
		sigma  =21.5e3                # shear stress
		ratio  =0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p) 
		mu_0   =pi*4e-7        # permeability of free space
		cofi   =0.9                 # power factor
		rho_Cu       =1.8*10**(-8)*1.4   # copper resisitivity
		h_sy0  =0
		#DFIG_h_sy   =0.04
		h_w    = 0.005
		m      =3
		b_s_tau_s=0.45
		b_r_tau_r=0.45
		self.q1=5
		K_rs     =1/(-1*self.S_Nmax)
		I_SN   = self.machine_rating/(sqrt(3)*3000)
		I_SN_r =I_SN/K_rs
		I_m    =0.3*I_SN_r
		I_RN_r =sqrt((I_SN_r**2)+(I_m**2))
		k_y1=sin(pi*0.5)*12/15
		q2=self.q1-1
		y_tau_p=12./15
		y_tau_pr=10./12
		k_y1=sin(pi*0.5*y_tau_p)
		k_q1=sin(pi/6)/(self.q1*sin(pi/(6*self.q1)))
		k_y2=sin(pi*0.5*y_tau_pr)
		k_q2=sin(pi/6)/(q2*sin(pi/(6*q2)))
		k_wd1=k_y1*k_q1
		k_wd2=k_q2*k_y2
		k_sfil =0.65								 # Slot fill factor
		P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
		P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T @50 Hz
		
		k_fes =0.9
		self.p=3
		freq=60
		n_1=self.n_nom/(1-self.S_Nmax)
		gear      =1  
		alpha_p=pi/2*.7
		dia=2*self.r_s             # air gap diameter
		g=(0.1+0.012*(self.machine_rating)**(1./3))*0.001  #in m
		self.K_rad=self.l_s/dia
		r_r=self.r_s-g             #rotor radius
		self.tau_p=(pi*dia/(2*self.p))
		#self.tau_p_act=(pi*dia/(2*self.p))  # number of pole pairs
		N_slot=2
		self.N_slots=N_slot*self.p*self.q1*m
		N_slots_pp=self.N_slots/(m*self.p*2)

		n  = self.N_slots/2*self.p/self.q1        #no of slots per pole per phase
		tau_s=self.tau_p/(m*self.q1)        #slot pitch
		self.b_s=b_s_tau_s*tau_s;       #slot width
		b_so=0.004;
		b_ro=0.004;
		self.b_t=tau_s-self.b_s              #tooth width


		self.Q_r=2*self.p*m*q2
		tau_r=pi*(dia-2*g)/self.Q_r
		self.b_r=b_r_tau_r*tau_r
		self.b_tr=tau_r-self.b_r
		mu_rs				=0.005
		mu_rr				=0.005
		W_s					=(self.b_s/mu_rs)*1e-3  #in m
		W_r					=(self.b_r/mu_rr)*1e-3  #in m
		self.Slot_aspect_ratio1=self.h_s/self.b_s
		self.Slot_aspect_ratio2=self.h_r/self.b_r
		gamma_s	= (2*W_s/g)**2/(5+2*W_s/g)
		K_Cs=(tau_s)/(tau_s-g*gamma_s*0.5)  #page 3-13
		gamma_r		= (2*W_r/g)**2/(5+2*W_r/g)
		K_Cr=(tau_r)/(tau_r-g*gamma_r*0.5)  #page 3-13
		K_C=K_Cs*K_Cr
		g_eff=K_C*g
	
	
		om_m=gear*2*pi*self.n_nom/60
		om_e=self.p*om_m
		K_s=0.3
		n_c1=2    #number of conductors per coil
		a1 =2    # number of parallel paths
		self.N_s=round(2*self.p*N_slots_pp*n_c1/a1)
		
		self.N_r=round(self.N_s*k_wd1*K_rs/k_wd2)
		n_c2=self.N_r/(self.Q_r/m)
		self.B_g1=mu_0*3*self.N_r*self.I_0*2**0.5*k_y2*k_q2/(pi*self.p*g_eff*(1+K_s))
		
		self.B_g=self.B_g1*K_C
		
		#W_1a=0.97*3000/(sqrt(3)*sqrt(2)*60*k_wd*B_g1*2*dia*self.tau_p) # chapter 14, 14.9 #Boldea
		self.h_ys = self.B_g*self.tau_p/(self.B_symax*pi)
		self.B_rymax = self.B_symax
		self.h_yr=self.h_ys
		d_se=dia+2*(self.h_ys+self.h_s+h_w)  # stator outer diameter
		self.D_ratio=d_se/dia
		skew_r =pi*self.p*1/self.Q_r
		self.f = self.n_nom*self.p/60
		if (2*self.r_s>2):
			K_fills=0.65
		else:
			K_fills=0.4
		beta_skew = tau_s/self.r_s
		k_wskew=sin(self.p*beta_skew/2)/(self.p*beta_skew/2)
		l_fs=2*(0.015+y_tau_p*self.tau_p/2/cos(40))+pi*(self.h_s)
		l_Cus = 2*self.N_s*(l_fs+self.l_s)/a1             #shortpitch
		A_s = self.b_s*(self.h_s-h_w)
		A_scalc=self.b_s*1000*(self.h_s*1000-h_w*1000)
		A_Cus = A_s*self.q1*self.p*K_fills/self.N_s
		self.A_Cuscalc = A_scalc*self.q1*self.p*K_fills/self.N_s
		R_s=l_Cus*rho_Cu/A_Cus
		tau_r_min=pi*(dia-2*(g+self.h_r))/self.Q_r
		self.b_trmin=tau_r_min-b_r_tau_r*tau_r_min
		self.B_trmax = self.B_g*tau_r/self.b_trmin
		K_01=1-0.033*(W_s**2/g/tau_s)
		sigma_ds=0.0042
		K_02=1-0.033*(W_r**2/g/tau_r)
		sigma_dr=0.0062
		L_ssigmas=(2*mu_0*self.l_s*n_c1**2*self.N_slots/m/a1**2)*((self.h_s-h_w)/(3*self.b_s)+h_w/b_so)  #slot leakage inductance
		L_ssigmaew=(2*mu_0*self.l_s*n_c1**2*self.N_slots/m/a1**2)*0.34*self.q1*(l_fs-0.64*self.tau_p*y_tau_p)/self.l_s                   #end winding leakage inductance
		L_ssigmag=(2*mu_0*self.l_s*n_c1**2*self.N_slots/m/a1**2)*(0.9*tau_s*self.q1*k_wd1*K_01*sigma_ds/g_eff)
		L_ssigma=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
		self.L_sm =6*mu_0*self.l_s*self.tau_p*(k_wd1*self.N_s)**2/(pi**2*(self.p)*g_eff*(1+K_s))
		self.L_s=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator  inductance
		l_fr=(0.015+y_tau_pr*tau_r/2/cos(40*pi/180))+pi*(self.h_r)
		L_rsl=(mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*((self.h_r-h_w)/(3*self.b_r)+h_w/b_ro)  #slot leakage inductance
		L_rel= (mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*0.34*q2*(l_fr-0.64*tau_r*y_tau_pr)/self.l_s                   #end winding leakage inductance                  #end winding leakage inductance
		L_rtl=(mu_0*self.l_s*(2*n_c2)**2*self.Q_r/m)*(0.9*tau_s*q2*k_wd2*K_02*sigma_dr/g_eff) # tooth tip leakage inductance
		self.L_r=(L_rsl+L_rtl+L_rel)/K_rs**2  # rotor leakage inductance
		sigma1=1-(self.L_sm**2/self.L_s/self.L_r)
		
		#Field winding
		k_fillr = 0.55
		diff=self.h_r-h_w
		A_Cur=k_fillr*self.p*q2*self.b_r*diff/self.N_r
		self.A_Curcalc=A_Cur*1e6
		L_cur=2*self.N_r*(l_fr+self.l_s)
		R_r=rho_Cu*L_cur/A_Cur
		self.R_R=R_r/(K_rs**2)
		om_s=(120*freq/self.p)*2*pi/60
		P_e=self.machine_rating/(1-self.S_Nmax)
		self.E_p=om_s*self.N_s*k_wd1*self.r_s*self.l_s*self.B_g1
		I_r=P_e/m/self.E_p         # rotor active current
		I_sm=self.E_p/(2*pi*freq*(self.L_s+self.L_sm)) # stator reactive current
		self.I_s=sqrt((I_r**2+I_sm**2))
		#I_s=P_e/3/self.E_p/0.8
		I_srated=self.machine_rating/3/K_rs/self.E_p
		self.J_s=self.I_s/(self.A_Cuscalc)
		self.J_r=I_r/(self.A_Curcalc)
		self.A_1=2*m*N_s*self.I_s/pi/(2*self.r_s)
		self.Current_ratio=self.I_0/I_srated
		
		V_Cuss=m*l_Cus*A_Cus
		V_Cusr=m*L_cur*A_Cur
		V_Fest=(self.l_s*pi*((self.r_s+self.h_s)**2-self.r_s**2)-(2*m*self.q1*self.p*self.b_s*self.h_s*self.l_s))
		V_Fesy=self.l_s*pi*((self.r_s+self.h_s+self.h_ys)**2-(self.r_s+self.h_s)**2)
		r_r=self.r_s-g
		V_Fert=pi*self.l_s*(r_r**2-(r_r-self.h_r)**2)-2*m*self.q1*self.p*self.b_r*self.h_r*self.l_s
		V_Fery=self.l_s*pi*((r_r-self.h_r)**2-(r_r-self.h_r-self.h_yr)**2)
		self.Cu=(V_Cuss+V_Cusr)*self.rho_Copper
		M_Fest=V_Fest*self.rho_Fe
		M_Fesy=V_Fesy*self.rho_Fe
		M_Fert=V_Fert*self.rho_Fe
		M_Fery=V_Fery*self.rho_Fe
		self.Iron=M_Fest+M_Fesy+M_Fert+M_Fery
		M_gen=(self.Cu)+(self.Iron)
		K_gen=self.Cu*self.K_Cu+(self.Iron)*self.K_Fe #%M_pm*K_pm;
		L_tot=self.l_s
		self.Structure=0.0002*M_gen**2+0.6457*M_gen+645.24
		self.Mass=M_gen+self.Structure
		self.B_tsmax=self.B_g*tau_s/(self.b_t)
		self.Costs=K_gen+self.K_Fes*self.Structure
		K_R=1.2
		
		# losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		P_Cuss=m*self.I_s**2*R_s*K_R
		P_Cusr=m*I_r**2*self.R_R
		P_Cusnom=P_Cuss+P_Cusr
		# B_tmax=B_pm*tau_s/b_t
		P_Hyys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
		P_Ftys=M_Fesy*(self.B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)
		P_Hyd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))
		P_Ftd=M_Fest*(self.B_tsmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)
		P_Hyyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0h*abs(self.S_Nmax)*om_e/(2*pi*60))
		P_Ftyr=M_Fery*(self.B_rymax/1.5)**2*(P_Fe0e*(abs(self.S_Nmax)*om_e/(2*pi*60))**2)
		P_Hydr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0h*abs(self.S_Nmax)*om_e/(2*pi*60))
		P_Ftdr=M_Fert*(self.B_trmax/1.5)**2*(P_Fe0e*(abs(self.S_Nmax)*om_e/(2*pi*60))**2)
		P_add=0.5*self.machine_rating/100
		P_Fesnom=P_Hyys+P_Ftys+P_Hyd+P_Ftd+P_Hyyr+P_Ftyr+P_Hydr+P_Ftdr
		delta_v=1
		p_b=3*delta_v*I_r
		
		self.Losses=P_Cusnom+P_Fesnom+p_b+P_add;
		self.gen_eff=(P_e-self.Losses)*100/P_e
		self.Overall_eff=self.gen_eff*self.Gearbox_efficiency
		self.J_s=self.I_s/self.A_Cuscalc
		S_GN=(self.machine_rating-self.S_Nmax*self.machine_rating)/self.gen_eff/0.01
		T_e=self.p *(self.machine_rating*1.002)/(2*pi*freq*(1-self.S_Nmax))
		self.TC1=T_e/(2*pi*sigma)
		self.TC2=self.r_s**2*self.l_s

		r_out=d_se*0.5
		self.I[0]   = (0.5*self.Mass*r_out**2)
		self.I[1]   = (0.25*self.Mass*r_out**2+(1/12)*self.Mass*self.l_s**2) 
		self.I[2]   = self.I[1]
		cm[0]  = self.highSpeedSide_cm[0] + self.highSpeedSide_length/2. + self.l_s/2.
		cm[1]  = self.highSpeedSide_cm[1]
		cm[2]  = self.highSpeedSide_cm[2]
		

		
class Drive_DFIG(Assembly):
	Eta_target=Float(iotype='in', desc='Target drivetrain efficiency')
	Gearbox_efficiency=Float(iotype='in', desc='Gearbox efficiency')
	highSpeedSide_cm = Array(np.array([0.0, 0.0, 0.0]),iotype='in', desc='High speed side CM')
	highSpeedSide_length = Float(iotype='in', desc='High speed side length')
	Objective_function=Str(iotype='in')
	Optimiser=Str(desc = 'Optimiser', iotype = 'in')
	L=Float(iotype='out')
	Mass=Float(iotype='out')
	Efficiency=Float(iotype='out')
	r_s=Float(iotype='out', desc='Optimised radius')
	l_s=Float(iotype='out', desc='Optimised generator length')
	I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
	cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' Center of mass [x, y,z]')
	
	DFIG_r_s = Float(iotype='in', desc='airgap radius r_s')
	DFIG_l_s = Float(iotype='in', desc='Stator core length l_s')
	DFIG_h_s = Float(iotype='in', desc='Stator slot height h_s')
	DFIG_h_r = Float(iotype='in', desc='Rotor slot height ')
	DFIG_S_Nmax =Float(iotype='in', desc='Slip ')
	DFIG_B_symax = Float(iotype='in', desc='Peak Yoke flux density B_ymax')
	DFIG_I_0= Float(iotype='in', desc='Rotor current at no-load')
	DFIG_P_rated=Float(iotype='in',desc='Rated power')
	DFIG_N_rated=Float(iotype='in',desc='Rated speed')
	
	K_Cu=Float( iotype='in', desc='Specific cost of copper')
	K_Fe=Float(iotype='in', desc='Specific cost of magnetic steel/iron')
	K_Fes=Float(iotype='in', desc='Specific cost of structural steel')
	
	rho_Fe=Float(iotype='in', desc='Steel density kg/m^3')
	rho_Copper=Float(iotype='in', desc='Copper density kg/m^3')
	
	def __init__(self,Optimiser='',Objective_function=''):
		
				super(Drive_DFIG,self).__init__()
				""" Creates a new Assembly containing DFIG and an optimizer"""
				self.add('DFIG',DFIG())
				self.connect('DFIG_r_s','DFIG.r_s')
				self.connect('DFIG_l_s','DFIG.l_s')
				self.connect('DFIG_h_s','DFIG.h_s')
				self.connect('DFIG_h_r','DFIG.h_r')
				self.connect('DFIG_B_symax','DFIG.B_symax')
				self.connect('DFIG_S_Nmax','DFIG.S_Nmax')
				self.connect('DFIG_I_0','DFIG.I_0')
				self.connect('DFIG_P_rated','DFIG.machine_rating')
				self.connect('DFIG_N_rated','DFIG.n_nom')
				self.connect('Gearbox_efficiency','DFIG.Gearbox_efficiency')
				self.connect('highSpeedSide_cm','DFIG.highSpeedSide_cm')
				self.connect('highSpeedSide_length','DFIG.highSpeedSide_length')
				self.connect('DFIG.Mass','Mass')
				self.connect('DFIG.Overall_eff','Efficiency')
				self.connect('DFIG.r_s','r_s')
				self.connect('DFIG.l_s','l_s')
				self.connect('DFIG.I','I')
				self.connect('DFIG.cm','cm')
				self.connect('K_Fe','DFIG.K_Fe')
				self.connect('K_Fes','DFIG.K_Fes')
				self.connect('K_Cu','DFIG.K_Cu')
				self.connect('rho_Fe','DFIG.rho_Fe')
				self.connect('rho_Copper','DFIG.rho_Copper')
				
				
				self.Optimiser=Optimiser
				self.Objective_function=Objective_function
				Opt1=globals()[self.Optimiser]
				self.add('driver',Opt1())
				Obj1='DFIG'+'.'+self.Objective_function
				self.driver.add_objective(Obj1)
				self.driver.design_vars=['DFIG_r_s','DFIG_l_s','DFIG_h_s','DFIG_h_r','DFIG_S_Nmax','DFIG_B_symax','DFIG_I_0']
				self.driver.add_parameter('DFIG_r_s', low=0.2, high=1)
				self.driver.add_parameter('DFIG_l_s', low=0.4, high=2)
				self.driver.add_parameter('DFIG_h_s', low=0.045, high=0.1)
				self.driver.add_parameter('DFIG_h_r', low=0.045, high=0.1)
				self.driver.add_parameter('DFIG_B_symax', low=1, high=2)
				self.driver.add_parameter('DFIG_S_Nmax', low=-0.3, high=-0.1)
				self.driver.add_parameter('DFIG_I_0', low=5, high=100)
				self.driver.iprint=1
				self.driver.add_constraint('DFIG.Overall_eff>=Eta_target')		  			      #constraint 1
				self.driver.add_constraint('DFIG.E_p>500.0')															  #constraint 2
				self.driver.add_constraint('DFIG.E_p<5000.0')															  #constraint 3
				self.driver.add_constraint('DFIG.TC1<DFIG.TC2')															#constraint 4
				self.driver.add_constraint('DFIG.B_g>=0.7')																	#constraint 5
				self.driver.add_constraint('DFIG.B_g<=1.2')																	#constraint 6
				self.driver.add_constraint('DFIG.B_rymax<2.')																#constraint 7
				self.driver.add_constraint('DFIG.B_trmax<2.')																#constraint 8
				self.driver.add_constraint('DFIG.B_tsmax<2.') 															#constraint 9
				self.driver.add_constraint('DFIG.A_1<60000')  															#constraint 10
				self.driver.add_constraint('DFIG.J_s<=6')											        			#constraint 11
				self.driver.add_constraint('DFIG.J_r<=6')  																	#constraint 12
				self.driver.add_constraint('DFIG.K_rad>=0.2')  					      			#constraint 13 #boldea Chapter 3
				self.driver.add_constraint('DFIG.K_rad<=1.5')						      			#constraint 14
				self.driver.add_constraint('DFIG.D_ratio>=1.37')  							      			#constraint 15 #boldea Chapter 3
				self.driver.add_constraint('DFIG.D_ratio<=1.4')  														#constraint 16
				self.driver.add_constraint('DFIG.Current_ratio>=0.1')												#constraint 17
				self.driver.add_constraint('DFIG.Current_ratio<=0.3')												#constraint 18
				self.driver.add_constraint('DFIG.Slot_aspect_ratio1>=4')										#constraint 19
				self.driver.add_constraint('DFIG.Slot_aspect_ratio1<=10')										#constraint 20
				
def optim_DFIG():
	opt_problem = Drive_DFIG('CONMINdriver','Costs')
	opt_problem.Eta_target=93
#	opt_problem.DFIG_P_rated=3e6
#	opt_problem.DFIG_N_rated=1200
#	opt_problem.Gearbox_efficiency=0.955
#	opt_problem.DFIG_r_s= 0.58 #meter
#	opt_problem.DFIG_l_s= 0.6 #meter
#	opt_problem.DFIG_h_s = 0.065 #meter
#	opt_problem.DFIG_h_r = 0.07 #meter
#	opt_problem.DFIG_I_0 = 24 #Ampere
#	opt_problem.DFIG_B_symax = 1.3 #Tesla
#	opt_problem.DFIG_S_Nmax = -0.2  #Tesla
	# initial design values for a DFIG designed for a 5MW turbine
	opt_problem.DFIG_P_rated=5e6
	opt_problem.DFIG_N_rated=1200
	opt_problem.Gearbox_efficiency=0.955
	opt_problem.DFIG_r_s= 0.65 #meter
	opt_problem.DFIG_l_s= 0.6 #meter
	opt_problem.DFIG_h_s = 0.1 #meter
	opt_problem.DFIG_h_r = 0.065 #meter
	opt_problem.DFIG_I_0 = 32 #Ampere
	opt_problem.DFIG_B_symax = 1.3 #Tesla
	opt_problem.DFIG_S_Nmax = -0.3  #Tesla
	
	# Specific costs
	opt_problem.K_Cu   =4.786                  # Unit cost of Copper $/kg
	opt_problem.K_Fe	= 0.556                    # Unit cost of Iron $/kg
	opt_problem.K_Fes =0.50139                   # specific cost of structure
	
	#Material properties
	opt_problem.rho_Fe = 7700                 #Steel density
	opt_problem.rho_Copper =8900                  # Kg/m3 copper density

	opt_problem.run()
	raw_data = {'Parameters': ['Rating','Objective function','Air gap diameter', "Stator length","Kra","Diameter ratio", "Pole pitch(tau_p)", " Number of Stator Slots","Stator slot height(h_s)","Slots/pole/phase","Stator slot width(b_s)", " Stator slot aspect ratio","Stator tooth width(b_t)", "Stator yoke height(h_ys)","Rotor slots", "Rotor yoke height(h_yr)", "Rotor slot height(h_r)", "Rotor slot width(b_r)"," Rotor Slot aspect ratio", "Rotor tooth width(b_t)", "Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Optimal Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading","Stator resistance", "Stator leakage inductance", "Excited magnetic inductance"," Rotor winding turns","Conductor cross-section","Magnetization current","I_mag/Is"," Rotor Current density","Rotor resitance", " Rotor leakage inductance", "Generator Efficiency","Overall drivetrain Efficiency","Iron mass","Copper mass","Structural Steel mass","Total Mass","Total Material Cost"],
		'Values': [opt_problem.DFIG.machine_rating/1e6,opt_problem.Objective_function,2*opt_problem.DFIG.r_s,opt_problem.DFIG.l_s,opt_problem.DFIG.K_rad,opt_problem.DFIG.D_ratio,opt_problem.DFIG.tau_p*1000,opt_problem.DFIG.N_slots,opt_problem.DFIG.h_s*1000,opt_problem.DFIG.q1,opt_problem.DFIG.b_s*1000,opt_problem.DFIG.Slot_aspect_ratio1,opt_problem.DFIG.b_t*1000,opt_problem.DFIG.h_ys*1000,opt_problem.DFIG.Q_r,opt_problem.DFIG.h_yr*1000,opt_problem.DFIG.h_r*1000,opt_problem.DFIG.b_r*1000,opt_problem.DFIG.Slot_aspect_ratio2,opt_problem.DFIG.b_tr*1000,opt_problem.DFIG.B_g,opt_problem.DFIG.B_g1,opt_problem.DFIG.B_symax,opt_problem.DFIG.B_rymax,opt_problem.DFIG.B_tsmax,opt_problem.DFIG.B_trmax,opt_problem.DFIG.p,opt_problem.DFIG.f,opt_problem.DFIG.E_p,opt_problem.DFIG.I_s,opt_problem.DFIG.S_Nmax,opt_problem.DFIG.N_s,opt_problem.DFIG.A_Cuscalc,opt_problem.DFIG.J_s,opt_problem.DFIG.A_1/1000,opt_problem.DFIG.R_s,opt_problem.DFIG.L_s,opt_problem.DFIG.L_sm,opt_problem.DFIG.N_r,opt_problem.DFIG.A_Curcalc,opt_problem.DFIG.I_0,opt_problem.DFIG.Current_ratio,opt_problem.DFIG.J_r,opt_problem.DFIG.R_R,opt_problem.DFIG.L_r,opt_problem.DFIG.gen_eff,opt_problem.DFIG.Overall_eff,opt_problem.DFIG.Iron/1000,opt_problem.DFIG.Cu/1000,opt_problem.DFIG.Structure/1000,opt_problem.DFIG.Mass/1000,opt_problem.DFIG.Costs/1000],
			'Limit': ['','','','','(0.2-1.5)','(1.37-1.4)','','','','','','(4-10)','','','','','','','(4-10)','','(0.7-1.2)','','2','2.1','2.1','2.1','','','(500-5000)','','(-0.002-0.3)','','','(3-6)','<60','','','','','','','(0.1-0.3)','(3-6)','','','',opt_problem.Eta_target,'','','','',''],
				'Units':['MW','','m','m','-','-','mm','-','mm','','mm','','mm','mm','-','mm','mm','mm','-','mm','T','T','T','T','T','T','-','Hz','V','A','','turns','mm^2','A/mm^2','kA/m','ohms','p.u','p.u','turns','mm^2','A','','A/mm^2','ohms','p.u','%','%','Tons','Tons','Tons','Tons','$1000']}
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print(df)
	df.to_excel('DFIG_'+str(opt_problem.DFIG_P_rated/1e6)+'MW.xlsx')

		
if __name__=="__main__":
	optim_DFIG()
	
	
	

 

#	
#	
