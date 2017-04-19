
"GeneratorSE"

from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import COBYLAdriver, CONMINdriver,NEWSUMTdriver,SLSQPdriver,Genetic
from PMDDOptim_5MW import PMDDOptim_5MW
from PMDDOptimdiscrotor_5MW import PMDDOptimdiscrotor_5MW
from DDSG_5MW import DDSG_5MW
from SCIG_Optim5MW import SCIG_Optim5MW
from DFIG_Optim5MW import DFIG_Optim5MW
from DFIG_exec import DFIG_exec
from SCIG_exec import SCIG_exec
from PMDD_exec import PMDD_exec
from DDSG_exec import DDSG_exec
from PMDDOptimdiscrotor_exec import PMDDOptimdiscrotor_exec
import numpy as np



from openmdao.lib.datatypes.api import Float
from termcolor import colored
A=input('Type of geNerator:(1)PMDD(2)DDSG(3)SCIG (4) DFIG:') 
if (A==1):
	  D=input('Type of structure:(1)Spoked arm stator and rotor (2) Spoked arm stator and disc rotor:') 
B=input('Enter your objective function:(1) minimise cost (2) minimise mass (3) maximise efficiency (4) minimise aspect ratio (l/d) (5) All of the above =')
C=input('Enter the number for your optimiser: (1) CONMINdriver (2)COBYLAdriver (3) SLSQPdriver (4) Genetic (5)NEWSUMTdriver =')                                      
#from pyopt_driver.pyopt_driver import pyOptDriver


class GeneratorSE(Assembly):
    """Unconstrained optimization of the PMDD Component."""
    def configure(self):
   
     if (C==1):   	
         #Create Optimizer instance
         self.add('driver',CONMINdriver())
         self.driver.itmax = 40
         self.driver.fdch = 0.01
         self.driver.fdchm = 0.01
         self.driver.ctlmin = 0.01
         self.driver.delfun = 0.001
         self.driver.iprint=4
         self.driver.conmin_diff = True
         
         
     elif (C==2):
     	   self.add('driver',COBYLAdriver())
     	   # COBYLA-specific Settings
     	   self.driver.rhobeg=1.0
     	   self.driver.rhoend = 1.0e-4
     	   self.driver.maxfun = 1000
     	   # Create PMDD component instances
     elif (C==3):
     		 self.add('driver',SLSQPdriver())
     		 # COBYLA-specific Settings
     		 self.driver.accuracy = 1.0e-6
     		 self.driver.maxiter = 50      
     elif (C==4):
         self.add('driver', Genetic())
         # Genetic-specific Settings
         self.driver.population_size = 90
         self.driver.crossover_rate = 0.9
         self.driver.mutation_rate = 0.02
         self.selection_method = 'rank'
     else:
     	   self.add('driver', NEWSUMTdriver())
     	   # NEWSUMT-specific Settings
     	   self.driver.itmax = 10 
     	   # Create PMDD/DDSSG component instances
     
     if (A==1):
     	  if (D==1):
     	       self.add('PMDDOptim_5MW', PMDDOptim_5MW())
     	       # Iteration Hierarchy
     	       self.driver.workflow.add('PMDDOptim_5MW')
     	       # Design Variable
     	       self.driver.add_parameter('PMDDOptim_5MW.r_s', low=0.5, high=9)
     	       self.driver.add_parameter('PMDDOptim_5MW.l_s', low=0.5, high=2.5)
     	       self.driver.add_parameter('PMDDOptim_5MW.h_s', low=0.04, high=0.1)
     	       self.driver.add_parameter('PMDDOptim_5MW.tau_p', low=0.04, high=0.1)
     	       self.driver.add_parameter('PMDDOptim_5MW.h_m', low=0.005, high=0.1)
     	       self.driver.add_parameter('PMDDOptim_5MW.n_r', low=5., high=15.)
     	       self.driver.add_parameter('PMDDOptim_5MW.t', low=0.045, high=0.25)
     	       self.driver.add_parameter('PMDDOptim_5MW.t_s', low=0.045, high=0.25)
     	       self.driver.add_parameter('PMDDOptim_5MW.b_r', low=0.1, high=1.5)
     	       self.driver.add_parameter('PMDDOptim_5MW.d_r', low=0.1, high=1.5)
     	       self.driver.add_parameter('PMDDOptim_5MW.t_wr', low=0.001, high=0.2)
     	       self.driver.add_parameter('PMDDOptim_5MW.n_s', low=5., high=15.)
     	       self.driver.add_parameter('PMDDOptim_5MW.b_st', low=0.1, high=1.5)
     	       self.driver.add_parameter('PMDDOptim_5MW.d_s', low=0.1, high=1.5)
     	       self.driver.add_parameter('PMDDOptim_5MW.t_ws', low=0.001, high=0.2)
     	       if (B==1):
     	       				self.driver.add_objective('PMDDOptim_5MW.TC')     	       		
     	       elif (B==2):
     	         			self.driver.add_objective('PMDDOptim_5MW.TM')
     	       elif (B==3):
     	         			self.driver.add_objective('PMDDOptim_5MW.TL')
     	       elif (B==4):
     	       	 			self.driver.add_objective('PMDDOptim_5MW.K_rad')
     	       else:
     	       	 			self.add('driver',SLSQPdriver())
     	       	 			self.driver.accuracy = 1.0e-6
     	       	 			self.driver.maxiter = 50
     	       	 			self.driver.add_objective('PMDDOptim_5MW.TC')
     	       	 			
     	       if (C==1) or (C==2) or (C==3) or (C==5):	 	
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.gen_eff>=93.5')									#1
     	         			self.driver.add_constraint('PMDDOptim_5MW.gen_eff<=96.5')									#2
     	         			self.driver.add_constraint('PMDDOptim_5MW.B_symax<2')										#3
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.B_rymax<2')										#4
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.B_tmax<2')									#5
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.E_p>=500')											#6
     	       				self.driver.add_constraint('PMDDOptim_5MW.E_p<15000')											#7
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.u_As<PMDDOptim_5MW.u_all_s')#8
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.z_As<PMDDOptim_5MW.z_all_s')#9
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.y_As<PMDDOptim_5MW.y_all')  #10
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.u_Ar<PMDDOptim_5MW.u_all_r')#11
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.z_A_r<PMDDOptim_5MW.z_all_r')#12
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.y_Ar<PMDDOptim_5MW.y_all') #13
     	       	 			#self.driver.add_constraint('PMDDOptim_5MW.t_w<PMDDOptim_5MW.d_s/2') #18
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.TC1<PMDDOptim_5MW.TC2')    #14
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.TC1<PMDDOptim_5MW.TC3')    #15
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.b_r<PMDDOptim_5MW.b_all_r')#16
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.b_st<PMDDOptim_5MW.b_all_s')#17
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.B_smax<PMDDOptim_5MW.B_g') #18
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.B_g>=0.7')  								#19                
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.B_g<=1.2') 								#20
     	       	 			#self.driver.add_constraint('PMDDOptim_5MW.B_pm1>=0.847645957')  								#21                
     	       	 			#self.driver.add_constraint('PMDDOptim_5MW.B_pm1<=1.453107356') 								#22
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.A_1<60000')									#21
     	       	 			#self.driver.add_constraint('PMDDOptim_5MW.J_s>=3') 										#22
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.J_s<=6') 										#23
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.A_Cuscalc>=5') 							#24
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.K_rad>0.2')									#25
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.K_rad<=0.27')								#26 
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.Slot_aspect_ratio>=4')					#27 			 
     	       	 			self.driver.add_constraint('PMDDOptim_5MW.Slot_aspect_ratio<=10')								#28	  
     	  if (D==2):
     	  		 self.add('PMDDOptimdiscrotor_5MW', PMDDOptimdiscrotor_5MW())
     	  		 # Iteration Hierarchy
     	  		 self.driver.workflow.add('PMDDOptimdiscrotor_5MW')
     	  		 # Design Variable
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.r_s', low=.9, high=6)  #Ul=6,l=0.9
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.l_s', low=0.5, high=2.5)   #UL =5 LL=0.2
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_s', low=0.04, high=0.1)
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.tau_p', low=0.04, high=0.1)
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_m', low=0.005, high=0.1)
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_yr', low=0.045, high=0.25) #UL=0.25
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_ys', low=0.045, high=0.25) #UL=0.25
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.t_d', low=0.1, high=0.25)  #UL=0.5
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.n_s', low=5., high=15.)
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.b_st', low=0.08, high=1.5) 
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.d_s', low=0.1, high=1.5)   #UL=0.6
     	  		 self.driver.add_parameter('PMDDOptimdiscrotor_5MW.t_ws', low=0.001, high=0.2)
     	  		 
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.r_s', low=3.66, high=3.66)  #Ul=6,l=0.9
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.l_s', low=1.43198648078, high=1.43198648078)   #UL =5 LL=0.2
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_s', low=0.07, high=0.07)
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.tau_p', low=0.07945, high=0.07945)
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_m', low=0.0227, high=0.0227)
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.B_ymax', low=0.7, high=1.2)
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_yr', low=0.045, high=0.25) #UL=0.25
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.h_ys', low=0.045, high=0.25) #UL=0.25
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.t_d', low=0.1, high=0.25)  #UL=0.5
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.n_s', low=5., high=15.)
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.b_st', low=0.08, high=1.5) 
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.d_s', low=0.1, high=1.5)   #UL=0.6
     	  		 #self.driver.add_parameter('PMDDOptimdiscrotor_5MW.t_ws', low=0.001, high=0.2)
     	  		 
     	  		 if (B==1):
     	  		 	      self.driver.add_objective('PMDDOptimdiscrotor_5MW.TC')
     	  		 elif (B==2):
     	  		 	      self.driver.add_objective('PMDDOptimdiscrotor_5MW.TM')
     	  		 elif (B==3):
     	  		 	      self.driver.add_objective('PMDDOptimdiscrotor_5MW.TL')
     	  		 elif (B==4):
     	  		 	      self.driver.add_objective('PMDDOptimdiscrotor_5MW.K_rad')
     	  		 else:
     	  		 	      self.add('driver',SLSQPdriver())
     	  		 	      self.driver.accuracy = 1.0e-6
     	  		 	      self.driver.maxiter = 50
     	  		 	      self.driver.add_objective('PMDDOptimdiscrotor_5MW.TC')
     	  		 	      
     	  		 if (C==1) or (C==2) or (C==3) or (C==5):
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.gen_eff>=93') 												#1
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.gen_eff<=93.5') 											#2
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.B_symax<=2')  												#3
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.B_rymax<=2')  												#4
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.B_tmax<=2')   												#5
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.E_p>500')    												  #6
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.E_p<=5000')   												#7
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.u_As<PMDDOptimdiscrotor_5MW.u_all_s') #8
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.z_As<PMDDOptimdiscrotor_5MW.z_all_s') #9
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.y_As<PMDDOptimdiscrotor_5MW.y_all')   #10
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.u_Ar<PMDDOptimdiscrotor_5MW.u_all_r') #11
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.y_Ar<PMDDOptimdiscrotor_5MW.y_all')   #12
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.TC1<PMDDOptimdiscrotor_5MW.TC2')      #13
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.TC1<PMDDOptimdiscrotor_5MW.TC3') 
     	  		 	 #self.driver.add_constraint('PMDDOptimdiscrotor_5MW.t_ws<PMDDOptimdiscrotor_5MW.d_s/2')     #14
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.b_st<PMDDOptimdiscrotor_5MW.b_all_s') #15
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.B_smax<PMDDOptimdiscrotor_5MW.B_g')   #16
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.B_g>=0.7')													  #17
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.B_g<=1.2') 													  #18
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.f>=10')																#19
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.f<=60')																#20
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.A_1<60000')														#21
     	  		 	 #self.driver.add_constraint('PMDDOptimdiscrotor_5MW.J_s>=3')															#
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.J_s<=6')															#22
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.A_Cuscalc>=5')												#23
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.K_rad>0.2')														#24
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.K_rad<=0.27')													#25
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.Slot_aspect_ratio>=4')								#26 			 
     	  		 	 self.driver.add_constraint('PMDDOptimdiscrotor_5MW.Slot_aspect_ratio<=10')								#27
     	  		 	      	       	 
     elif (A==2):
     	       self.add('DDSG_5MW', DDSG_5MW())
     	       # Iteration Hierarchy
     	       self.driver.workflow.add('DDSG_5MW')
     	       # Design Variable
     	       self.driver.add_parameter('DDSG_5MW.r_s', low=0.5, high=9)
     	       self.driver.add_parameter('DDSG_5MW.l_s', low=0.5, high=2.5)
     	       self.driver.add_parameter('DDSG_5MW.h_s', low=0.06, high=0.15)
     	       self.driver.add_parameter('DDSG_5MW.tau_p', low=0.04, high=.2)
     	       self.driver.add_parameter('DDSG_5MW.N_f', low=10, high=300)
     	       self.driver.add_parameter('DDSG_5MW.I_f', low=1, high=500)
     	       self.driver.add_parameter('DDSG_5MW.h_ys', low=0.01, high=0.25)
     	       self.driver.add_parameter('DDSG_5MW.h_yr', low=0.01, high=0.25)
     	       self.driver.add_parameter('DDSG_5MW.n_s', low=5., high=15.)
     	       self.driver.add_parameter('DDSG_5MW.b_st', low=0.1, high=1.5)
     	       self.driver.add_parameter('DDSG_5MW.d_s', low=0.1, high=1.5)   #UL=0.6
     	       self.driver.add_parameter('DDSG_5MW.t_ws', low=0.001, high=0.2)
     	       self.driver.add_parameter('DDSG_5MW.n_r', low=5., high=15.)
     	       self.driver.add_parameter('DDSG_5MW.b_r', low=0.1, high=1.5)
     	       self.driver.add_parameter('DDSG_5MW.d_r', low=0.1, high=1.5)   #UL=0.6
     	       self.driver.add_parameter('DDSG_5MW.t_wr', low=0.001, high=0.2)
     	       
     	       #self.driver.add_parameter('DDSG_5MW.r_s', low=4.256187666, high=4.256187666)
     	       #self.driver.add_parameter('DDSG_5MW.l_s', low=1.93, high=1.93)
     	       #self.driver.add_parameter('DDSG_5MW.h_s', low=0.06627, high=0.06627)
     	       #self.driver.add_parameter('DDSG_5MW.tau_p', low=0.2, high=0.2)
     	       #self.driver.add_parameter('DDSG_5MW.N_f', low=65, high=65)
     	       #self.driver.add_parameter('DDSG_5MW.I_f', low=151.42, high=151.42)
     	       #self.driver.add_parameter('DDSG_5MW.h_ys', low=0.21026, high=0.21026)
     	       #self.driver.add_parameter('DDSG_5MW.h_yr', low=0.21062, high=0.21062)
     	       #self.driver.add_parameter('DDSG_5MW.n_s', low=5., high=15.)
     	       #self.driver.add_parameter('DDSG_5MW.b_st', low=0.65213, high=0.65213)
     	       #self.driver.add_parameter('DDSG_5MW.d_s', low=1.5, high=1.5)   #UL=0.6
     	       #self.driver.add_parameter('DDSG_5MW.t_ws', low=0.02144, high=0.02144)
     	       #self.driver.add_parameter('DDSG_5MW.n_r', low=5., high=15.)
     	       #self.driver.add_parameter('DDSG_5MW.b_r', low=0.64942, high=0.64942)
     	       #self.driver.add_parameter('DDSG_5MW.d_r', low=1.5, high=1.5)   #UL=0.6
     	       #self.driver.add_parameter('DDSG_5MW.t_wr', low=0.06078, high=0.06078)
     	       #self.driver.add_parameter('DDSG_5MW.h_pc', low=0.049566, high=0.12)
     	       #self.driver.add_parameter('DDSG_5MW.h_ps', low=0.016522, high=0.02)
     	       #self.driver.add_parameter('DDSG_5MW.b_pc', low=0.066008959, high=0.08)
     	       
     	       if (B==1):
     	       				self.driver.add_objective('DDSG_5MW.TC')
     	       				  	       	 			
     	       elif (B==2):
     	         			self.driver.add_objective('DDSG_5MW.TM')
     	         			     	         			
     	       elif (B==3):
     	         			self.driver.add_objective('DDSG_5MW.TL')
     	       elif (B==4):
     	       	 			self.driver.add_objective('DDSG_5MW.K_rad')
     	       else:
     	       			self.add('driver',SLSQPdriver())
     	       			self.driver.accuracy = 1.0e-6
     	       			self.driver.maxiter = 50
     	       			self.driver.add_objective('DDSG_5MW.TM')
     	       if (C==1) or (C==2) or (C==3) or (C==5):
     	       	      self.driver.add_constraint('DDSG_5MW.gen_eff>=93') 						  #1
     	       	      self.driver.add_constraint('DDSG_5MW.gen_eff<=93.5')             #2
     	       	      self.driver.add_constraint('DDSG_5MW.u_As<DDSG_5MW.u_all_s')  #3
     	       	      self.driver.add_constraint('DDSG_5MW.z_As<DDSG_5MW.z_all_s')  #4
     	       	      self.driver.add_constraint('DDSG_5MW.y_As<DDSG_5MW.y_all')	  #5
     	       	      self.driver.add_constraint('DDSG_5MW.u_Ar<DDSG_5MW.u_all_r')  #6
     	       	      self.driver.add_constraint('DDSG_5MW.z_A_r<DDSG_5MW.z_all_r') #7
     	       	      self.driver.add_constraint('DDSG_5MW.y_Ar<DDSG_5MW.y_all')    #8
     	       	      self.driver.add_constraint('DDSG_5MW.b_r<DDSG_5MW.b_all_r')   #9
     	       	      self.driver.add_constraint('DDSG_5MW.b_st<DDSG_5MW.b_all_s')  #10
     	       	      self.driver.add_constraint('DDSG_5MW.TC1<DDSG_5MW.TC2')				#11
     	       	      self.driver.add_constraint('DDSG_5MW.TC1<DDSG_5MW.TC3')				#12
     	       	      self.driver.add_constraint('DDSG_5MW.E_s>=500')								#13
     	       	      self.driver.add_constraint('DDSG_5MW.E_s<=5000')								#14
     	       	      self.driver.add_constraint('DDSG_5MW.B_gfm>=0.617031')          #15
     	       	      self.driver.add_constraint('DDSG_5MW.B_gfm<=1.057768')						#16
     	       	      self.driver.add_constraint('DDSG_5MW.B_g>=0.7')                 #17
     	       	      self.driver.add_constraint('DDSG_5MW.B_g<=1.2')									#18
     	       	      self.driver.add_constraint('DDSG_5MW.B_symax<=2')								#19
     	       	      self.driver.add_constraint('DDSG_5MW.B_rymax<=2')								#20
     	       	      self.driver.add_constraint('DDSG_5MW.B_tmax<=2')								#21
     	       	      self.driver.add_constraint('DDSG_5MW.B_pc<=2')									#22
     	       	      self.driver.add_constraint('DDSG_5MW.A_1<60000')								#23
     	       	      self.driver.add_constraint('DDSG_5MW.J_s<=6')										#24
     	       	      self.driver.add_constraint('DDSG_5MW.J_f<=6')										#25
     	       	      self.driver.add_constraint('DDSG_5MW.A_Cuscalc>=5')							#26
     	       	      self.driver.add_constraint('DDSG_5MW.A_Cuscalc<=300')						#27
     	       	      self.driver.add_constraint('DDSG_5MW.A_Curcalc>=10')						#28
     	       	      self.driver.add_constraint('DDSG_5MW.A_Curcalc<=300')						#29
     	       	      self.driver.add_constraint('DDSG_5MW.K_rad>0.2')								#30
     	       	      self.driver.add_constraint('DDSG_5MW.K_rad<=0.27')							#31
     	       	      self.driver.add_constraint('DDSG_5MW.n_brushes<=6')							#32
     	       	      self.driver.add_constraint('DDSG_5MW.Slot_aspect_ratio>=4')      #33
     	       	      self.driver.add_constraint('DDSG_5MW.Slot_aspect_ratio<=10')     #34
     	       	      #self.driver.add_constraint('DDSG_5MW.Load_mmf_ratio>2')							
     	       	      #self.driver.add_constraint('DDSG_5MW.Load_mmf_ratio<2.5')							
     	       	      self.driver.add_constraint('DDSG_5MW.Power_ratio<2')							#35
     	       	      #self.driver.add_constraint('DDSG_5MW.l_Cur<3e4')		  
     	       	      #self.driver.add_constraint('DDSG_5MW.B_pc>1.3')
     	       	      #self.driver.add_constraint('DDSG_5MW.B_pc<=1.35')	  	      
     	       	 			
     elif (A==3):
     	       self.add('SCIG_Optim5MW', SCIG_Optim5MW())
     	       # Iteration Hierarchy
     	       self.driver.workflow.add('SCIG_Optim5MW')
     	       # Design Variable
     	       self.driver.add_parameter('SCIG_Optim5MW.r_s', low=0.2, high=1)
     	       self.driver.add_parameter('SCIG_Optim5MW.l_s', low=0.25, high=2.0)
     	       self.driver.add_parameter('SCIG_Optim5MW.h_s', low=0.04, high=0.1)
     	       self.driver.add_parameter('SCIG_Optim5MW.h_r', low=0.04, high=0.1)
     	       #self.driver.add_parameter('SCIG_Optim5MW.tau_p', low=0.1, high=0.4)
     	       self.driver.add_parameter('SCIG_Optim5MW.I_0', low=5, high=200)
     	       #self.driver.add_parameter('SCIG_Optim5MW.h_ys', low=0.04, high=0.2)
     	       #self.driver.add_parameter('SCIG_Optim5MW.S_N', low=-0.3, high=-0.002)
     	       self.driver.add_parameter('SCIG_Optim5MW.B_symax', low=1, high=2)

     	       if (B==1):
     	       				self.driver.add_objective('SCIG_Optim5MW.TC')
     	       elif (B==2):
     	         			self.driver.add_objective('SCIG_Optim5MW.TM')

     	       elif (B==3):
     	         			self.driver.add_objective('SCIG_Optim5MW.TL')
     	       elif (B==4):
     	       	 			self.driver.add_objective('SCIG_Optim5MW.lambda_ratio')
     	       else:
     	       	 			self.add('driver',SLSQPdriver())
     	       	 			self.driver.accuracy = 1.0e-6
     	       	 			self.driver.maxiter = 50
     	       	 			self.driver.add_objective('SCIG_Optim5MW.TC')
     	       if (C==1) or (C==2) or (C==3) or (C==5):  			
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.gen_eff>=93.0')							#1
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.TC1<SCIG_Optim5MW.TC2')		#2
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.B_g>=0.7')										#3
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.B_g<=1.2')										#4
     	       	 			#self.driver.add_constraint('SCIG_Optim5MW.B_symax<2.')								#5
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.B_rymax<2.')								#5
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.B_trmax<2.')								#6
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.B_tsmax<2.') 								#7
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.E_p>500') 										#8
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.E_p<5000') 									#9
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.A_1<=60000')   							#12
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.f>=60')											#13
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.f<=120')											#14
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.J_s<=6')											#15
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.J_r<=6')											#16
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.Lambda_ratio>=SCIG_Optim5MW.Lambda_ratio_LL')  					#17 #boldea Chapter 3
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.Lambda_ratio<=SCIG_Optim5MW.Lambda_ratio_UL')						#18 
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.D_ratio>=SCIG_Optim5MW.D_ratio_LL')  							#19 #boldea Chapter 3
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.D_ratio<=SCIG_Optim5MW.D_ratio_UL')  							#20
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.Slot_aspect_ratio1>=4')			#21
     	       	 			self.driver.add_constraint('SCIG_Optim5MW.Slot_aspect_ratio1<=10')			#22
     	       	 			#self.driver.add_constraint('SCIG_Optim5MW.r_r<=SCIG_Optim5MW.r_rmax')			#22
     	       	 			#self.driver.add_constraint('SCIG_Optim5MW.Slot_aspect_ratio2>=4')			#23
     	       	 			#self.driver.add_constraint('SCIG_Optim5MW.Slot_aspect_ratio2<=10')			#24
     	       	 			
     	
     else:
     	       self.add('DFIG_Optim5MW', DFIG_Optim5MW())
     	       # Iteration Hierarchy
     	       self.driver.workflow.add('DFIG_Optim5MW')
     	       # Design Variable
     	       self.driver.add_parameter('DFIG_Optim5MW.r_s', low=0.2, high=1)
     	       self.driver.add_parameter('DFIG_Optim5MW.l_s', low=0.4, high=2)
     	       self.driver.add_parameter('DFIG_Optim5MW.h_s', low=0.045, high=0.1)
     	       self.driver.add_parameter('DFIG_Optim5MW.h_r', low=0.045, high=0.1)
     	       #self.driver.add_parameter('DFIG_Optim5MW.h_ys', low=0.04, high=0.15)
     	       self.driver.add_parameter('DFIG_Optim5MW.B_symax', low=1, high=2)
     	       self.driver.add_parameter('DFIG_Optim5MW.S_N', low=-0.3, high=-0.002)
     	       self.driver.add_parameter('DFIG_Optim5MW.I_f', low=5, high=100)
     	       
     	       if (B==1):
     	       				self.driver.add_objective('DFIG_Optim5MW.TC')
     	       elif (B==2):
     	         			self.driver.add_objective('DFIG_Optim5MW.TM')
     	       elif (B==3):
     	         			self.driver.add_objective('DFIG_Optim5MW.TL')
     	       elif (B==4):
     	       	 			self.driver.add_objective('DFIG_Optim5MW.K_rad')
     	       else:
     	       	 			self.add('driver',SLSQPdriver())
     	       	 			self.driver.accuracy = 1.0e-6
     	       	 			self.driver.maxiter = 50
     	       	 			self.driver.add_objective('DFIG_Optim5MW.TC')
     	       if (C==1) or (C==2) or (C==3) or (C==5):       	 			
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.gen_eff>=93.0')								#1
     	       	 			#self.driver.add_constraint('DFIG_Optim5MW.gen_eff<=93.5')  						#2
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.E_p>=500.0')     						#2
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.E_p<=5000.0')								#3
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.TC1<DFIG_Optim5MW.TC2')		#4
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.B_g>=0.7')										#5
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.B_g<=1.2')										#6
     	       	 			#self.driver.add_constraint('DFIG_Optim5MW.B_symax<2.')								#7
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.B_rymax<2.')								#8
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.B_trmax<2.')								#9
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.B_tsmax<2.') 								#10
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.A_1<60000')  								#11
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.J_s<=6')											#14
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.J_r<=6')											#15
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.lambda_ratio>=0.2')  					#16 #boldea Chapter 3
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.lambda_ratio<=1.5')						#17 
     	       	 			#self.driver.add_constraint('DFIG_Optim5MW.tau_p>=DFIG_Optim5MW.tau_p_act')						#17 
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.D_ratio>=1.37')  							#17 #boldea Chapter 3
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.D_ratio<=1.4')  							#18
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.Current_ratio>=0.1')					#18s
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.Current_ratio<=0.3')					#19  
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.Slot_aspect_ratio1>=4')			#20
     	       	 			self.driver.add_constraint('DFIG_Optim5MW.Slot_aspect_ratio1<=10')			#21
     	       	 			#self.driver.add_constraint('DFIG_Optim5MW.Slot_aspect_ratio2<=4')			#22

if __name__ == "__main__":
	opt_problem = GeneratorSE()
	import time
	tt = time.time()
	opt_problem.run()
	print "\n"
if(A==1):
		if (D==1):
			print "Minimum found at (%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f,%f, %f, %f, %f, %f)" %(opt_problem.PMDDOptim_5MW.r_s,\
			                                                                             	opt_problem.PMDDOptim_5MW.l_s,\
			                                                                             	opt_problem.PMDDOptim_5MW.h_s,\
			                                                                             	opt_problem.PMDDOptim_5MW.tau_p,\
			                                                                             	opt_problem.PMDDOptim_5MW.B_g,\
			                                                                             	opt_problem.PMDDOptim_5MW.B_symax,\
			                                                                             	opt_problem.PMDDOptim_5MW.n_r,\
                                                                       					   	opt_problem.PMDDOptim_5MW.t,\
                                                                       					   	opt_problem.PMDDOptim_5MW.b_r,\
                                                                       					   	opt_problem.PMDDOptim_5MW.d_r,\
                                                                       					   	opt_problem.PMDDOptim_5MW.t_wr,\
                                                                       					   	opt_problem.PMDDOptim_5MW.n_s,\
                                                                       					   	opt_problem.PMDDOptim_5MW.b_st,\
                                                                       					   	opt_problem.PMDDOptim_5MW.d_s,\
                                                                       					   	opt_problem.PMDDOptim_5MW.t_ws,\
                                                                       					   	opt_problem.PMDDOptim_5MW.W_1a)
		elif(D==2):
			print "Minimum found at (%f, %f, %f, %f, %f,%f,%f, %f, %f, %f, %f,%f, %f, %f)" %(opt_problem.PMDDOptimdiscrotor_5MW.r_s,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.l_s,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.h_s,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.tau_p,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.B_g,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.B_symax,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.h_yr,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.t_d,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.n_s,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.t_s,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.b_st,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.d_s,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.t_ws,\
																																										opt_problem.PMDDOptimdiscrotor_5MW.W_1a)
elif(A==2):
  	print "Minimum found at (%f, %f, %f, %f, %f, %f)" %(opt_problem.DDSG_5MW.r_s,\
                                                        opt_problem.DDSG_5MW.l_s,\
                                                        opt_problem.DDSG_5MW.h_s,\
                                                        opt_problem.DDSG_5MW.tau_p,\
                                                        opt_problem.DDSG_5MW.N_f,\
                                                        opt_problem.DDSG_5MW.I_f)
elif(A==3):
 		print "Minimum found at (%f, %f, %f, %f, %f, %f)" %(opt_problem.SCIG_Optim5MW.r_s,\
                                                        opt_problem.SCIG_Optim5MW.l_s,\
                                                        opt_problem.SCIG_Optim5MW.h_s,\
                                                        opt_problem.SCIG_Optim5MW.tau_p,\
                                                        opt_problem.SCIG_Optim5MW.h_ys,\
                                                        opt_problem.SCIG_Optim5MW.I_0)                                            
else:
		print "Minimum found at (%f, %f, %f, %f, %f, %f, %f,%f)" %(opt_problem.DFIG_Optim5MW.r_s,\
                                                        opt_problem.DFIG_Optim5MW.l_s,\
                                                        opt_problem.DFIG_Optim5MW.h_s,\
                                                        opt_problem.DFIG_Optim5MW.h_r,\
                                                        opt_problem.DFIG_Optim5MW.h_ys,\
                                                        opt_problem.DFIG_Optim5MW.tau_p,\
                                                        opt_problem.DFIG_Optim5MW.I_f,\
                                                        opt_problem.DFIG_Optim5MW.S_N)                                             
                                                        
                                                        
                                                        

print "Elapsed time: ", time.time()-tt, "seconds"
import pandas
if(A==1):
 if(D==1):
		my_comp = PMDD_exec()
		my_comp.r_s = opt_problem.PMDDOptim_5MW.r_s
		my_comp.l_s = opt_problem.PMDDOptim_5MW.l_s
		my_comp.h_s = opt_problem.PMDDOptim_5MW.h_s
		my_comp.tau_p = opt_problem.PMDDOptim_5MW.tau_p
		my_comp.h_m = opt_problem.PMDDOptim_5MW.h_m
		my_comp.n_r = opt_problem.PMDDOptim_5MW.n_r
		my_comp.t = opt_problem.PMDDOptim_5MW.t
		my_comp.t_s = opt_problem.PMDDOptim_5MW.t_s
		my_comp.b_r = opt_problem.PMDDOptim_5MW.b_r
		my_comp.d_r = opt_problem.PMDDOptim_5MW.d_r
		my_comp.t_wr = opt_problem.PMDDOptim_5MW.t_wr
		my_comp.n_s = opt_problem.PMDDOptim_5MW.n_s
		my_comp.b_st = opt_problem.PMDDOptim_5MW.b_st
		my_comp.d_s = opt_problem.PMDDOptim_5MW.d_s
		my_comp.t_ws = opt_problem.PMDDOptim_5MW.t_ws
		
		my_comp.run()
 		#Dimensions=["Stator Arms", "Stator Axial arm dimension","Stator Circumferential arm dimension"," Stator arm Thickness" ,"Rotor Axial arm dimension","Rotor Circumferential arm dimension" ,"Rotor arm Thickness"," Stator Radial deflection", "Stator Axial deflection","Stator circum deflection"," Rotor Radial deflection", "Rotor Axial deflection","Rotor circum deflection", "Air gap diameter", "Stator length", "Pole pitch", "Stator slot height","Stator slotwidth","Stator tooth width", "Stator yoke height", "Rotor yoke height", "Magnet height", "Magnet width", "Peak air gap flux density","Peak stator yoke flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current", "Stator resistance","Synchronous inductance", "Current density ","Generator Efficiency ","Electrical Steel","Magnet mass","Copper mass","Structural Mass", "Total Mass","Stator Turns"]
 		#Units=["m","m","mm",",mm","mm","mm","mm","mm","mm","mm","T","T","1","Hz","V","A","p.u","p.u","A/mm^2","%","tons","Turns"]
 		#Data=[my_comp.n_s,my_comp.d_s*1000,my_comp.b_st*1000,my_comp.t_ws*1000,my_comp.d_r*1000,my_comp.b_r*1000,my_comp.t_wr*1000,my_comp.Stator_radial*1000,my_comp.Stator_axial*1000,my_comp.Stator_circum*1000,my_comp.Rotor_radial*1000,my_comp.Rotor_axial*1000,my_comp.Rotor_circum*1000,2*my_comp.r_s,my_comp.l_s,my_comp.tau_p*1000,my_comp.h_s*1000,my_comp.b_s*1000,my_comp.b_t*1000,my_comp.h_ys*1000,my_comp.h_yr*1000,my_comp.h_m*1000,my_comp.b_m*1000,my_comp.B_g,my_comp.B_ymax,my_comp.p,my_comp.f,my_comp.E_p,my_comp.I_s,my_comp.R_s,my_comp.L_s,my_comp.J_s,my_comp.gen_eff,my_comp.Electrical_Steel/1000,my_comp.mass_PM/1000,my_comp.M_Cus/1000,my_comp.Inactive/1000,my_comp.M_actual/1000 ,my_comp.W_1a]
 		#df=pandas.DataFrame(Data,Dimensions)
 		#print df
 		#print(df.to_csv('PMDD.csv',sep='\t',index='False'))
 		raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor arms','Rotor Axial arm dimension','Rotor Circumferential arm dimension' ,'Rotor arm Thickness',' Stator Radial deflection', 'Stator Axial deflection','Stator circum deflection',' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Air gap diameter','Overall Outer diameter', 'Stator length', 'l/d ratio','Slot_aspect_ratio','Pole pitch', 'Stator slot height','Stator slotwidth','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental','Peak stator yoke flux density','Peak rotor yoke flux density','Flux density above magnet','Maximum Stator flux density','Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage', 'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns','Conductor cross-section','Stator Current density ','Specific current loading','Generator Efficiency ','Iron mass','Magnet mass','Copper mass','Structural Mass', 'Total Mass', 'Stator Mass','Rotor Mass','Total Material Cost'],
			'Values': [my_comp.P_gennom/1000000,my_comp.n_s,my_comp.d_s*1000,my_comp.b_st*1000,my_comp.t_ws*1000,my_comp.n_r,my_comp.d_r*1000,my_comp.b_r*1000,my_comp.t_wr*1000,my_comp.Stator_radial*1000,my_comp.Stator_axial*1000,my_comp.Stator_circum*1000,my_comp.Rotor_radial*1000,my_comp.Rotor_axial*1000,my_comp.Rotor_circum*1000,2*my_comp.r_s,my_comp.R_out*2,my_comp.l_s,my_comp.K_rad,my_comp.Slot_aspect_ratio,my_comp.tau_p*1000,my_comp.h_s*1000,my_comp.b_s*1000,my_comp.b_t*1000,my_comp.t_s*1000,my_comp.t*1000,my_comp.h_m*1000,my_comp.b_m*1000,my_comp.B_g,my_comp.B_symax,my_comp.B_rymax,my_comp.B_pm1,my_comp.B_smax,my_comp.B_tmax,my_comp.p,my_comp.f,my_comp.E_p,my_comp.I_s,my_comp.R_s,my_comp.L_s,my_comp.N_slots,my_comp.W_1a,my_comp.A_Cuscalc,my_comp.J_s,my_comp.A_1/1000,my_comp.gen_eff,my_comp.Iron/1000,my_comp.mass_PM/1000,my_comp.M_Cus/1000,my_comp.Inactive/1000,my_comp.M_actual/1000,my_comp.Stator/1000,my_comp.Rotor/1000,my_comp.Cost/1000],
				'Limit': ['','','',my_comp.b_all_s*1000,'','','',my_comp.b_all_r*1000,'',my_comp.u_all_s*1000,my_comp.y_all*1000,my_comp.z_all_s*1000,my_comp.u_all_r*1000,my_comp.y_all*1000,my_comp.z_all_r*1000,'','','','(0.2-0.27)','(4-10)','','','','','','','','','','<2','<2','<2',my_comp.B_g,'','','','','>500','','','','','5','3-6','60','>93%','','','','','','','',''],
				'Units':['MW','unit','mm','mm','mm','mm','mm','','mm','mm','mm','mm','mm','mm','mm','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','A/mm^2','slots','turns','mm^2','kA/m','%','tons','tons','tons','tons','tons','ton','ton','k$']}
		#df=pandas.DataFrame(raw_data,Dimensions)
		df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
		print df
		if (B==1):
		 df.to_csv('PMDDOptim_5MW_KR.csv')
 		elif (B==2):
 		 df.to_csv('PMDDOptim_5MW_mass.csv')
 		elif (B==3):
 		 df.to_csv('PMDDOptim_5MW_Efficiency.csv')
 		
	
 else:
		my_comp = PMDDOptimdiscrotor_exec()
		my_comp.r_s = opt_problem.PMDDOptimdiscrotor_5MW.r_s
		my_comp.l_s = opt_problem.PMDDOptimdiscrotor_5MW.l_s
		my_comp.h_s = opt_problem.PMDDOptimdiscrotor_5MW.h_s
		my_comp.tau_p = opt_problem.PMDDOptimdiscrotor_5MW.tau_p
		#my_comp.B_g = opt_problem.PMDDOptimdiscrotor_5MW.B_g
		#my_comp.B_ymax = opt_problem.PMDDOptimdiscrotor_5MW.B_ymax
		my_comp.h_m = opt_problem.PMDDOptimdiscrotor_5MW.h_m
		my_comp.h_yr = opt_problem.PMDDOptimdiscrotor_5MW.h_yr
		my_comp.h_ys = opt_problem.PMDDOptimdiscrotor_5MW.h_ys
		my_comp.t_d = opt_problem.PMDDOptimdiscrotor_5MW.t_d
		my_comp.n_s = opt_problem.PMDDOptimdiscrotor_5MW.n_s
		my_comp.t_s = opt_problem.PMDDOptimdiscrotor_5MW.t_s
		my_comp.b_st = opt_problem.PMDDOptimdiscrotor_5MW.b_st
		my_comp.d_s = opt_problem.PMDDOptimdiscrotor_5MW.d_s
		my_comp.t_ws = opt_problem.PMDDOptimdiscrotor_5MW.t_ws
		
		my_comp.run()
		#Dimensions=["Stator Arms", "Stator Axial arm dimension","Stator Circumferential arm dimension"," Stator arm Thickness" ," Rotor disc thickness", " Rotor Radial deflection", "Rotor Axial deflection", "Stator Radial deflcetion"," Stator Axial deflection"," Stator Circumferential deflection","Air gap Diameter", "Stator length", "Pole pitch", "Stator slot height","Stator slot width","Stator tooth width", "Stator yoke height", "Rotor yoke height", "Magnet height", "Magnet width", "Peak air gap flux density","Peak stator yoke flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current", "Stator resistance", "Synchronous inductance", "Current density ","Generator Efficiency ", "Iron mass", " Magnet mass", "Copper mass","Inactive mass","Total Mass","Stator Turns","Actual Cost"]
		#Units=["m","m","mm",",mm","mm","mm","mm","mm","mm","mm","T","T","1","Hz","V","A","p.u","p.u","A/mm^2","%","tons","Turns"]
		#Data=[my_comp.n_s,my_comp.d_s*1000,my_comp.b_st*1000,my_comp.t_ws*1000,my_comp.t_d*1000,my_comp.Radial_delta_rotor*1000,my_comp.Axial_delta_rotor*1000,my_comp.Radial_delta_stator*1000,my_comp.Axial_delta_stator*1000,my_comp.Circum_delta_stator*1000,2*my_comp.r_s,my_comp.l_s,my_comp.tau_p*1000,my_comp.h_s*1000,my_comp.b_s*1000,my_comp.b_t*1000,my_comp.h_ys*1000,my_comp.h_yr*1000,my_comp.h_m*1000,my_comp.b_m*1000,my_comp.B_g,my_comp.B_ymax,my_comp.p,my_comp.f,my_comp.E_p,my_comp.I_s,my_comp.R_s,my_comp.L_s,my_comp.J_s,my_comp.gen_eff,my_comp.Iron/1000,my_comp.PM/1000,my_comp.Copper/1000,my_comp.Inactive/1000,my_comp.M_actual/1000,my_comp.W_1a,my_comp.Cost_actual/1000]
		#df=pandas.DataFrame(Data,Dimensions)
		#print df
		#print(df.to_csv('PMDD.csv',sep='\t',index='False'))
		
		raw_data = {'Parameters': ['Generator Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,' Rotor disc thickness', ' Rotor Radial deflection', 'Rotor Axial deflection', 'Stator Radial deflcetion',' Stator Axial deflection',' Stator Circumferential deflection','Air gap Diameter', 'Overall outer diameter','Stator length', 'Aspect ratio','Pole pitch', 'Stator slot height','Stator slot width','Slot aspect ratio','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Flux density above magnet','Peak stator yoke flux density','Peak rotor yoke flux density','Fundamental air gap flux density peak','Peak Stator flux density','Peak Tooth flux density','Pole pairs','Number of stator slots', 'Generator output frequency', 'Generator output phase voltage', 'Generator Output phase current', 'Stator resistance', 'Synchronous inductance', 'Stator Turns','Conductor Cross-section','Current density ','Specifc current loading','Generator Efficiency ', 'Iron mass', ' Magnet mass', 'Copper mass','Inactive mass','Total Mass','Total Cost'],
			'Values': [my_comp.P_gennom/1000000,my_comp.n_s,my_comp.d_s*1000,my_comp.b_st*1000,my_comp.t_ws*1000,my_comp.t_d*1000,my_comp.Radial_delta_rotor*1000,my_comp.Axial_delta_rotor*1000,my_comp.Radial_delta_stator*1000,my_comp.Axial_delta_stator*1000,my_comp.Circum_delta_stator*1000,2*my_comp.r_s,2*my_comp.R_out,my_comp.l_s,my_comp.K_rad,my_comp.tau_p*1000,my_comp.h_s*1000,my_comp.b_s*1000,my_comp.Slot_aspect_ratio,my_comp.b_t*1000,my_comp.h_ys*1000,my_comp.h_yr*1000,my_comp.h_m*1000,my_comp.b_m*1000,my_comp.B_pm,my_comp.B_symax,my_comp.B_rymax,my_comp.B_g,my_comp.B_smax,my_comp.B_tmax,my_comp.p,my_comp.N_slots,my_comp.f,my_comp.E_p,my_comp.I_s,my_comp.R_s,my_comp.L_s,my_comp.W_1a,my_comp.A_Cuscalc,my_comp.J_s,my_comp.A_1/1000,my_comp.gen_eff,my_comp.Iron/1000,my_comp.PM/1000,my_comp.Copper/1000,my_comp.Inactive/1000,my_comp.M_actual/1000,my_comp.Cost/1000],
				'Limit': ['','','',my_comp.b_all_s*1000,'','',my_comp.u_all_r*1000,my_comp.y_all*1000,my_comp.u_all_s*1000,my_comp.y_all*1000,my_comp.z_all*1000,'','','','(0.2-0.27)','','','','(4-10)','','','','','','1.2','<2','<2','1.2',my_comp.B_g,'<2','','','10-60','','','','','','>5','3-6','<60','>93','','','','','',''],
				'Units':['MW','unit','mm','mm','mm','mm','mm','mm','mm','mm','mm','m','m','m','','mm','mm','mm','','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Slots','Hz','V','A','ohm/phase','p.u','turns','mm^2','A/mm^2','kA/m','%','tons','tons','tons','tons','tons','$1000']}
		#df=pandas.DataFrame(raw_data,Dimensions)
		df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
		print df
		if (B==1):
		 df.to_csv('PMDDOptimdiscrotor_5MW_cost.csv')
		if (B==2):
		 df.to_csv('PMDDOptimdiscrotor_5MW_mass.csv')
		if (B==3):
		 df.to_csv('PMDDOptimdiscrotor_5MW_Efficiency.csv')
		
				
	#import matlab.engine
#eng = matlab.engine.start_matlab()
elif (A==2):
	my_comp = DDSG_exec()
	my_comp.r_s = opt_problem.DDSG_5MW.r_s
	my_comp.l_s = opt_problem.DDSG_5MW.l_s
	my_comp.h_s = opt_problem.DDSG_5MW.h_s
	my_comp.tau_p = opt_problem.DDSG_5MW.tau_p
	#my_comp.h_pc = opt_problem.DDSG_5MW.h_pc #
	#my_comp.h_ps = opt_problem.DDSG_5MW.h_ps #
	#my_comp.b_pc = opt_problem.DDSG_5MW.b_pc #
	my_comp.I_f= opt_problem.DDSG_5MW.I_f
	#my_comp.W_1a = opt_problem.DDSG_5MW.W_1a
	my_comp.N_f = opt_problem.DDSG_5MW.N_f
	my_comp.h_yr= opt_problem.DDSG_5MW.h_yr
	my_comp.h_ys= opt_problem.DDSG_5MW.h_ys
	my_comp.n_s = opt_problem.DDSG_5MW.n_s
	my_comp.b_st = opt_problem.DDSG_5MW.b_st
	my_comp.d_s = opt_problem.DDSG_5MW.d_s
	my_comp.t_ws = opt_problem.DDSG_5MW.t_ws
	my_comp.n_r = opt_problem.DDSG_5MW.n_r
	my_comp.b_r = opt_problem.DDSG_5MW.b_r
	my_comp.d_r = opt_problem.DDSG_5MW.d_r
	my_comp.t_wr = opt_problem.DDSG_5MW.t_wr
	
	
	my_comp.run()
	
	raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor Arms', 'Rotor Axial arm dimension','Rotor Circumferential arm dimension','Rotor Arm thickness', ' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Stator Radial deflection',' Stator Axial deflection',' Stator Circumferential deflection','Air gap diameter', 'Stator length','l/D ratio', 'Pole pitch', 'Stator slot height','Stator slot width','Slot aspect ratio','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Rotor pole height', 'Rotor pole width', 'Average no load flux density', 'Peak air gap flux density','Peak stator yoke flux density','Peak rotor yoke flux density','Stator tooth flux density','Rotor pole core flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage(rms value)', 'Generator Output phase current', 'Stator resistance', 'Synchronous inductance','Stator slots','Stator turns','Stator conductor cross-section','Stator Current density ','Specific current loading','Field turns','Conductor cross-section','Field Current','D.C Field resistance','MMF ratio at rated load(Rotor/Stator)','Excitation Power (% of Rated Power)','Number of brushes/polarity','Field Current density','Generator Efficiency', 'Iron mass', 'Copper mass','Inactive mass','Total Mass','Total Cost'],
		'Values': [my_comp.P_gennom/1e6,my_comp.n_s,my_comp.d_s*1000,my_comp.b_st*1000,my_comp.t_ws*1000,my_comp.n_r,my_comp.d_r*1000,my_comp.b_r*1000,my_comp.t_wr*1000,my_comp.Radial_delta_rotor*1000,my_comp.Axial_delta_rotor*1000,my_comp.Circum_delta_rotor*1000,my_comp.Radial_delta_stator*1000,my_comp.Axial_delta_stator*1000,my_comp.Circum_delta_stator*1000,2*my_comp.r_s,my_comp.l_s,my_comp.K_rad,my_comp.tau_p*1000,my_comp.h_s*1000,my_comp.b_s*1000,my_comp.Slot_aspect_ratio,my_comp.b_t*1000,my_comp.h_ys*1000,my_comp.h_yr*1000,my_comp.h_p*1000,my_comp.b_p*1000,my_comp.B_gfm,my_comp.B_g,my_comp.B_symax,my_comp.B_rymax,my_comp.B_tmax,my_comp.B_pc,my_comp.p,my_comp.f,my_comp.E_s,my_comp.I_s,my_comp.R_s,my_comp.L_m,my_comp.N_slots,my_comp.W_1a,my_comp.A_Cuscalc,my_comp.J_s,my_comp.A_1/1000,my_comp.N_f,my_comp.A_Curcalc,my_comp.I_f,my_comp.R_r,my_comp.Load_mmf_ratio,my_comp.Power_ratio,my_comp.n_brushes,my_comp.J_f,my_comp.gen_eff,my_comp.Iron/1000,my_comp.Copper/1000,my_comp.Inactive/1000,my_comp.M_actual/1000,my_comp.Cost/1000],
			'Limit': ['','','',my_comp.b_all_s*1000,'','','',my_comp.b_all_r*1000,'',my_comp.u_all_r*1000,my_comp.y_all*1000,my_comp.z_all_r*1000,my_comp.u_all_s*1000,my_comp.y_all*1000,my_comp.z_all_s*1000,'','','(0.2-0.27)','','','','(4-10)','','','','','','(0.62-1.05)','1.2','1.5','1.5','2','1.8','','(10-60)','','','','','','','','(3-6)','<60','','','','','(2-2.5)','<2%','','(3-6)','>93','','','','',''],
				'Units':['MW','unit','mm','mm','mm','unit','mm','mm','mm','mm','mm','mm','mm','mm','mm','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','om/phase','p.u','slots','turns','mm^2','A/mm^2','kA/m','turns','mm^2','A','ohm','%','%','brushes','A/mm^2','turns','%','tons','tons','tons','1000$']} 
	#df=pandas.DataFrame(raw_data,Dimensions)
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print df
	if (B==1):
		df.to_csv('DDSG_5MW_cost.csv')
	if (B==2):
		df.to_csv('DDSG_5MW_mass.csv')
	if (B==3):
		df.to_csv('DDSG_5MW_Efficiency.csv')
	else :
	  df.to_csv('DDSG_5MW_Aspect_ratio.csv')
	
	
	
	
elif (A==3):
	my_comp = SCIG_exec()
	my_comp.r_s = opt_problem.SCIG_Optim5MW.r_s
	my_comp.l_s = opt_problem.SCIG_Optim5MW.l_s
	my_comp.h_s = opt_problem.SCIG_Optim5MW.h_s
	my_comp.h_r = opt_problem.SCIG_Optim5MW.h_r
	#my_comp.tau_p = opt_problem.SCIG_Optim5MW.tau_p
	my_comp.h_ys = opt_problem.SCIG_Optim5MW.h_ys
	my_comp.I_0=opt_problem.SCIG_Optim5MW.I_0
	#my_comp.S_N=opt_problem.SCIG_Optim5MW.S_N
	my_comp.B_symax=opt_problem.SCIG_Optim5MW.B_symax
	my_comp.run()
	Tons=my_comp.M_actual/1000
	#p=my_comp.b_s
	print "Actual mass Optimised of SCIG_Optim5MW turbine at %f tons" % Tons
	raw_data = {'Parameters': ['Rating',"Air gap diameter", "Stator length","Lambda ratio","Diameter ratio", "Pole pitch(tau_p)", " Number of Stator Slots","Stator slot height(h_s)","Stator slot width(b_s)","Stator Slot aspect ratio", "Stator tooth width(b_t)", "Stator yoke height(h_ys)","Rotor slots", "Rotor slot height(h_r)", "Rotor slot width(b_r)","Rotor tooth width(b_tr)","Rotor yoke height(h_yr)","Rotor Slot_aspect_ratio","Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak Rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Optimal Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading","Stator resistance", "Excited magnetic inductance","Magnetization current","Conductor cross-section"," Rotor Current density","Rotor resitance", "Generator Efficiency","Copper mass","Iron Mass", "Structural mass","Total Mass","Total Material Cost"],
		           'Values': [my_comp.P_gennom/1e6,2*my_comp.r_s,my_comp.l_s,my_comp.Lambda_ratio,my_comp.D_ratio,my_comp.tau_p*1000,my_comp.N_slots,my_comp.h_s*1000,my_comp.b_s*1000,my_comp.Slot_aspect_ratio1,my_comp.b_t*1000,my_comp.h_ys*1000,my_comp.Q_r,my_comp.h_r*1000,my_comp.b_r*1000,my_comp.b_tr*1000,my_comp.h_yr*1000,my_comp.Slot_aspect_ratio2,my_comp.B_g,my_comp.B_g1,my_comp.B_symax,my_comp.B_rymax,my_comp.B_tsmax,my_comp.B_trmax,my_comp.p,my_comp.f,my_comp.E_p,my_comp.I_s,my_comp.S_N,my_comp.W_1a,my_comp.A_Cuscalc,my_comp.J_s,my_comp.A_1/1000,my_comp.R_s,my_comp.L_sm,my_comp.I_0,my_comp.A_bar*1e6,my_comp.J_r,my_comp.R_r,my_comp.gen_eff,my_comp.Copper/1000,my_comp.Iron/1000,my_comp.Inactive/1000,my_comp.M_actual/1000,my_comp.TC/1000],
		           	'Limit': ['','','',"("+str(my_comp.Lambda_ratio_LL)+"-"+str(my_comp.Lambda_ratio_UL)+")","("+str(my_comp.D_ratio_LL)+"-"+str(my_comp.D_ratio_UL)+")",'','','','','(4-10)','','','','','','','','(4-10)','(0.7-1.2)','','2','2','2','2','','(10-60)','(500-5000)','','(-30% to -0.2%)','','','(3-6)','<60','','','','','','','>93','','','','',''],
		           	'Units':['MW','m','m','-','-','mm','-','mm','mm','-','mm','mm','-','mm','mm','mm','mm','','T','T','T','T','T','T','-','Hz','V','A','%','turns','mm^2','A/mm^2','kA/m','ohms','p.u','A','mm^2','A/mm^2','ohms','%','Tons','Tons','Tons','Tons','$1000']} 
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print df
	if (B==1):
		df.to_csv('SCIG_Optim5MW_cost.csv')
	elif (B==2):
		df.to_csv('SCIG_Optim5MW_mass.csv')
	elif (B==3):
		df.to_csv('SCIG_Optim5MW_Efficiency.csv')
	else :
	  df.to_csv('SCIG_Optim5MW_Aspect_ratio.csv') 
	
else:
	my_comp = DFIG_exec()
	my_comp.r_s = opt_problem.DFIG_Optim5MW.r_s
	my_comp.l_s = opt_problem.DFIG_Optim5MW.l_s
	my_comp.h_s = opt_problem.DFIG_Optim5MW.h_s
	my_comp.h_r = opt_problem.DFIG_Optim5MW.h_r
	my_comp.B_symax=opt_problem.DFIG_Optim5MW.B_symax
	#my_comp.h_ys = opt_problem.DFIG_Optim5MW.h_ys
	my_comp.S_N = opt_problem.DFIG_Optim5MW.S_N
	my_comp.I_f = opt_problem.DFIG_Optim5MW.I_f
	
	my_comp.run()
	Tons=my_comp.M_actual/1000
	#p=my_comp.b_s
	print "Actual mass optimised at %f tons" % Tons
	raw_data = {'Parameters': ['Rating',"Air gap diameter", "Stator length","Lambda ratio","Diameter ratio", "Pole pitch(tau_p)", " Number of Stator Slots","Stator slot height(h_s)","Slots/pole/phase","Stator slot width(b_s)", " Stator slot aspect ratio","Stator tooth width(b_t)", "Stator yoke height(h_ys)","Rotor slots", "Rotor yoke height(h_yr)", "Rotor slot height(h_r)", "Rotor slot width(b_r)"," Rotor Slot aspect ratio", "Rotor tooth width(b_t)", "Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Optimal Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading","Stator resistance", "Stator leakage inductance", "Excited magnetic inductance"," Rotor winding turns","Conductor cross-section","Magnetization current","I_mag/Is"," Rotor Current density","Rotor resitance", " Rotor leakage inductance", "Generator Efficiency","Iron mass","Copper mass","Structural Steel mass","Total Mass","Total Material Cost"],
		           'Values': [my_comp.P_gennom/1e6,2*my_comp.r_s,my_comp.l_s,my_comp.lambda_ratio,my_comp.D_ratio,my_comp.tau_p*1000,my_comp.N_slots,my_comp.h_s*1000,my_comp.q1,my_comp.b_s*1000,my_comp.Slot_aspect_ratio1,my_comp.b_t*1000,my_comp.h_ys*1000,my_comp.Q_r,my_comp.h_yr*1000,my_comp.h_r*1000,my_comp.b_r*1000,my_comp.Slot_aspect_ratio2,my_comp.b_tr*1000,my_comp.B_g,my_comp.B_g1,my_comp.B_symax,my_comp.B_rymax,my_comp.B_tsmax,my_comp.B_trmax,my_comp.p,my_comp.f,my_comp.E_p,my_comp.I_s,my_comp.S_N,my_comp.W_1a,my_comp.A_Cuscalc,my_comp.J_s,my_comp.A_1/1000,my_comp.R_s,my_comp.L_s,my_comp.L_sm,my_comp.W_2,my_comp.A_Curcalc,my_comp.I_f,my_comp.Current_ratio,my_comp.J_r,my_comp.R_R,my_comp.L_r,my_comp.gen_eff,my_comp.Iron/1000,my_comp.Cu/1000,my_comp.Inactive/1000,my_comp.M_actual/1000,my_comp.TC/1000],
		           	'Limit': ['','','','(0.2-1.5)','(1.37-1.4)','','','','','','(4-10)','','','','','','','(4-10)','','(0.7-1.2)','','2','2.1','2.1','2.1','','','(500-5000)','','(-0.002-0.3)','','','(3-6)','<60','','','','','','','(0.1-0.3)','(3-6)','','','>93','','','','',''],
		           	'Units':['MW','m','m','-','-','mm','-','mm','','mm','','mm','mm','-','mm','mm','mm','-','mm','T','T','T','T','T','T','-','Hz','V','A','','turns','mm^2','A/mm^2','kA/m','ohms','p.u','p.u','turns','mm^2','A','','A/mm^2','ohms','p.u','%','Tons','Tons','Tons','Tons','$1000']} 
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	data= colored(df,'red',attrs=['reverse', 'blink'])
	print(data)
	if (B==1):
		df.to_csv('DFIG_Optim5MW_cost.csv')
	elif (B==2):
		df.to_csv('DFIG_Optim5MW_mass.csv')
	elif (B==3):
		df.to_csv('DFIG_Optim5MW_Efficiency.csv')
	else :
	  df.to_csv('DFIG_Optim5MW_Aspect_ratio.csv')
	


	

	
	
