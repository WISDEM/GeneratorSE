
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
     	 self.add('PMDD_disc', PMDD_disc())
  		 self.add('PMDDOptimdiscrotor_5MW', PMDDOptimdiscrotor_5MW())
       self.add('DDSG_5MW', DDSG_5MW())
       self.add('SCIG_Optim5MW', SCIG_Optim5MW())
       self.add('DFIG_Optim5MW', DFIG_Optim5MW())


if __name__ == "__main__":
	opt_problem = GeneratorSE()
	import time
	tt = time.time()
	opt_problem.run()
	print "\n"



	

	
	
