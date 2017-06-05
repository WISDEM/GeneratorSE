from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import SLSQPdriver
from DFIG_new import Drive_DFIG

class DFIG_exec(Assembly):
    def configure(self):

        # suboptimization with conmin, optimizing Costs
        self.add('DFIG', Drive_DFIG('CONMINdriver','Costs'))
        self.add('driver',SLSQPdriver())
        self.driver.workflow.add('DFIG')
        self.driver.accuracy = 1.0e-6
        self.driver.maxiter = 50
        self.driver.add_parameter('DFIG.Target_Efficiency', low=1e-3, high=0.999)
        self.driver.add_parameter('DFIG.Gearbox_efficiency', low=1e-3, high=0.999)
        self.driver.add_objective('DFIG.Efficiency')

        self.DFIG.DFIG_r_s= 0.5  #meter
        self.DFIG.DFIG_l_s= 1.5 #meter
        self.DFIG.DFIG_h_s = 0.1 #meter
        self.DFIG.DFIG_h_r = 0.1 #meter
        self.DFIG.DFIG_I_0 = 14.5  # Ampere
        self.DFIG.DFIG_B_symax = 1. # Tesla
        self.DFIG.DFIG_S_N = -0.1 # Tesla

        # Dummy value
        self.DFIG.Generator_input = 20
    
DFIG_exec().run()
