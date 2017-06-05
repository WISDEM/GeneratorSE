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

        self.driver.add_parameter('DFIG.DFIG_r_s', low=0.2, high=1)
        self.driver.add_parameter('DFIG.DFIG_l_s', low=0.4, high=2)
        self.driver.add_parameter('DFIG.DFIG_h_s', low=0.045, high=0.1)
        self.driver.add_parameter('DFIG.DFIG_h_r', low=0.045, high=0.1)
        self.driver.add_parameter('DFIG.DFIG_B_symax', low=1, high=2)
        self.driver.add_parameter('DFIG.DFIG_S_N', low=-0.3, high=-0.002)
        self.driver.add_parameter('DFIG.DFIG_I_0', low=5, high=100)
        self.driver.add_objective('-1*DFIG.Efficiency')

        self.DFIG.DFIG_r_s= 0.5  #meter
        self.DFIG.DFIG_l_s= 1.5 #meter
        self.DFIG.DFIG_h_s = 0.1 #meter
        self.DFIG.DFIG_h_r = 0.1 #meter
        self.DFIG.DFIG_I_0 = 14.5  # Ampere
        self.DFIG.DFIG_B_symax = 1. # Tesla
        self.DFIG.DFIG_S_N = -0.1 # Tesla

        # Dummy value
        self.DFIG.Generator_input = 20
    
top = DFIG_exec()
top.Targt_efficiency = .95
top.Gearbox_efficiency = .95
top.run()
print 'optimum is DFIG.DFIG_r_s %f '%top.DFIG.DFIG_r_s
print 'optimum is DFIG.DFIG_l_s %f '%top.DFIG.DFIG_l_s
print 'efficiency %f'%top.DFIG.Efficiency
