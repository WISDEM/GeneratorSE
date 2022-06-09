import openmdao.api as om
import generatorse.ms_pmsg.magnetics_design as md
from generatorse.ms_pmsg.femm_fea import FEMM_Geometry
from generatorse.ms_pmsg.structural import PMSG_Inner_Rotor_Structural


class MS_PMSG_Cost(om.ExplicitComponent):
    def setup(self):
        self.add_input("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        self.add_input("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        self.add_input("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        self.add_input("C_PM", 0.0, units="USD/kg", desc="Specific cost of Magnet")

        # Mass of each material type
        self.add_input("Copper", 0.0, units="kg", desc="Copper mass")
        self.add_input("Iron", 0.0, units="kg", desc="Iron mass")
        self.add_input("mass_PM", 0.0, units="kg", desc="Magnet mass")
        self.add_input("Structural_mass", 0.0, units="kg", desc="Structural mass")

        self.add_input("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        self.add_input("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")
        
        # Outputs
        self.add_output("mass_total", 0.0, units="kg", desc="Structural mass")
        self.add_output("cost_total", 0.0, units="USD", desc="Total cost")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["mass_total"] = (inputs["Copper"] + inputs["Iron"] +
                                 inputs["mass_PM"] + inputs["Structural_mass"] +
                                 inputs["mass_adder"])

        outputs["cost_total"] = (inputs["Copper"] * inputs["C_Cu"] + 
                                 inputs["Iron"] * inputs["C_Fe"] + 
                                 inputs["mass_PM"] * inputs["C_PM"] + 
                                 inputs["Structural_mass"] * inputs["C_Fes"] +
                                 inputs["cost_adder"])


class PMSG_Inner_rotor_Opt(om.Group):

    def initialize(self):
        self.options.declare("magnetics_by_fea",default=True)

        # self.linear_solver = lbgs = om.LinearBlockJac()  # om.LinearBlockGS()
        # self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        # nlbgs.options["maxiter"] = 3
        # nlbgs.options["atol"] = 1e-2
        # nlbgs.options["rtol"] = 1e-8
        # nlbgs.options["iprint"] = 2
    
    def setup(self):

        ivcs = om.IndepVarComp()
        
        ivcs.add_discrete_output('Eta_target',0.0,desc='Target drivetrain efficiency')
        ivcs.add_output('P_rated', 0.0,units='W',desc='Rated Power')
        ivcs.add_output('T_rated', 0.0,units='N*m',desc='Torque')
        ivcs.add_output('E_p_target', 0.0,units='V',desc='Target voltage')
        ivcs.add_output('N_nom', 0.0,units='rpm',desc='rated speed')
        ivcs.add_output('r_g', 0.0,units='m',desc='Air-gap radius')
        ivcs.add_output('l_s', 0.0,units='m',desc='core length')
        ivcs.add_output('h_s', 0.0,units='m',desc='slot height')
        ivcs.add_output('p', 0.0,desc='Pole pairs')
        ivcs.add_output('g', 0.0,units='m',desc='air gap length')
        ivcs.add_output('I_s', 0.0,units='A',desc='Stator current')
        ivcs.add_output('h_m', 0.0,units='m',desc='magnet height')
        ivcs.add_output('N_c', 0.0,desc='Turns')

        ivcs.add_output('h_yr', 0.0,units='m',desc='Rotor yoke height'  )
        ivcs.add_output('h_ys', 0.0,units='m',desc='Stator yoke height')
        ivcs.add_output('h_sr', 0.0,units='m',desc='Rotor structural rim thickness')
        ivcs.add_output('h_ss', 0.0,units='m',desc='Stator structural rim thickness' )
        ivcs.add_output('n_r',0.0,desc='Rotor arms')
        ivcs.add_output('n_s',0.0,desc='stator arms')
        
        ivcs.add_output('t_wr', 0.0,units='m',desc='rotor arm thickness')
        ivcs.add_output('t_ws', 0.0,units='m',desc='stator arm thickness')
        ivcs.add_output('b_r', 0.0,units='m',desc='rotor arm circumferential dimension')
        ivcs.add_output('b_st', 0.0,units='m',desc='stator arm circumferential dimension')
        ivcs.add_output('d_r', 0.0,units='m',desc='rotor arm depth')
        ivcs.add_output('d_s', 0.0,units='m',desc='stator arm depth')
        ivcs.add_output('t_s', 0.0,units='m',desc='Stator disc thickness' )
       
        
        ivcs.add_output('rho_Fe', 0.0,units='kg/m**3',desc='Electrical steel density')
        ivcs.add_output("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        ivcs.add_output("rho_PM", 0.0, units="kg/(m**3)", desc="magnet mass density ")
        ivcs.add_output("resisitivty_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        ivcs.add_output("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        ivcs.add_output("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        ivcs.add_output("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        ivcs.add_output("C_PM", 0.0, units="USD/kg", desc="Specific cost of Magnet")
        
        
        
        ivcs.add_output("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        ivcs.add_output("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")
        
        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem('sys',md.PMSG_active(), promotes =['*'])
        
        if self.options["magnetics_by_fea"]:
            self.add_subsystem('geom',FEMM_Geometry(), promotes =['*'])
        else:
            self.add_subsystem('Results_by_analytical_model',Results_by_analytical_model(), promotes =['*'])

        self.add_subsystem("results", md.Results(), promotes=["*"])
        self.add_subsystem("struct", PMSG_Inner_Rotor_Structural(), promotes=["*"])
        self.add_subsystem("cost", PMSG_Cost(), promotes=["*"])
