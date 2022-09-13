import numpy as np
import openmdao.api as om
import generatorse.ipm.magnetics_design as md
from generatorse.common.cost import Generator_Cost
from generatorse.ipm.femm_fea import FEMM_Geometry
from generatorse.ipm.structural import PMSG_Outer_Rotor_Structural


class PMSG_Outer_rotor_Opt(om.Group):
    def initialize(self):
        self.options.declare("debug_prints", default=False)
        
    def setup(self):

        ivcs = om.IndepVarComp()

        ivcs.add_output("P_rated", 0.0, units="W", desc="Rated Power")
        ivcs.add_output("T_rated", 0.0, units="N*m", desc="Torque")
        ivcs.add_output("E_p_target", 0.0, units="V", desc="Target voltage")
        ivcs.add_output("N_nom", 0.0, units="rpm", desc="rated speed")
        ivcs.add_output("D_a", 0.0, units="m", desc="Stator outer diameter")
        ivcs.add_output("l_s", 0.0, units="m", desc="Core length")
        ivcs.add_output("h_t", 0.0, units="m", desc="tooth height")
        ivcs.add_output("b", 0.0, desc="Slot pole combination")
        ivcs.add_output("c", 0.0, desc="Slot pole combination")
        ivcs.add_output("h_s1", 0.010, desc="Slot Opening height")
        ivcs.add_output("h_s2", 0.010, desc="Wedge Opening height")
        ivcs.add_output("b_t", 0.0, units="m", desc="tooth width ")
        ivcs.add_output("pp", 0.0, desc="Pole pairs")
        ivcs.add_output("g", 0.0, units="m", desc="air gap length")
        ivcs.add_output("N_c", 0.0, desc="Turns")
        ivcs.add_output("J_s", 0.0, units="A/mm**2", desc="Turns")
        ivcs.add_output("d_mag", 0.0, units="m", desc="nagnet distance from rotor inner radius")
        ivcs.add_output("h_m", 0.0, units="m", desc="magnet height")
        ivcs.add_output('magnet_l_pc', 0.0, desc='Length of magnet divided by max magnet length (slot length)' )
        ivcs.add_output("h_yr", 0.0, units="m", desc="Rotor yoke height")
        ivcs.add_output("h_ys", 0.0, units="m", desc="Stator yoke height")
        ivcs.add_output("rho_Fe", 0.0, units="kg/m**3", desc="Electrical steel density")
        ivcs.add_output("h_ew", 0.25, desc="??")
        ivcs.add_output("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")
        ivcs.add_output("k_sfil", 0.65, desc="slot fill factor")
        ivcs.add_output("P_Fe0h", 4.0, desc="specific hysteresis losses W/kg @ 1.5 T")
        ivcs.add_output("P_Fe0e", 1.0, desc="specific eddy losses W/kg @ 1.5 T")

        ivcs.add_discrete_output("m", 3, desc="number of phases")
        ivcs.add_discrete_output("q", 2, desc="slots per pole")

        ivcs.add_output("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        ivcs.add_output("rho_PM", 0.0, units="kg/(m**3)", desc="magnet mass density ")
        ivcs.add_output("resistivity_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        ivcs.add_output("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        ivcs.add_output("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        ivcs.add_output("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        ivcs.add_output("C_PM", 0.0, units="USD/kg", desc="Specific cost of Magnet")

        ivcs.add_output("hvac_mass_coeff", 0.025, units="kg/kW/m")
        ivcs.add_output("hvac_mass_cost_coeff", 124.0, units="USD/kg")
        ivcs.add_output("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        ivcs.add_output("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("geom", FEMM_Geometry(debug_prints = self.options['debug_prints']), promotes=["*"])
        self.add_subsystem("results", md.Results(debug_prints = self.options['debug_prints']), promotes=["*"])
        self.add_subsystem("struct", PMSG_Outer_Rotor_Structural(), promotes=["*"])
        self.add_subsystem("cost", Generator_Cost(), promotes=["*"])
        self.connect("D_a", "D_generator")
