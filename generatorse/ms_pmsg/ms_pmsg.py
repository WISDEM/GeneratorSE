import numpy as np
import openmdao.api as om
from generatorse.common.cost import Generator_Cost
import generatorse.ms_pmsg.magnetics_design as md
from generatorse.ms_pmsg.femm_fea import FEMM_Geometry
from generatorse.ms_pmsg.structural import PMSG_Inner_Rotor_Structural


class PMSG_Inner_rotor_Opt(om.Group):
    def initialize(self):
        self.options.declare("magnetics_by_fea", default=True)

    def setup(self):

        ivcs = om.IndepVarComp()

        ivcs.add_output("P_rated", 0.0, units="W", desc="Rated Power")
        ivcs.add_output("T_rated", 0.0, units="N*m", desc="Torque")
        ivcs.add_output("E_p_target", 0.0, units="V", desc="Target voltage")
        ivcs.add_output("N_nom", 0.0, units="rpm", desc="input speed")
        ivcs.add_output("N_rated", 0.0, units="rpm", desc="rated speed")
        ivcs.add_output("r_g", 0.0, units="m", desc="Air-gap radius")
        ivcs.add_output("l_s", 0.0, units="m", desc="core length")
        ivcs.add_output("h_s", 0.0, units="m", desc="slot height")
        ivcs.add_output("h_s1", 0.010, desc="Slot Opening height")
        ivcs.add_output("h_s2", 0.010, desc="Wedge Opening height")
        ivcs.add_output("p", 0.0, desc="Pole pairs")
        ivcs.add_output("g", 0.0, units="m", desc="air gap length")
        ivcs.add_output("I_s", 0.0, units="A", desc="Stator current")
        ivcs.add_output("h_m", 0.0, units="m", desc="magnet height")
        ivcs.add_output("N_c", 0.0, desc="Turns")
        ivcs.add_output("mu_0", np.pi * 4 * 1e-7, desc="permeability of free space")
        ivcs.add_output("mu_r", 0.0, desc="relative permeability ")
        ivcs.add_output("k_sfil", 0.65, desc="slot fill factor")
        ivcs.add_output("k_fes", 0.95, desc="Iron stacking factor")
        ivcs.add_output("m", 3, desc=" no of phases")
        ivcs.add_output("q1", 1, desc=" no of slots per pole per phase")
        ivcs.add_output("b_s_tau_s", 0.45, desc="ratio of Slot width to slot pitch ")
        ivcs.add_output("P_Fe0h", 4.0, desc="specific hysteresis losses W/kg @ 1.5 T")
        ivcs.add_output("P_Fe0e", 1.0, desc="specific eddy losses W/kg @ 1.5 T")
        ivcs.add_output("B_r", 0.0, units="T", desc="Tesla remnant flux density")

        ivcs.add_output("h_yr", 0.0, units="m", desc="Rotor yoke height")
        ivcs.add_output("h_ys", 0.0, units="m", desc="Stator yoke height")
        #ivcs.add_output("h_sr", 0.0, units="m", desc="Rotor structural rim thickness")
        #ivcs.add_output("h_ss", 0.0, units="m", desc="Stator structural rim thickness")
        #ivcs.add_output("n_r", 0.0, desc="Rotor arms")
        #ivcs.add_output("n_s", 0.0, desc="stator arms")

        #ivcs.add_output("t_wr", 0.0, units="m", desc="rotor arm thickness")
        #ivcs.add_output("t_ws", 0.0, units="m", desc="stator arm thickness")
        #ivcs.add_output("b_r", 0.0, units="m", desc="rotor arm circumferential dimension")
        #ivcs.add_output("b_st", 0.0, units="m", desc="stator arm circumferential dimension")
        #ivcs.add_output("d_r", 0.0, units="m", desc="rotor arm depth")
        #ivcs.add_output("d_s", 0.0, units="m", desc="stator arm depth")
        #ivcs.add_output("t_s", 0.0, units="m", desc="Stator disc thickness")
        ivcs.add_output("ratio", 0.0, desc="ratio of magnet width to pole pitch")

        ivcs.add_output("rho_Fe", 0.0, units="kg/m**3", desc="Electrical steel density")
        ivcs.add_output("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        ivcs.add_output("rho_PM", 0.0, units="kg/(m**3)", desc="magnet mass density ")
        ivcs.add_output("resisitivty_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        ivcs.add_output("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        ivcs.add_output("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        ivcs.add_output("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        ivcs.add_output("C_PM", 0.0, units="USD/kg", desc="Specific cost of Magnet")
        ivcs.add_output("H_c", 0.0, units="A/m", desc="coercivity")

        ivcs.add_output("hvac_mass_coeff", 0.025, units="kg/kW/m")
        ivcs.add_output("hvac_mass_cost_coeff", 124.0, units="USD/kg")
        ivcs.add_output("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        ivcs.add_output("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys", md.PMSG_active(), promotes=["*"])

        if self.options["magnetics_by_fea"]:
            self.add_subsystem("geom", FEMM_Geometry(), promotes=["*"])
        else:
            self.add_subsystem("Results_by_analytical_model", md.Results_by_analytical_model(), promotes=["*"])

        self.add_subsystem("results", md.Results(), promotes=["*"])
        self.add_subsystem("struct", PMSG_Inner_Rotor_Structural(), promotes=["*"])
        self.add_subsystem("cost", Generator_Cost(), promotes=["*"])
        self.connect("D_outer", "D_generator")

if __name__ == "__main__":
    femm_flag = True
    prob = om.Problem()
    prob.model = PMSG_Inner_rotor_Opt(magnetics_by_fea=femm_flag)
    prob.setup()
    prob.run_model()
    
    
