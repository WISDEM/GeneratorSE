import numpy as np
import openmdao.api as om
import generatorse.ipm.magnetics_design as md
from generatorse.ipm.femm_fea import FEMM_Geometry
from generatorse.ipm.structural import PMSG_Outer_Rotor_Structural


class PMSG_Cost(om.ExplicitComponent):
    def setup(self):
        self.add_input("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        self.add_input("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        self.add_input("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        self.add_input("C_PM", 0.0, units="USD/kg", desc="Specific cost of Magnet")

        # Mass of each material type
        self.add_input("Copper", 0.0, units="kg", desc="Copper mass")
        self.add_input("Iron", 0.0, units="kg", desc="Iron mass")
        self.add_input("mass_PM", 0.0, units="kg", desc="Magnet mass")
        self.add_input("structural_mass", 0.0, units="kg", desc="Structural mass")

        self.add_input("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        self.add_input("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")

        # Outputs
        self.add_output("mass_total", 0.0, units="kg", desc="Structural mass")
        self.add_output("cost_total", 0.0, units="USD", desc="Total cost")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["mass_total"] = (
            inputs["Copper"] + inputs["Iron"] + inputs["mass_PM"] + inputs["structural_mass"] + inputs["mass_adder"]
        )

        outputs["cost_total"] = (
            inputs["Copper"] * inputs["C_Cu"]
            + inputs["Iron"] * inputs["C_Fe"]
            + inputs["mass_PM"] * inputs["C_PM"]
            + inputs["structural_mass"] * inputs["C_Fes"]
            + inputs["cost_adder"]
        )


class PMSG_Outer_rotor_Opt(om.Group):
    def setup(self):
        # self.linear_solver = lbgs = om.LinearBlockJac()  # om.LinearBlockGS()
        # self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        # nlbgs.options["maxiter"] = 3
        # nlbgs.options["atol"] = 1e-2
        # nlbgs.options["rtol"] = 1e-8
        # nlbgs.options["iprint"] = 2

        ivcs = om.IndepVarComp()

        ivcs.add_output("P_rated", 0.0, units="W", desc="Rated Power")
        ivcs.add_output("T_rated", 0.0, units="N*m", desc="Torque")
        ivcs.add_output("E_p_target", 0.0, units="V", desc="Target voltage")
        ivcs.add_output("N_nom", 0.0, units="rpm", desc="rated speed")
        ivcs.add_output("D_a", 0.0, units="m", desc="Stator outer diameter")
        ivcs.add_output("l_s", 0.0, units="m", desc="Core length")
        ivcs.add_output("h_t", 0.0, units="m", desc="tooth height")
        # ivcs.add_output("b_t", 0.0, units="m", desc="Stator core height")
        ivcs.add_output("b", 0.0, desc="Slot pole combination")
        ivcs.add_output("c", 0.0, desc="Slot pole combination")
        ivcs.add_output("h_s1", 0.010, desc="Slot Opening height")
        ivcs.add_output("h_s2", 0.010, desc="Wedge Opening height")
        ivcs.add_output("p", 0.0, desc="Pole pairs")
        ivcs.add_output("g", 0.0, units="m", desc="air gap length")
        ivcs.add_output("I_s", 0.0, units="A", desc="Stator current")
        ivcs.add_output("N_c", 0.0, desc="Turns")
        ivcs.add_output("J_s", 0.0, units="A/mm**2", desc="Turns")
        ivcs.add_output("d_mag", 0.0, units="m", desc="nagnet distance from rotor inner radius")
        # ivcs.add_output("d_sep", 0.0, units="m", desc="bridge separation distance")
        ivcs.add_output("h_m", 0.0, units="m", desc="magnet height")
        # ivcs.add_output("alpha_v", 0.0, units="deg", desc="V-angle")
        # ivcs.add_output("alpha_m", 0.5, desc="pole pair width")
        ivcs.add_output('magnet_l_pc', 0.0, desc='Length of magnet divided by max magnet length (slot length)' )
        ivcs.add_output("h_yr", 0.0, units="m", desc="Rotor yoke height")
        ivcs.add_output("h_ys", 0.0, units="m", desc="Stator yoke height")
        #        ivcs.add_output('t_r', 0.0,units='m',desc='Rotor disc thickness')
        #        ivcs.add_output('t_s', 0.0,units='m',desc='Stator disc thickness' )
        #        ivcs.add_output('h_ss',0.0,units='m',desc='Stator rim thickness')
        #        ivcs.add_output('h_sr', 0.0,units='m',desc='Rotor rim thickness')
        #ivcs.add_output("w_fe", 0.0, units="m", desc="distance from air gap radius ") #unused
        # ivcs.add_output("l_fe_ratio", 0.0, desc="bridge width ratio ")
        ivcs.add_output("m_sep", 0.0, units="m", desc="magnet distance from center ")
        ivcs.add_output("rho_Fe", 0.0, units="kg/m**3", desc="Electrical steel density")
        ivcs.add_output("h_ew", 0.25, desc="??")
        ivcs.add_output("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")
        ivcs.add_output("k_sfil", 0.65, desc="slot fill factor")
        ivcs.add_output("mu_0", 4 * np.pi / 1e7, desc="premeability of free space")
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

        ivcs.add_output("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        ivcs.add_output("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys", md.PMSG_active(), promotes=["*"])
        self.add_subsystem("geom", FEMM_Geometry(), promotes=["*"])
        self.add_subsystem("results", md.Results(), promotes=["*"])
        self.add_subsystem("struct", PMSG_Outer_Rotor_Structural(), promotes=["*"])
        self.add_subsystem("cost", PMSG_Cost(), promotes=["*"])
