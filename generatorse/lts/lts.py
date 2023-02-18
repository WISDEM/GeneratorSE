import openmdao.api as om
from generatorse.common.cost import Generator_Cost
import generatorse.lts.magnetics_design as md
from generatorse.lts.femm_fea import FEMM_Geometry
from generatorse.lts.structural import LTS_Outer_Rotor_Structural


class LTS_Outer_Rotor_Opt(om.Group):

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
        ivcs.add_output("E_p_target", 0.0, units="V", desc="Target terminal voltage")
        ivcs.add_output("N_nom", 0.0, units="rpm", desc="input speed")
        ivcs.add_output("N_rated", 0.0, units="rpm", desc="rated speed")
        ivcs.add_output("D_a", 0.0, units="m", desc="Armature outer diameter")
        ivcs.add_output("delta_em", 0.0, units="m", desc="Field coil diameter")
        ivcs.add_output("h_s", 0.0, units="m", desc="Slot height")
        ivcs.add_output("p", 0, desc="Pole pairs")
        ivcs.add_output("h_yr", 0.0, units="m", desc="Rotor yoke height")
        ivcs.add_output("h_sc", 0.0, units="m", desc="SC coil height ")

        # ivcs.add_output("alpha_p", 0.0, desc="pole arc coefficient")
        ivcs.add_output("alpha", 0.0, units="deg", desc="Start angle of field coil")
        ivcs.add_output("dalpha", 0.0, desc="Field coil fraction of available space")
        ivcs.add_output("N_sc", 0.0, desc="Number of turns of SC field coil")
        ivcs.add_output("N_c", 0.0, desc="Number of turns of armature winding")
        ivcs.add_output("J_s", 0.0, units="A/mm/mm", desc="Armature current density")
        ivcs.add_output("l_s", 0.0, units="m", desc="Stator core length")


        ivcs.add_discrete_output("m", 6, desc="number of phases")
        ivcs.add_discrete_output("q", 2, desc="slots per pole")
        ivcs.add_output("b_s_tau_s", 0.45, desc="??")
        ivcs.add_output("conductor_area", 1.8 * 1.2e-6, desc="??")
        ivcs.add_output("Y", 0.0, desc="coil pitch")
        ivcs.add_output("K_h", 2.0, desc="??")
        ivcs.add_output("K_e", 0.5, desc="??")
        ivcs.add_output("load_margin", 0.0, desc="SC coil current loading margin %")

        ivcs.add_output("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        ivcs.add_output("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        ivcs.add_output("rho_NbTi", 0.0, units="kg/(m**3)", desc="SC conductor mass density ")
        ivcs.add_output("resisitivty_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        ivcs.add_output("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        ivcs.add_output("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        ivcs.add_output("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        ivcs.add_output("C_NbTi", 0.0, units="USD/kg", desc="Specific cost of superconductor")

        ivcs.add_output("U_b", 0.0, units="V", desc="brush voltage ")
        ivcs.add_output("hvac_mass_coeff", 0.025, units="kg/kW/m")
        ivcs.add_output("hvac_mass_cost_coeff", 124.0, units="USD/kg")
        ivcs.add_output("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        ivcs.add_output("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")

        # ivcs.add_output("r_strand", 0.0, units="mm", desc="radius of the SC wire strand")
        # ivcs.add_output("k_pf_sc", 0.0, units="mm", desc="packing factor for SC wires")
        # ivcs.add_output("J_c", 0.0, units="A/(mm*mm)", desc="SC critical current density")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys", md.LTS_active(), promotes=["*"])
        self.add_subsystem("geom", FEMM_Geometry(), promotes=["*"])
        self.add_subsystem("results", md.Results(), promotes=["*"])
        self.add_subsystem("struct", LTS_Outer_Rotor_Structural(), promotes=["*"])
        self.add_subsystem("cost", Generator_Cost(), promotes=["*"])
        self.connect("Torque_actual", "T_e")
        self.connect("D_a", "D_generator")
