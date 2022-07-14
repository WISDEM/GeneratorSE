"""LTS_outer_stator.py
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071 """

import numpy as np
import openmdao.api as om
from generatorse.common.struct_util import shell_constant, plate_constant


class PMSG_rotor_inactive(om.ExplicitComponent):

    """Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator."""

    def setup(self):

        self.add_input("R_sh", 0.0, units="m", desc="airgap radius ")
        self.add_input("E", 2e11, units="N/m**2", desc="Young's modulus of elasticity")
        # Assign values to universal constants
        self.add_input("g1", 9.806, units="m/s**2", desc="acceleration due to gravity")
        self.add_input("phi", 0, units="deg", desc="rotor tilt -90 degrees during transportation")
        self.add_input("v", 0.3, desc="Poisson's ratio")
        self.add_input("h_ew", 0.25, desc="??")
        self.add_input("y_sh", units="m", desc="Shaft deflection")
        self.add_input("theta_sh", 0.0, units="rad", desc="slope")

        #self.add_input("T_e", 0.0, units="N*m", desc="Rated torque ")
        self.add_input("r_g", 0.0, units="m", desc="air gap radius")
        self.add_input("h_m", 0.0, units="m", desc="Magnet height ")
        self.add_input("l_s", 0.0, units="m", desc="core length")
        self.add_input("Sigma_normal", 0.0, units="N/m**2", desc="Normal Stress")
        self.add_input("h_yr", 0.0, units="m", desc="Rotor yoke height ")

        self.add_input("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        self.add_input("y_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        self.add_input("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")

        # structural design variables
        self.add_input("t_r", 0.0, units="m", desc="Rotor disc thickness")
        self.add_input("h_sr", 0.0, units="m", desc="Yoke height ")

        self.add_input("K_rad", desc="Aspect ratio")

        # Material properties
        self.add_input("rho_Fes", 0.0, units="kg/(m**3)", desc="Structural Steel density")
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        self.add_input("mass_PM", 0.0, units="kg", desc="Magnet density ")

        self.add_output("structural_mass_rotor", 0.0, units="kg", desc="Rotor mass (kg)")
        self.add_output("u_ar", 0.0, units="m", desc="Radial deformation")
        self.add_output("y_ar", 0.0, units="m", desc="Axial deformation")
        self.add_output("twist_r", 0.0, units="deg", desc="torsional twist")

        self.add_output("u_allowable_r", 0.0, units="m", desc="Allowable Radial deflection")
        self.add_output("y_allowable_r", 0.0, units="m", desc="Allowable Radial deflection")
        self.add_input("Sigma_shear", 0.0, units="N/m**2", desc="Shear stress")
        self.add_input("r_outer_active", 0.0, units="m", desc="rotor outer radius ")
        self.add_output("D_outer", 0.0, units="m", desc="rotor outer diameter ")
        self.add_input("r_mag_center", 0.0, units="m", desc="rotor magnet radius ")

        self.add_output("con_uar", val=0.0, desc=" Radial deflection constraint-rotor")
        self.add_output("con_yar", val=0.0, desc=" Axial deflection constraint-rotor")

    def compute(self, inputs, outputs):

        # Radial deformation of rotor

        # Unpack inputs
        y_sh = inputs["y_sh"]
        theta_sh = inputs["theta_sh"]
        h_sr = inputs["h_sr"]
        t_r = inputs["t_r"]
        l_s = inputs["l_s"]
        h_ew = inputs["h_ew"]
        E = inputs["E"]
        v = inputs["v"]
        g1 = inputs["g1"]
        r_g = inputs["r_g"]
        r_outer_active = inputs["r_outer_active"]
        r_mag_center = inputs["r_mag_center"]
        rho_Fes = inputs["rho_Fes"]
        rho_Fe = inputs["rho_Fe"]
        #T_e = inputs["T_e"]
        R_sh = inputs["R_sh"]
        phi = inputs["phi"]
        Sigma_normal = inputs["Sigma_normal"]
        u_allow_pcent = inputs["u_allow_pcent"]
        y_allow_pcent = inputs["y_allow_pcent"]
        K_rad = inputs["K_rad"]
        mass_PM = inputs["mass_PM"]

        h_yr = r_outer_active - r_g
        outputs["D_outer"] = (r_outer_active + h_sr) * 2
        R = r_g + h_yr + h_sr * 0.5

        L_r = l_s + t_r + h_ew
        constants_x_0 = shell_constant(R, t_r, L_r, 0, v, E)
        constants_x_L = shell_constant(R, t_r, L_r, L_r, v, E)

        f_d_denom1 = R / (E * ((R) ** 2 - (R_sh) ** 2)) * ((1 - v) * R**2 + (1 + v) * (R_sh) ** 2)
        f_d_denom2 = (
            t_r
            / (2 * constants_x_0[0] * (constants_x_0[1]) ** 3)
            * (
                constants_x_0[2] / (2 * constants_x_0[3]) * constants_x_0[4]
                - constants_x_0[5] / constants_x_0[3] * constants_x_0[6]
                - 0.5 * constants_x_0[7]
            )
        )

        f = Sigma_normal * (R) ** 2 * t_r / (E * (h_yr + h_sr) * (f_d_denom1 + f_d_denom2))

        u_d = (
            f
            / (2 * constants_x_L[0] * (constants_x_L[1]) ** 3)
            * (
                (
                    constants_x_L[2] / (2 * constants_x_L[3]) * constants_x_L[4]
                    - constants_x_L[5] / constants_x_L[3] * constants_x_L[6]
                    - 0.5 * constants_x_L[7]
                )
            )
            + y_sh
        )

        outputs["u_ar"] = (Sigma_normal * (R) ** 2) / (E * (h_yr + h_sr)) - u_d

        outputs["u_ar"] = abs(outputs["u_ar"] + y_sh)

        outputs["u_allowable_r"] = 2 * r_g / 1000 * u_allow_pcent / 100

        ###################################################### Electromagnetic design#############################################
        # return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17
        # axial deformation of rotor
        W_back_iron = plate_constant(r_g + h_yr, R_sh, v, r_g + h_yr, t_r, E)
        W_ssteel = plate_constant(r_outer_active + h_sr, R_sh, v, r_outer_active + h_sr, t_r, E)
        W_mag = plate_constant(r_mag_center, R_sh, v, r_mag_center, t_r, E)

        W_ir = rho_Fe * g1 * np.sin((np.deg2rad(phi))) * (L_r - t_r) * (h_sr)
        #a_i = R
        y_ai1r = (
            -W_ir
            * (R) ** 4
            / (R_sh * W_back_iron[0])
            * (W_back_iron[1] * W_back_iron[4] / W_back_iron[3] - W_back_iron[2])
        )
        W_sr = rho_Fes * g1 * np.sin((np.deg2rad(phi))) * (L_r - t_r) * h_yr
        y_ai2r = (
            -W_sr * (r_g + h_yr) ** 4 / (R_sh * W_ssteel[0]) * (W_ssteel[1] * W_ssteel[4] / W_ssteel[3] - W_ssteel[2])
        )
        W_m = np.sin((np.deg2rad(phi))) * mass_PM / (2 * np.pi * r_mag_center)
        y_ai3r = -W_m * r_mag_center * 4 / (R_sh * W_mag[0]) * (W_mag[1] * W_mag[4] / W_mag[3] - W_mag[2])

        w_disc_r = rho_Fes * g1 * np.sin((np.deg2rad(phi))) * t_r

        a_ii = r_g
        r_oii = R_sh
        M_rb = (
            -w_disc_r
            * a_ii**2
            / W_ssteel[3]
            * (W_ssteel[4] * 0.5 / (a_ii * R_sh) * (a_ii**2 - r_oii**2) - W_ssteel[9])
        )
        Q_b = w_disc_r * 0.5 / R_sh * (a_ii**2 - r_oii**2)

        y_aiir = (
            M_rb * a_ii**2 / W_ssteel[0] * W_ssteel[1]
            + Q_b * a_ii**3 / W_ssteel[0] * W_ssteel[2]
            - w_disc_r * a_ii**4 / W_ssteel[0] * W_ssteel[7]
        )

        I = np.pi * 0.25 * (R**4 - (R_sh) ** 4)
        F_ecc = Sigma_normal * 2 * np.pi * K_rad * r_g**3
        M_ar = F_ecc * L_r * 0.5 * 0

        outputs["y_ar"] = (
            (y_ai1r + y_ai2r + y_ai3r) + y_aiir + (R * theta_sh) + M_ar * L_r**2 * 0 / (2 * E * I)
        )

        outputs["y_allowable_r"] = L_r / 100 * y_allow_pcent

        # Torsional deformation of rotor
        #J_dr = 0.5 * np.pi * (r_g**4 - R_sh**4)

        #J_cylr = 0.5 * np.pi * ((r_g + h_yr + h_sr) ** 4 - r_g**4)

        # outputs['twist_r']=180/np.pi*inputs['T_e']/G*(t_r/J_dr+(L_r-t_r)/J_cylr)

        outputs["structural_mass_rotor"] = (
            rho_Fes
            * np.pi
            * (((r_g) ** 2 - (R_sh) ** 2) * t_r + ((r_outer_active + h_sr) ** 2 - (r_outer_active) ** 2) * L_r)
        )

        outputs["con_uar"] = np.abs(outputs["u_ar"]) / outputs["u_allowable_r"]
        outputs["con_yar"] = np.abs(outputs["y_ar"]) / outputs["y_allowable_r"]


class PMSG_stator_inactive(om.ExplicitComponent):
    """Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator."""

    def setup(self):

        self.add_input("R_no", 0.0, units="m", desc="Nose outer radius ")
        self.add_input("D_a", 0.0, units="m", desc="Stator diameter ")
        self.add_input("y_bd", units="W", desc="Deflection of the bedplate")
        self.add_input("theta_bd", 0.0, units="m", desc="Slope at the bedplate")
        self.add_input("E", 2e11, units="N/m**2", desc="Young's modulus of elasticity")
        # Assign values to universal constants
        self.add_input("g1", 9.806, units="m/s**2", desc="acceleration due to gravity")
        self.add_input("phi", 0, units="deg", desc="rotor tilt -90 degrees during transportation")
        self.add_input("v", 0.3, desc="Poisson's ratio")
        self.add_input("h_ew", 0.25, desc="Poisson's ratio")

        #self.add_input("T_e", 0.0, units="N*m", desc="Rated torque ")
        self.add_input("r_g", 0.0, units="m", desc="air gap radius")
        self.add_input("h_t", 0.0, units="m", desc="tooth height")
        self.add_input("l_s", 0.0, units="m", desc="core length")
        self.add_input("Sigma_normal", 0.0, units="N/m**2", desc="Normal stress")
        self.add_input("h_ys", 0.0, units="m", desc="Stator yoke height ")

        # structural design variables

        self.add_input("t_s", 0.0, units="m", desc="Stator disc thickness")
        self.add_input("h_ss", 0.0, units="m", desc="Stator yoke height ")
        self.add_input("K_rad", desc="Aspect ratio")

        self.add_input("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        self.add_input("y_allow_pcent", 0.0, desc="Axial deflection as a percentage of air gap diameter")
        self.add_input("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")

        # Material properties
        self.add_input("rho_Fes", 0.0, units="kg/(m**3)", desc="Structural Steel density")
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        self.add_input("M_Fest", 0.0, units="kg", desc="Stator teeth mass ")
        self.add_input("mass_copper", 0.0, units="kg", desc="Copper mass ")

        self.add_output("structural_mass_stator", 0.0, units="kg", desc="Stator mass (kg)")

        self.add_output("u_as", 0.0, units="m", desc="Radial deformation")
        self.add_output("y_as", 0.0, units="m", desc="Axial deformation")
        self.add_output("twist_s", 0.0, units="deg", desc="Stator torsional twist")
        self.add_input("Sigma_shear", 0.0, units="N/m**2", desc="Shear stress")

        self.add_input("structural_mass_rotor", 0.0, units="kg", desc="Rotor mass (kg)")
        self.add_output("mass_structural", 0.0, units="kg", desc="Total structural mass (kg)")

        self.add_output(
            "u_allowable_s", 0.0, units="m", desc="Allowable Radial deflection as a percentage of air gap diameter"
        )
        self.add_output(
            "y_allowable_s", 0.0, units="m", desc="Allowable Axial deflection as a percentage of air gap diameter"
        )
        self.add_output("con_uas", val=0.0, desc=" Radial deflection constraint-stator")
        self.add_output("con_yas", val=0.0, desc=" Axial deflection constraint-stator")

    def compute(self, inputs, outputs):

        # Assign values to universal constants

        structural_mass_rotor = inputs["structural_mass_rotor"]
        R_no = inputs["R_no"]
        y_bd = inputs["y_bd"]
        theta_bd = inputs["theta_bd"]
        h_t = inputs["h_t"]
        h_ew = inputs["h_ew"]
        #T_e = inputs["T_e"]
        D_a = inputs["D_a"]
        r_g = inputs["r_g"]
        h_ys = inputs["h_ys"]
        h_ss = inputs["h_ss"]
        l_s = inputs["l_s"]
        t_s = inputs["t_s"]
        E = inputs["E"]
        v = inputs["v"]
        g1 = inputs["g1"]
        rho_Fe = inputs["rho_Fe"]
        rho_Fes = inputs["rho_Fes"]
        Copper = inputs["mass_copper"]
        M_Fest = inputs["M_Fest"]
        phi = inputs["phi"]
        Sigma_normal = inputs["Sigma_normal"]
        structural_mass_rotor = inputs["structural_mass_rotor"]
        K_rad = inputs["K_rad"]
        u_allow_pcent = inputs["u_allow_pcent"]
        y_allow_pcent = inputs["y_allow_pcent"]

        # Radial deformation of Stator
        L_s = l_s + t_s + h_ew
        R_is = D_a * 0.5 - h_t - (h_ys + h_ss)
        constants_x_0 = shell_constant(R_is, t_s, L_s, 0, v, E)
        constants_x_L = shell_constant(R_is, t_s, L_s, L_s, v, E)
        f_d_denom1 = R_is / (E * ((R_is) ** 2 - (R_no) ** 2)) * ((1 - v) * R_is**2 + (1 + v) * (R_no) ** 2)
        f_d_denom2 = (
            t_s
            / (2 * constants_x_0[0] * (constants_x_0[1]) ** 3)
            * (
                constants_x_0[2] / (2 * constants_x_0[3]) * constants_x_0[4]
                - constants_x_0[5] / constants_x_0[3] * constants_x_0[6]
                - 0.5 * constants_x_0[7]
            )
        )
        f = Sigma_normal * (R_is) ** 2 * t_s / (E * (h_ys + h_ss) * (f_d_denom1 + f_d_denom2))
        outputs["u_as"] = (
            (Sigma_normal * (R_is) ** 2) / (E * (h_ys + h_ss))
            - f
            / (2 * constants_x_L[0] * (constants_x_L[1]) ** 3)
            * (
                (
                    constants_x_L[2] / (2 * constants_x_L[3]) * constants_x_L[4]
                    - constants_x_L[5] / constants_x_L[3] * constants_x_L[6]
                    - 1 / 2 * constants_x_L[7]
                )
            )
            + y_bd
        )

        outputs["u_as"] = abs(outputs["u_as"] + y_bd)

        outputs["u_allowable_s"] = 2 * r_g / 1000 * u_allow_pcent / 100

        ###################################################### Electromagnetic design#############################################

        # axial deformation of stator
        W_back_iron = plate_constant(R_is + h_ss + h_ys, R_no, v, R_is + h_ss + h_ys, t_s, E)
        W_ssteel = plate_constant(R_is + h_ss, R_no, v, R_is + h_ss, t_s, E)
        W_active = plate_constant(D_a * 0.5 - h_t * 0.5, R_no, v, D_a * 0.5 - h_t * 0.5, t_s, E)

        W_is = rho_Fe * g1 * np.sin(np.deg2rad(phi)) * (L_s - t_s) * h_ys
        y_ai1s = (
            -W_is
            * (h_ys * 0.5 + h_ss + R_is) ** 4
            / (R_no * W_back_iron[0])
            * (W_back_iron[1] * W_back_iron[4] / W_back_iron[3] - W_back_iron[2])
        )

        W_ss = rho_Fes * g1 * np.sin(np.deg2rad(phi)) * (L_s - t_s) * h_ss
        y_ai2s = (
            -W_ss
            * (h_ss * 0.5 + R_is) ** 4
            / (R_no * W_ssteel[0])
            * (W_ssteel[1] * W_ssteel[4] / W_ssteel[3] - W_ssteel[2])
        )
        W_cu = np.sin((np.deg2rad(phi))) * (M_Fest + Copper) / (2 * np.pi * (D_a * 0.5 - h_t * 0.5))
        y_ai3s = (
            -W_cu
            * (D_a * 0.5 - h_t * 0.5) ** 4
            / (R_no * W_active[0])
            * (W_active[1] * W_active[4] / W_active[3] - W_active[2])
        )

        w_disc_s = rho_Fes * g1 * np.sin((np.deg2rad(phi))) * t_s

        a_ii = R_is
        r_oii = R_no
        M_rb = (
            -w_disc_s
            * a_ii**2
            / W_ssteel[3]
            * (W_ssteel[4] * 0.5 / (a_ii * R_no) * (a_ii**2 - r_oii**2) - W_ssteel[9])
        )
        Q_b = w_disc_s * 0.5 / R_no * (a_ii**2 - r_oii**2)

        y_aiis = (
            M_rb * a_ii**2 / W_ssteel[0] * W_ssteel[1]
            + Q_b * a_ii**3 / W_ssteel[0] * W_ssteel[2]
            - w_disc_s * a_ii**4 / W_ssteel[0] * W_ssteel[7]
        )

        I = np.pi * 0.25 * (R_is**4 - (R_no) ** 4)
        F_ecc = Sigma_normal * 2 * np.pi * K_rad * (D_a * 0.5) ** 2
        M_as = F_ecc * L_s * 0.5 * 0

        outputs["y_as"] = abs(
            y_ai1s + y_ai2s + y_ai3s + y_aiis + (D_a * 0.5 * theta_bd) + M_as * L_s**2 * 0 / (2 * E * I)
        )

        outputs["y_allowable_s"] = L_s * y_allow_pcent / 100

        # Torsional deformation of stator
        #J_ds = 0.5 * np.pi * (R_is**4 - R_no**4)

        #J_cyls = 0.5 * np.pi * ((D_a * 0.5 - h_t) ** 4 - R_is**4)

        # outputs['twist_s']=180.0/np.pi*inputs['T_rated']/G*(t_s/J_ds+(L_s-t_s)/J_cyls)

        outputs["structural_mass_stator"] = rho_Fes * (
            np.pi * ((R_is**2 - (R_no) ** 2) * t_s) + np.pi * ((R_is + h_ss) ** 2 - R_is**2) * L_s
        )

        outputs["mass_structural"] = outputs["structural_mass_stator"] + structural_mass_rotor

        outputs["con_uas"] = np.abs(outputs["u_as"]) / outputs["u_allowable_s"]
        outputs["con_yas"] = np.abs(outputs["y_as"]) / outputs["y_allowable_s"]


class PMSG_Outer_Rotor_Structural(om.Group):
    def setup(self):
        #        self.linear_solver = lbgs = om.LinearBlockJac() #om.LinearBlockGS()
        #        self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        #        nlbgs.options["maxiter"] = 3
        #        nlbgs.options["atol"] = 1e-2
        #        nlbgs.options["rtol"] = 1e-8
        #        nlbgs.options["iprint"] = 2

        ivcs = om.IndepVarComp()
        ivcs.add_output("y_sh", units="m", desc="Deflection at the shaft")
        ivcs.add_output("theta_sh", 0.0, units="rad", desc="slope of shaft deflection")
        ivcs.add_output("y_bd", units="W", desc="Deflection of the bedplate")
        ivcs.add_output("theta_bd", 0.0, units="m", desc="Slope at the bedplate")
        ivcs.add_output("R_sh", 0.0, units="m", desc=" Main shaft outer radius")
        ivcs.add_output("R_no", 0.0, units="m", desc=" Bedplate nose outer radius")
        ivcs.add_output("phi", 0.0, units="deg", desc=" Main shaft tilt angle")
        ivcs.add_output("h_sr", 0.0, units="m", desc="rotor yoke thickness")
        ivcs.add_output("h_ss", 0.0, units="m", desc="stator yoke thickness")
        ivcs.add_output("t_r", 0.0, units="m", desc="rotor disc thickness")
        ivcs.add_output("t_s", 0.0, units="m", desc="stator disc thickness")
        ivcs.add_output("E", 2e11, units="N/m**2", desc="Young's modulus of elasticity")
        ivcs.add_output("v", 0.3, desc="Poisson's ratio")
        ivcs.add_output("g1", 9.8106, units="m/s/s", desc="Acceleration due to gravity")
        ivcs.add_output("rho_Fes", 0.0, units="kg/m**3", desc="Structural steel mass density")
        ivcs.add_output("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        ivcs.add_output("y_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys1", PMSG_rotor_inactive(), promotes=["*"])
        self.add_subsystem("sys2", PMSG_stator_inactive(), promotes=["*"])


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = PMSG_Outer_Rotor_Structural()

    prob.driver = om.ScipyOptimizeDriver()  # pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SLSQP"  #'COBYLA'
    prob.driver.options["maxiter"] = 10
    # prob.driver.opt_settings['IPRINT'] = 4
    # prob.driver.opt_settings['ITRM'] = 3
    # prob.driver.opt_settings['ITMAX'] = 10
    # prob.driver.opt_settings['DELFUN'] = 1e-3
    # prob.driver.opt_settings['DABFUN'] = 1e-3
    # prob.driver.opt_settings['IFILE'] = 'CONMIN_LST.out'
    # prob.root.deriv_options['type']='fd'

    recorder = om.SqliteRecorder("log.sql")
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)
    prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("h_ss", lower=0.0250, upper=0.5, ref=0.3)
    prob.model.add_design_var("h_sr", lower=0.025, upper=0.6, ref=0.35)
    prob.model.add_design_var("t_r", lower=0.025, upper=0.5, ref=0.3)
    prob.model.add_design_var("t_s", lower=0.025, upper=0.5, ref=0.3)
    prob.model.add_objective("mass_structural")

    prob.model.add_constraint("con_uar", upper=1.0)
    prob.model.add_constraint("con_yar", upper=1.0)
    prob.model.add_constraint("con_uas", upper=1.0)
    prob.model.add_constraint("con_yas", upper=1.0)

    prob.model.approx_totals(method="fd")

    prob.setup()
    # --- Design Variables ---

    # Initial design variables for a PMSG designed for a 15MW turbine
    prob["Sigma_shear"] = 74.99692029e3
    prob["Sigma_normal"] = 378.45123826e3
    prob["T_e"] = 9e06
    prob["l_s"] = 1.44142189  # rev 1 9.94718e6
    prob["D_a"] = 7.74736313
    prob["r_g"] = 4
    prob["h_t"] = 0.1803019703
    prob["rho_Fes"] = 7850
    prob["rho_Fe"] = 7700
    prob["phi"] = 90.0
    prob["R_sh"] = 1.25
    prob["R_nor"] = 0.95
    prob["u_allow_pcent"] = 50
    prob["y_allow_pcent"] = 20
    prob["h_sr"] = 0.1254730934
    prob["h_ss"] = 0.050
    prob["t_r"] = 0.05
    prob["t_s"] = 0.1
    prob["y_bd"] = 0.00
    prob["theta_bd"] = 0.00
    prob["y_sh"] = 0.00
    prob["theta_sh"] = 0.00

    prob["mass_copper"] = 60e3
    prob["M_Fest"] = 4000

    prob.model.approx_totals(method="fd")

    prob.run_model()
    # prob.run_driver()

    # prob.model.list_outputs(values = True, hierarchical=True)

    raw_data = {
        "Parameters": [
            "Rotor disc thickness",
            "Rotor yoke thickness",
            "Stator disc thickness",
            "Stator yoke thickness",
            "Rotor radial deflection",
            "Rotor axial deflection",
            "Stator radial deflection",
            "Stator axial deflection",
            "Rotor structural mass",
            "Stator structural mass",
            "Total structural mass",
        ],
        "Values": [
            prob.get_val("t_rdisc", units="mm"),
            prob.get_val("h_sr", units="mm"),
            prob.get_val("t_sdisc", units="mm"),
            prob.get_val("h_ss", units="mm"),
            prob.get_val("u_ar", units="mm"),
            prob.get_val("y_ar", units="mm"),
            prob.get_val("u_as", units="mm"),
            prob.get_val("y_as", units="mm"),
            prob.get_val("structural_mass_rotor", units="t"),
            prob.get_val("structural_mass_stator", units="t"),
            prob.get_val("mass_structural", units="t"),
        ],
        "Limit": [
            "",
            "",
            "",
            "",
            prob.get_val("u_allowable_r", units="mm"),
            prob.get_val("y_allowable_r", units="mm"),
            prob.get_val("u_allowable_s", units="mm"),
            prob.get_val("y_allowable_s", units="mm"),
            "",
            "",
            "",
        ],
        "Units": [
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "tons",
            "tons",
            "tons",
        ],
    }
    # print(raw_data)
    # df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])

    # print(df)

    df.to_excel("Optimized_structure_LTSG_MW.xlsx")
