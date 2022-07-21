"""LTS_outer_stator.py
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071 """

import pandas as pd
import numpy as np
import openmdao.api as om
from generatorse.common.struct_util import structural_constraints


class PMSG_rotor_inactive(om.ExplicitComponent):
    def setup(self):

        self.add_output("u_ar", 0.0, units="m", desc="Rotor radial deflection")
        self.add_output("y_ar", 0.0, units="m", desc="Rotor axial deflection")
        self.add_output("z_ar", 0.0, units="m", desc="Rotor circumferential deflection")
        self.add_output("u_allowable_r", 0.0, units="m", desc="Allowable radial rotor")
        self.add_output("y_allowable_r", 0.0, units="m", desc="Allowable axial rotor")
        self.add_output("z_allowable_r", 0.0, units="m", desc="Allowable circum rotor")
        self.add_output("b_allowable_r", 0.0, units="m", desc="Allowable arm dimensions")
        
        self.add_input("u_allow_pcent", 0.0, desc="Allowable radial deflection percent")
        self.add_input("y_allow_pcent", 0.0, desc="Allowable axial deflection percent")
        self.add_input("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist ")

        self.add_input("n_r", 0.0, desc="number of arms n")
        self.add_input("b_r", 0.0, units="m", desc="arm width b_r")
        self.add_input("d_r", 0.0, units="m", desc="arm depth d_r")
        self.add_input("t_wr", 0.0, units="m", desc="arm depth thickness self.t_wr")
        self.add_input("h_sr", 0.0, units="m", desc="rotor rim thickness")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke thickness")
        self.add_input("l_s", 0.0, units="m", desc="stack length")
        self.add_input("phi", 0.0, units="deg", desc="tilt angle")

        self.add_input("r_g", 0.0, units="m", desc="air gap radius")
        self.add_input("h_m", 0.0, units="m", desc="magnet thickness")
        self.add_input("g", 0.0, units="m", desc="airgap length")
        self.add_input("R_sh", 0.0, units="m", desc="shaft radius")
        #self.add_input("B_g", 0.0, units="T", desc="Peak air gap flux density")
        self.add_input("E", 2.0e11, desc="Youngs modulus")
        self.add_input("g1", 9.806, desc="Acceleration due to gravity")
        self.add_input("rho_PM", 0.0, units="kg/m**3", desc="Magnet density kg/m^3")
        self.add_input("ratio", 0.0, desc="ratio of magnet width to pole pitch(bm/self.tau_p")
        self.add_input("Sigma_normal", 0.0, units="N/m**2", desc="Normal stress")
        self.add_input("Sigma_shear", 0.0, units="N/m**2", desc="Normal stress")
        self.add_input("rho_Fes", 0.0, units="kg/m**3", desc="Structural Steel density kg/m^3")
        self.add_input("rho_Fe", 0.0, units="kg/m**3", desc="Magnetic Steel density kg/m^3")
        self.add_output("mass_PM", 0.0, units="kg", desc="magnet mass kg")
        self.add_output("mass_Fe_rotor", 0.0, units="kg", desc="Iron mass in rotor")
        self.add_output("mass_structural_rotor", 0.0, units="kg", desc="Rotor structural mass kg")
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        r_g = inputs["r_g"]
        g = inputs["g"]
        h_m = inputs["h_m"]
        N_r = inputs["n_r"]
        b_r = inputs["b_r"]
        d_r = inputs["d_r"]
        t_wr = inputs["t_wr"]
        h_yr = inputs["h_yr"]
        h_sr = inputs["h_sr"]
        R_sh = inputs["R_sh"]
        l = inputs["l_s"]
        t = h_yr + h_sr
        u_allow_pcent = inputs["u_allow_pcent"]
        y_allow_pcent = inputs["y_allow_pcent"]
        E = inputs["E"]
        ratio = inputs["ratio"]
        g1 = inputs["g1"]
        rho_PM = inputs["rho_PM"]
        Sigma_normal = inputs["Sigma_normal"]
        Sigma_shear = inputs["Sigma_shear"]
        rho_Fes = inputs["rho_Fes"]
        rho_Fe = inputs["rho_Fe"]
        phi = inputs["phi"]
        z_allow_deg = inputs["z_allow_deg"]
        ###Deflection Calculations##

        # rotor structure calculations

        a_r = (b_r * d_r) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr))  # cross-sectional area of rotor arms
        A_r = l * t  # cross-sectional area of rotor cylinder
        #N_r = np.round(n_r)  # rotor arms
        theta_r = np.pi * 1 / N_r  # half angle between spokes
        I_r = l * t**3 / 12  # second moment of area of rotor cylinder
        # second moment of area of rotor arm
        I_arm_axi_r = (b_r * d_r**3) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr) ** 3) / 12  
        # second moment of area of rotot arm w.r.t torsion
        I_arm_tor_r = (d_r * b_r**3) - ((d_r - 2 * t_wr) * (b_r - 2 * t_wr) ** 3) / 12  
        R = r_g - g - h_m - 0.5 * t

        # Rotor mean radius
        outputs["y_allowable_r"] = y_allow_pcent * l / 100
        outputs["u_allowable_r"] = u_allow_pcent * g / 100  # allowable radial deflection
        R_1 = R - t * 0.5  # inner radius of rotor cylinder
        k_1 = np.sqrt(I_r / A_r)  # radius of gyration
        m1 = (k_1 / R) ** 2
        l_ir = R  # length of rotor arm beam at which rotor cylinder acts
        l_iir = R_1
        outputs["b_allowable_r"] = 2 * np.pi * R_sh / N_r  # allowable circumferential arm dimension for rotor
        outputs["mass_PM"] = 2 * np.pi * (R + 0.5 * t) * l * h_m * ratio * rho_PM  # magnet mass

        # Calculating radial deflection of the rotor
        Numer = R**3 * (
            (0.25 * (np.sin(theta_r) - (theta_r * np.cos(theta_r))) / (np.sin(theta_r)) ** 2)
            - (0.5 / np.sin(theta_r))
            + (0.5 / theta_r)
        )
        Pov = ((theta_r / (np.sin(theta_r)) ** 2) + 1 / np.tan(theta_r)) * ((0.25 * R / A_r) + (0.25 * R**3 / I_r))
        Qov = R**3 / (2 * I_r * theta_r * (m1 + 1))
        Lov = (R_1 - R_sh) / a_r
        Denom = I_r * (Pov - Qov + Lov)  # radial deflection % rotor
        outputs["u_ar"] = (Sigma_normal * R**2 / E / t) * (1 + Numer / Denom)

        # Calculating axial deflection of the rotor under its own weight
        # uniformly distributed load of the weight of the rotor arm
        w_r = rho_Fes * g1 * np.sin((np.deg2rad(phi))) * a_r

        # TODO: The mass_Fe here seems the same accounting as "mass_iron" in the magnetics code
        outputs["mass_Fe_rotor"] = mass_st_lam = rho_Fe * 2 * np.pi * (R) * l * h_yr  # mass of rotor yoke steel

        # weight of 1/nth of rotor cylinder
        W = g1 * np.sin((np.deg2rad(phi))) * (mass_st_lam / N_r + (outputs["mass_PM"]) / N_r)

        y_a1 = W * l_ir**3 / 12 / E / I_arm_axi_r  # deflection from weight component of back iron
        y_a2 = w_r * l_iir**4 / 24 / E / I_arm_axi_r  # deflection from weight component of yhe arms
        outputs["y_ar"] = y_a1 + y_a2  # axial deflection

        # Calculating # circumferential deflection of the rotor
        # allowable torsional deflection
        outputs["z_allowable_r"] = z_allow_deg * 2 * np.pi * R / 360  
        # circumferential deflection
        outputs["z_ar"] = (
            (2 * np.pi * (R - 0.5 * t) * l / N_r) * Sigma_shear * (l_ir - 0.5 * t) ** 3 / 3 / E / I_arm_tor_r
        )
        outputs["mass_structural_rotor"] = (N_r * (R_1 - R_sh) * a_r * rho_Fes) + np.pi * (
            (r_g - g - h_m - h_yr) ** 2 - (r_g - g - h_m - t) ** 2
        ) * l * rho_Fes
        


class PMSG_stator_inactive(om.ExplicitComponent):
    def setup(self):

        self.add_input("n_s", 0.0, desc="number of stator arms n_s")
        self.add_input("b_st", 0.0, units="m", desc="arm width b_r")
        self.add_input("d_s", 0.0, units="m", desc="arm depth d_r")
        self.add_input("t_ws", 0.0, units="m", desc="arm depth thickness self.t_wr")
        self.add_input("t_s", 0.0, units="m", desc="stator back iron ")
        self.add_input("g", 0.0, units="m", desc="airgap length")
               
        self.add_input("u_allow_pcent", 0.0, desc="Allowable radial deflection percent")
        self.add_input("y_allow_pcent", 0.0, desc="Allowable axial deflection percent")
        self.add_input("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist ")
        self.add_input("r_g", 0.0, units="m", desc="air gap radius")
        self.add_input("h_s", 0.0, units="m", desc="Yoke height h_s")
        self.add_input("h_ys", 0.0, units="m", desc="Yoke height")
        self.add_input("phi", 0.0, units="deg", desc="tilt angle")
        self.add_input("h_ss", 0.0, units="m", desc="stator rim thickness")
        self.add_input("L_t", 0.0, units="m", desc="effective length for support structure design")
        #self.add_input("B_g", 0.0, units="T", desc="Peak air gap flux density")
        self.add_input("E", 2.0e11, desc="Youngs modulus")
        self.add_input("g1", 9.806, desc="Acceleration due to gravity")
        self.add_input("R_no", 0.0, units="m", desc="nose radius")
        self.add_input("Sigma_normal", 0.0, units="N/m**2", desc="Normal stress")
        self.add_input("Sigma_shear", 0.0, units="N/m**2", desc="Shear stress")
        self.add_input("rho_Fes", 0.0, units="kg/m**3", desc="Structural Steel density kg/m^3")
        self.add_input("rho_Fe", 0.0, units="kg/m**3", desc="Magnetic Steel density kg/m^3")
        self.add_input("mass_Fe_rotor", 0.0, units="kg", desc="Iron mass")
        self.add_input("mass_copper", 0.0, units="kg", desc="Copper Mass")
        self.add_input("M_Fest", 0.0, units="kg", desc="Stator teeth mass")
        self.add_input("mass_structural_rotor", 0.0, units="kg", desc="Rotor structural mass kg")
        self.add_input("h_s1", 0.010, desc="Slot Opening height")
        self.add_input("h_s2", 0.010, desc="Wedge Opening height")

        self.add_output("u_as", 0.0, units="m", desc="Stator radial deflection")
        self.add_output("y_as", 0.0, units="m", desc="Stator axial deflection")
        self.add_output("D_out", 0.0, units="m", desc="Stator outer diameter")
        self.add_output("z_as", 0.0, units="m", desc="Stator circumferential deflection")
        self.add_output("u_allowable_s", 0.0, units="m", desc="Allowable radial stator")
        self.add_output("y_allowable_s", 0.0, units="m", desc="Allowable axial")
        self.add_output("b_allowable_s", 0.0, units="m", desc="Allowable circumferential")
        self.add_output("z_allowable_s", 0.0, units="m", desc="Allowable circum stator")

        self.add_output("mass_Fe", 0.0, units="kg", desc="Iron mass")
        self.add_output("mass_Fe_stator", 0.0, units="kg", desc="Iron mass in stator")
        self.add_output("mass_structural_stator", 0.0, units="kg", desc="Structural mass of stator")
        self.add_output("mass_structural", 0.0, units="kg", desc="Total structural mass")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        Sigma_shear = inputs["Sigma_shear"]
        r_g = inputs["r_g"]
        N_st = inputs["n_s"]
        b_st = inputs["b_st"]
        d_s = inputs["d_s"]
        t_ws = inputs["t_ws"]
        h_ys = inputs["h_ys"]
        h_ss = inputs["h_ss"]
        h_s = inputs["h_s"]
        R_no = inputs["R_no"]
        t_s = h_ys + h_ss
        Sigma_normal = inputs["Sigma_normal"]
        E = inputs["E"]
        rho_Fes = inputs["rho_Fes"]
        rho_Fe = inputs["rho_Fe"]
        Copper = inputs["mass_copper"]
        M_Fest = inputs["M_Fest"]
        L_t = inputs["L_t"]  # end winding length
        g = inputs["g"]
        phi = inputs["phi"]
        u_allow_pcent = inputs["u_allow_pcent"]
        y_allow_pcent = inputs["y_allow_pcent"]
        z_allow_deg = inputs["z_allow_deg"]
        g1 = inputs["g1"]
        h_s1 = float(inputs["h_s1"])
        h_s2 = float(inputs["h_s2"])

        # stator structure deflection calculation
        a_s = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))  # cross-sectional area of stator armms
        A_st = L_t * t_s  # cross-sectional area of stator cylinder
        #N_st = np.round(n_s)  # stator arms
        theta_s = np.pi * 1 / N_st  # half angle between spokes
        I_st = L_t * t_s**3 / 12  # second moment of area of stator cylinder
        k_2 = np.sqrt(I_st / A_st)  # radius of gyration

        # second moment of area of stator arm
        I_arm_axi_s = (b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws) ** 3) / 12  
        # second moment of area of rotot arm w.r.t torsion
        I_arm_tor_s = (d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws) ** 3) / 12  
        R_st = r_g + h_s + h_s1 + h_s2 + t_s * 0.5  # stator cylinder mean radius
        R_1s = R_st - t_s * 0.5  # inner radius of stator cylinder, m
        m2 = (k_2 / R_st) ** 2

        outputs["D_out"] = 2 * r_g + 2 * (h_ys + h_s + h_s1 + h_s2 + h_ss)
        l_is = R_st - R_no  # distance at which the weight of the stator cylinder acts
        l_iis = l_is  # distance at which the weight of the stator cylinder acts
        l_iiis = l_is  # distance at which the weight of the stator cylinder acts

        # TODO: The mass_Fe here seems the same accounting as "mass_iron" in the magnetics code
        outputs["mass_Fe_stator"] = mass_st_lam_s = M_Fest + np.pi * L_t * rho_Fe * (
            (r_g + h_s + h_s1 + h_s2 + h_ys) ** 2 - (r_g + h_s + h_s1 + h_s2) ** 2
        )
        # length of stator arm beam at which self-weight acts
        W_is = 0.5 * g1 * np.sin(np.deg2rad(phi)) * (rho_Fes * L_t * d_s**2)
        # weight of stator cylinder and teeth
        W_iis = (g1 * np.sin((np.deg2rad(phi))) * (mass_st_lam_s + Copper +
                                                   np.pi * ((R_st + t_s * 0.5) ** 2 - (R_st + t_s * 0.5 - h_ss) ** 2) * L_t * rho_Fes
                                                   ) / N_st)

        w_s = rho_Fes * g1 * np.sin(np.deg2rad(phi)) * a_s  # uniformly distributed load of the arms

        # print (M_Fest+self.Copper)*g1

        outputs["mass_structural_stator"] = (
            2 * (N_st * (R_1s - R_no) * a_s * rho_Fes)
            + np.pi * ((R_st + t_s * 0.5) ** 2 - (R_st + t_s * 0.5 - h_ss) ** 2) * L_t * rho_Fes
        )

        outputs["y_allowable_s"] = y_allow_pcent * L_t / 100
        outputs["u_allowable_s"] = u_allow_pcent * g / 100  # allowable radial deflection
        # Calculating radial deflection of the stator

        Numers = R_st**3 * (
            (0.25 * (np.sin(theta_s) - (theta_s * np.cos(theta_s))) / (np.sin(theta_s)) ** 2)
            - (0.5 / np.sin(theta_s))
            + (0.5 / theta_s)
        )
        Povs = ((theta_s / (np.sin(theta_s)) ** 2) + 1 / np.tan(theta_s)) * (
            (0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st)
        )
        Qovs = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs = (R_1s - R_no) * 0.5 / (a_s)
        Denoms = I_st * (Povs - Qovs + Lovs)

        outputs["u_as"] = (Sigma_normal * R_st**2 / E / t_s) * (1 + Numers / Denoms)

        # Calculating axial deflection of the stator

        X_comp1 = (
            W_is * l_is**3 / 12 / E / I_arm_axi_s
        )  # deflection component due to stator arm beam at which self-weight acts
        X_comp2 = W_iis * l_iis**3 / 12 / E / I_arm_axi_s  # deflection component due to 1/nth of stator cylinder
        X_comp3 = w_s * l_iiis**4 / 24 / E / I_arm_axi_s  # deflection component due to weight of arms

        outputs["y_as"] = X_comp1 + X_comp2 + X_comp3  # axial deflection

        # Calculating circumferential deflection of the stator
        outputs["z_as"] = (
            np.pi
            * (R_st + 0.5 * t_s)
            * L_t
            * 2
            / (2 * N_st)
            * Sigma_shear
            * (l_is + 0.5 * t_s) ** 3
            / 3
            / E
            / I_arm_tor_s
        )
        outputs["z_allowable_s"] = z_allow_deg * 2 * np.pi * R_st / 360  # allowable torsional deflection
        outputs["b_allowable_s"] = 2 * np.pi * R_no / N_st  # allowable circumferential arm dimension

        # TODO: The mass_Fe here seems the same accounting as "mass_iron" in the magnetics code
        outputs["mass_Fe"] = outputs["mass_Fe_stator"] + inputs["mass_Fe_rotor"]
        outputs["mass_structural"] = outputs["mass_structural_stator"] + inputs["mass_structural_rotor"]


class PMSG_Inner_Rotor_Structural(om.Group):
    def setup(self):
        #        self.linear_solver = lbgs = om.LinearBlockJac() #om.LinearBlockGS()
        #        self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        #        nlbgs.options["maxiter"] = 3
        #        nlbgs.options["atol"] = 1e-2
        #        nlbgs.options["rtol"] = 1e-8
        #        nlbgs.options["iprint"] = 2

        ivcs = om.IndepVarComp()
        ivcs.add_output("gamma", 1.5, desc="Partial safety factor")
        ivcs.add_output("R_sh", 0.0, units="m", desc=" Main shaft outer radius")
        ivcs.add_output("R_no", 0.0, units="m", desc=" Bedplate nose outer radius")
        ivcs.add_output("phi", 0.0, units="deg", desc=" Main shaft tilt angle")
        ivcs.add_output("h_sr", 0.0, units="m", desc="Rotor rim thickness")
        ivcs.add_output("h_ss", 0.0, units="m", desc="Stator rim thickness")
        ivcs.add_output("n_r", 0.0, desc="Rotor arms")
        ivcs.add_output("n_s", 0.0, desc="Stator arms")
        ivcs.add_output("t_wr", 0.0, units="m", desc="rotor arm thickness")
        ivcs.add_output("t_ws", 0.0, units="m", desc="stator arm thickness")
        ivcs.add_output("b_r", 0.0, units="m", desc="rotor arm circumferential dimension")
        ivcs.add_output("b_st", 0.0, units="m", desc="stator arm circumferential dimension")
        ivcs.add_output("d_r", 0.0, units="m", desc="rotor arm depth")
        ivcs.add_output("d_s", 0.0, units="m", desc="stator arm depth")
        ivcs.add_output("t_s", 0.0, units="m", desc="Stator disc thickness")
        #ivcs.add_output("ratio", 0.0, desc="ratio of magnet width to pole pitch(bm/self.tau_p")

        ivcs.add_output("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        ivcs.add_output("y_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        ivcs.add_output("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")

        #ivcs.add_output("L_t", 0.0, units="m", desc="effective length for support structure design")
        #ivcs.add_output("B_g", 0.0, units="T", desc="Peak air gap flux density")
        #ivcs.add_output("Sigma_normal", 0.0, units="N/m**2", desc="Normal stress")
        #ivcs.add_output("Sigma_shear", 0.0, units="N/m**2", desc="Shear stress")
        ivcs.add_output("rho_Fes", 0.0, units="kg/m**3", desc="Structural Steel density kg/m^3")
        #ivcs.add_output("rho_Fe", 0.0, units="kg/m**3", desc="Magnetic Steel density kg/m^3")
        #ivcs.add_output("mass_copper", 0.0, units="kg", desc="Copper Mass")
        #ivcs.add_output("M_Fest", 0.0, units="kg", desc="Stator teeth mass")
        ivcs.add_output("g1", 9.806, desc="Acceleration due to gravity")
        ivcs.add_output("E", 2.0e11, desc="Youngs modulus")

        # ivcs.add_output("perc_allowable_radial", 0.0, desc=" Allowable radial % deflection ")
        # ivcs.add_output("perc_allowable_axial", 0.0, desc=" Allowable axial % deflection ")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys1", PMSG_rotor_inactive(), promotes=["*"])
        self.add_subsystem("sys2", PMSG_stator_inactive(), promotes=["*"])
        self.add_subsystem("con", structural_constraints(), promotes=["*"])


if __name__ == "__main__":

    prob_struct = om.Problem()
    prob_struct.model = PMSG_Inner_Rotor_Structural()

    prob_struct.driver = om.ScipyOptimizeDriver()
    prob_struct.driver.options["optimizer"] = "SLSQP"
    prob_struct.driver.options["maxiter"] = 100

    # recorder = om.SqliteRecorder("log.sql")
    # prob_struct.driver.add_recorder(recorder)
    # prob_struct.add_recorder(recorder)
    # prob_struct.driver.recording_options["excludes"] = ["*_df"]
    # prob_struct.driver.recording_options["record_constraints"] = True
    # prob_struct.driver.recording_options["record_desvars"] = True
    # prob_struct.driver.recording_options["record_objectives"] = True

    prob_struct.model.add_design_var("h_sr", lower=0.045, upper=0.25)
    prob_struct.model.add_design_var("h_ss", lower=0.045, upper=0.25)
    prob_struct.model.add_design_var("n_r", lower=5, upper=15)
    prob_struct.model.add_design_var("b_r", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("d_r", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("t_wr", lower=0.001, upper=0.2)
    prob_struct.model.add_design_var("n_s", lower=5, upper=15)
    prob_struct.model.add_design_var("b_st", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("d_s", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("t_ws", lower=0.001, upper=0.2)
    prob_struct.model.add_objective("mass_structural")

    prob_struct.model.add_constraint("U_rotor_radial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_rotor_axial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_stator_radial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_stator_axial_constraint", upper=1.0)

    prob_struct.model.approx_totals(method="fd")

    prob_struct.setup()
    # --- Design Variables ---

    # Initial design variables for a PMSG designed for a 15MW turbine
    prob_struct["Sigma_shear"] = 74.99692029e3
    prob_struct["Sigma_normal"] = 535e3
    prob_struct["T_e"] = 17e06
    prob_struct["r_g"] = 1
    prob_struct["g"] = 0.010
    prob_struct["h_m"] = 0.1
    prob_struct["l_s"] = 0.6
    prob_struct["h_s"] = 0.1803019703
    prob_struct["ratio"] = 0.95

    prob_struct["rho_Fes"] = 7850
    prob_struct["rho_Fe"] = 7700
    prob_struct["rho_Copper"] = 8800
    prob_struct["rho_PM"] = 7600

    prob_struct["phi"] = 90.0
    prob_struct["u_allow_pcent"] = 10
    prob_struct["y_allow_pcent"] = 20
    prob_struct["z_allow_deg"] = 0.5
    prob_struct["h_yr"] = 0.1254730934
    prob_struct["h_ys"] = 0.050
    prob_struct["n_s"] = 5
    prob_struct["b_st"] = 0.4
    prob_struct["n_r"] = 6
    prob_struct["b_r"] = 0.200
    prob_struct["d_r"] = 0.500
    prob_struct["d_s"] = 0.6
    prob_struct["t_wr"] = 0.02
    prob_struct["t_ws"] = 0.04
    prob_struct["h_sr"] = 0.02
    prob_struct["h_ss"] = 0.02
    prob_struct["R_sh"] = 0.3
    prob_struct["R_no"] = 0.35

    prob_struct["L_t"] = 1.6
    prob_struct["B_g"] = 1.49

    prob_struct["mass_copper"] = 5000
    prob_struct["M_Fest"] = 15000
    prob_struct["Rotor_active"] = 20000

    prob_struct["u_allow_pcent"] = 5.0  #
    prob_struct["y_allow_pcent"] = 20.0
    prob_struct["z_allow_deg"] = 0.5
    prob_struct["gamma"] = 1.5

    prob_struct.model.approx_totals(method="fd")

    # prob_struct.run_model()
    prob_struct.run_driver()

    prob_struct.model.list_outputs(val=True, hierarchical=True)

    raw_data = {
        "Parameters": [
            "# Rotor spokes",
            "Rotor spoke thickness",
            "Rotor rim thickness",
            "Rotor spoke circumferential dimension",
            "Rotor spoke depth",
            "# Stator spokes",
            "Stator spoke thickness",
            "Stator rim thickness",
            "Stator spoke circumferential dimension",
            "Stator spoke depth",
            "rotor radial deflection",
            "rotor axial deflection",
            "rotor twist",
            "stator radial deflection",
            "stator axial deflection",
            "stator twist" "Rotor structural mass",
            "Stator structural mass",
            "Total structural mass",
        ],
        "Values": [
            prob_struct.get_val("n_r"),
            prob_struct.get_val("t_wr", units="mm"),
            prob_struct.get_val("h_sr", units="mm"),
            prob_struct.get_val("b_r", units="mm"),
            prob_struct.get_val("d_r", units="mm"),
            prob_struct.get_val("n_s"),
            prob_struct.get_val("t_ws", units="mm"),
            prob_struct.get_val("h_ss", units="mm"),
            prob_struct.get_val("b_st", units="mm"),
            prob_struct.get_val("d_s", units="mm"),
            prob_struct.get_val("u_ar", units="mm"),
            prob_struct.get_val("y_ar", units="mm"),
            prob_struct.get_val("z_a_r", units="mm"),
            prob_struct.get_val("u_as", units="mm"),
            prob_struct.get_val("y_as", units="mm"),
            prob_struct.get_val("z_a_s", units="mm"),
            prob_struct.get_val("Structural_rotor", units="t"),
            prob_struct.get_val("Structural_stator", units="t"),
            prob_struct.get_val("mass_structural", units="t"),
        ],
        "Limit": [
            "",
            "",
            "",
            prob.get_val("b_allowable_r", units="mm"),
            "",
            "",
            "",
            "",
            prob.get_val("b_allowable_s", units="mm"),
            "",
            prob.get_val("u_allowable_r", units="mm"),
            prob.get_val("y_all", units="mm"),
            prob.get_val("z_allowable_r", units="mm"),
            prob.get_val("u_allowable_r", units="mm"),
            prob.get_val("y_all", units="mm"),
            prob.get_val("z_allowable_s", units="mm"),
            "",
            "",
            "",
        ],
        "Units": [
            "",
            "mm",
            "mm",
            "mm",
            "mm",
            "",
            "mm",
            "mm",
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
    df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])
    # df.to_excel("Optimized_structure_MS_PMSG_MW.xlsx")
    print(df)
