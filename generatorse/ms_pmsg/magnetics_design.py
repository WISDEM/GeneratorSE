"""
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071
"""

import numpy as np
import openmdao.api as om


class PMSG_active(om.ExplicitComponent):
    def setup(self):
        self.add_input("r_g", 0.0, units="m", desc="airgap radius")
        self.add_input("g", 0.0, units="m", desc="airgap length")
        self.add_input("l_s", 0.0, units="m", desc="Stator core length ")
        self.add_input("h_s", 0.0, units="m", desc="Yoke height h_s")
        #self.add_input("cofi", 0.8, desc="Power factor")
        self.add_output("tau_p", 0.0, units="m", desc="Pole pitch self.tau_p")
        self.add_output("tau_s", 0.0, units="m", desc="Stator slot pitch")

        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        #self.add_input("T_rated", 0.0, units="N*m", desc="Rated torque ")
        #self.add_input("h_m", 0.0, units="m", desc="magnet height")
        #self.add_input("h_ys", 0.0, units="m", desc="Yoke height")
        #self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("N_c", 0.0, desc="turns per coil")
        self.add_input("b_s_tau_s", 0.45, desc="ratio of Slot width to slot pitch ")
        #self.add_input("B_r", 0, units="T", desc="Tesla remnant flux density")
        self.add_input("ratio", 0.00, desc="ratio of magnet width to pole pitch")
        #self.add_input("mu_0", np.pi * 4e-7, desc="permeability of free space")
        #self.add_input("mu_r", 0.0, desc="relative permeability ")
        self.add_input("k_sfil", 0.65, desc="slot fill factor")

        self.add_input("m", 3, desc=" no of phases")
        self.add_input("q1", 1, desc=" no of slots per pole per phase")
        # specific hysteresis losses W/kg @ 1.5 T
        self.add_input("resisitivty_Cu", 0.0, units="ohm*m", desc=" Copper resisitivty")

        # PMSG_arms generator design outputs
        # Magnetic loading

        self.add_output("L_t", 0.0, units="m", desc="effective stator length for support structure")

        # Stator design
        self.add_output("N_s", 0.0, desc="Number of turns in the stator winding")
        self.add_output("b_s", 0.0, units="m", desc="slot width")
        self.add_output("b_t", 0.0, units="m", desc="tooth width")
        self.add_output("A_Cuscalc", 0.0, units="mm**2", desc="Conductor cross-section mm^2")

        # Rotor magnet dimension
        self.add_output("b_m", 0.0, units="m", desc="magnet width")
        self.add_input("p", desc="No of pole pairs")
        self.add_output("k_wd", desc="Winding factor")

        # Electrical performance
        self.add_output("f", 0.0, units="Hz", desc="Generator output frequency")
        self.add_input("I_s", 0.0, units="A", desc="Generator output phase current")
        self.add_output("R_s", 0.0, units="ohm", desc="Stator resistance")

        self.add_output("A_1", 0.0, units="A/m", desc="Electrical loading")
        self.add_output("J_s", 0.0, units="A/mm**2", desc="Current density")

        # Objective functions
        self.add_output("Mass", 0.0, units="kg", desc="Actual mass")
        self.add_output("K_rad", 0.0, desc="Aspect ratio")
        #self.add_input("h_s1", 0.010, desc="Slot Opening height")
        #self.add_input("h_s2", 0.010, desc="Wedge Opening height")

        # Other parameters
        self.add_output("S", desc="Stator slots")
        self.add_output("Slot_aspect_ratio", desc="Slot aspect ratio")

        # Mass Outputs
        self.add_output("Copper", 0.0, units="kg", desc="Copper Mass")

        self.add_output("P_Cu", 0.0, units="W", desc="Copper losses")
        self.add_output("P_Ftm", 0.0, units="W", desc="Magnet losses")

        # Material properties

        self.add_input("rho_Copper", 0.0, units="kg/m**3", desc="Copper density kg/m^3")
        self.add_input("rho_PM", 0.0, units="kg/m**3", desc="Magnet density kg/m^3")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        # Unpack inputs
        r_g = inputs["r_g"]
        g = inputs["g"]
        m = inputs["m"]
        q1 = inputs["q1"]
        l_s = inputs["l_s"]
        h_s = inputs["h_s"]
        p = inputs["p"]
        #h_ys = inputs["h_ys"]
        #h_yr = inputs["h_yr"]
        #h_m = inputs["h_m"]
        I_s = inputs["I_s"]

        #T_rated = inputs["T_rated"]
        N_nom = inputs["N_nom"]
        b_s_tau_s = inputs["b_s_tau_s"]
        k_sfil = inputs["k_sfil"]
        ratio = inputs["ratio"]
        #B_r = inputs["B_r"]
        N_c = inputs["N_c"]

        # specific hysteresis losses W/kg @ 1.5 T
        rho_Cu = inputs["resisitivty_Cu"]

        rho_Copper = inputs["rho_Copper"]

        #mu_0 = inputs["mu_0"]
        #mu_r = inputs["mu_r"]
        #cofi = inputs["cofi"]
        #h_s1 = float(inputs["h_s1"])
        #h_s2 = float(inputs["h_s2"])
        # Assign values to design constants

        #h_i = 0.001  # coil insulation thickness

        #k_fes = 0.9  # Stator iron fill factor per Grauers
        #b_so = 2 * g  # Slot opening
        #alpha_p = np.pi / 2 * ratio
        #N_c = np.round(N_c)
        #p = np.round(p)

        outputs["K_rad"] = l_s / (2 * r_g)  # Aspect ratio

        dia = 2 * r_g  # air gap diameter
        outputs["tau_p"] = tau_p = 2 * np.pi * (r_g - g) / (2 * p)
        # equivalent core length
        outputs["b_m"] = ratio * tau_p  # magnet width

        outputs["f"] = N_nom * p / 60  # outout frequency
        outputs["S"] = 2 * p * q1 * m  # Stator slots
        # Stator turns per phase
        outputs["tau_s"] = tau_s = np.pi * dia / outputs["S"]
        # Stator slot pitch
        outputs["b_s"] = b_s = b_s_tau_s * outputs["tau_s"]  # slot width
        outputs["b_t"] = tau_s - (outputs["b_s"])  # tooth width
        outputs["Slot_aspect_ratio"] = h_s / outputs["b_s"]

        #om_m = 2 * np.pi * N_nom / 60
        #om_e = p * om_m / 2

        # Calculating winding factor
        outputs["k_wd"] = np.sin(np.pi / 6) / q1 / np.sin(np.pi / 6 / q1)

        outputs["L_t"] = l_s + 2 * tau_p
        outputs["N_s"] = N_s = inputs["N_c"] * 2 * outputs["S"] / (m * q1)
        # Calculating no-load voltage induced in the stator

        l_Cus = N_s * (2 * tau_p + l_s)
        A_s = b_s * (h_s) * 0.5

        #A_scalc = A_s * q1 * p
        A_Cus = A_s * k_sfil / (N_c)

        outputs["R_s"] = l_Cus * rho_Cu / (A_Cus)

        outputs["J_s"] = I_s / (A_Cus * 1e6)

        # Calculating leakage inductance in  stator

        # Calculating stator current and electrical loading
        # self.I_s= sqrt(Z**2+(((self.E_p-G**0.5)/(om_e*self.L_s)**2)**2))

        outputs["A_1"] = 3 * N_s * inputs["I_s"] / (np.pi * dia) * 0.707

        V_Cus = m * l_Cus * A_Cus
        outputs["Copper"] = V_Cus * rho_Copper

        ##1. Copper Losses
        K_R = 1.2  # Skin effect correction co-efficient
        outputs["P_Cu"] = m * (I_s / 2**0.5) ** 2 * outputs["R_s"] * K_R

        # Stator winding length ,cross-section and resistance
        pFtm = 300  # specific magnet loss
        outputs["P_Ftm"] = pFtm * 2 * p * outputs["b_m"] * l_s


class Results_by_analytical_model(om.ExplicitComponent):
    def setup(self):

        self.add_input("r_g", 0.0, units="m", desc="airgap radius")
        self.add_input("g", 0.0, units="m", desc="airgap length")
        self.add_input("l_s", 0.0, units="m", desc="Stator core length ")
        self.add_input("L_t", 0.0, units="m", desc="Stator core length ")
        self.add_input("tau_s", 0.0, units="m", desc="Slot pitch")
        self.add_input("tau_p", 0.0, units="m", desc="Pole pitch ")
        self.add_input("T_rated", 0.0, units="N*m", desc="Machine rating")
        #self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("h_m", 0.0, units="m", desc="magnet height")

        self.add_input("h_ys", 0.0, units="m", desc="Yoke height")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("N_s", 0.0, desc="turns per phase")
        self.add_output("L_s", 0.0, units="m", desc="Stator synchronising inductance")
        self.add_input("B_r", 0.0, units="T", desc="Tesla remnant flux density")
        self.add_input("mu_0", np.pi * 4 * 1e-7, desc="permeability of free space")
        self.add_input("mu_r", 0.0, desc="relative permeability ")
        self.add_input("m", 3, desc=" no of phases")
        self.add_input("q1", 1, desc=" no of slots per pole per phase")
        self.add_output("M_Fest", 0.0, units="kg", desc="Stator teeth mass")
        self.add_output("M_Fesy", 0.0, units="kg", desc="Stator yoke mass")
        self.add_output("M_Fery", 0.0, units="kg", desc="Rotor yoke mass")
        self.add_output("Iron", 0.0, units="kg", desc="Electrical Steel Mass")
        self.add_input("rho_Fe", 0.0, units="kg/m**3", desc="Magnetic Steel density kg/m^3")

        self.add_input("h_s", 0.0, units="m", desc="slot height ")
        self.add_input("b_t", 0.0, units="m", desc="tooth width")
        self.add_input("b_s", 0.0, units="m", desc="slot width")

        self.add_input("A_1", 0.0, units="A/m", desc="specific current loading")
        self.add_input("h_s1", 0.010, desc="Slot Opening height")
        self.add_input("h_s2", 0.010, desc="Wedge Opening height")

        self.add_input("p", desc="No of pole pairs")

        self.add_output("B_symax", 0.0, units="T", desc="Peak Stator Yoke flux density B_ymax")
        self.add_output("B_tmax", 0.0, units="T", desc="Peak Teeth flux density")
        self.add_output("B_rymax", 0.0, units="T", desc="Peak Rotor yoke flux density")
        self.add_output("B_smax", 0.0, units="T", desc="Peak Stator flux density")
        self.add_output("B_pm1", 0.0, units="T", desc="Fundamental component of peak air gap flux density")
        self.add_output("B_g", 0.0, units="T", desc="Peak air gap flux density B_g")

        # Rotor magnet dimension
        self.add_input("b_m", 0.0, units="m", desc="magnet width")
        self.add_input("k_fes", 0.95, desc="Iron stacking factor")
        self.add_input("ratio", 0.0, desc="ratio of magnet width to pole pitch(bm/self.tau_p")

        #self.add_input("I_s", 0.0, units="A", desc="Generator output phase current")

        # Objective functions

        self.add_output("Sigma_shear", 0.0, units="N/m**2", desc="Shear stress N/m**2")
        self.add_output("T_e", 0.0, units="N*m", desc="Torque ")
        self.add_output("Sigma_normal", 0.0, units="N/m**2", desc="Normal stress N/m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        # Unpack inputs
        r_g = inputs["r_g"]
        g = inputs["g"]
        l_s = inputs["l_s"]
        h_ys = inputs["h_ys"]
        h_yr = inputs["h_yr"]
        h_m = inputs["h_m"]
        b_m = inputs["b_m"]
        #I_s = inputs["I_s"]
        #N_nom = inputs["N_nom"]
        B_r = inputs["B_r"]
        p = inputs["p"]
        m = inputs["m"]
        mu_r = inputs["mu_r"]
        k_fes = inputs["k_fes"]
        mu_0 = inputs["mu_0"]
        q1 = inputs["q1"]
        rho_Fe = inputs["rho_Fe"]
        L_t = inputs["L_t"]
        tau_p = float(inputs["tau_p"])
        tau_s = float(inputs["tau_s"])
        b_s = inputs["b_s"]
        b_t = inputs["b_t"]
        h_s = inputs["h_s"]
        N_s = inputs["N_s"]
        ratio = inputs["ratio"]
        A_1 = inputs["A_1"]

        h_s1 = inputs["h_s1"]
        h_s2 = inputs["h_s2"]
        h_w = h_s2
        alpha_p = np.pi / 2 * ratio
        b_so = 2 * g  # Slot opening

        y_tau_p = 1  # Coil span to pole pitch
        l_u = k_fes * l_s  # useful iron stack length
        # air gap diameter

        #l_b = 2 * tau_p  # end winding length
        l_e = l_s + 2 * g  # equivalent core length

        h_w = h_s1 + h_s2
        # Calculating Carter factor for statorand effective air gap length
        gamma = (
            4
            / np.pi
            * (
                b_so / 2 / (g + h_m / mu_r) * np.arctan(b_so / 2 / (g + h_m / mu_r))
                - np.log(np.sqrt(1 + (b_so / 2 / (g + h_m / mu_r)) ** 2))
            )
        )
        k_C = tau_s / (tau_s - gamma * (g + h_m / mu_r))  # carter coefficient
        g_eff = k_C * (g + h_m / mu_r)
        # angular frequency in radians
        #om_m = 2 * np.pi * N_nom / 60
        #om_e = p * om_m

        # Calculating winding factor
        k_wd = np.sin(np.pi / 6) / q1 / np.sin(np.pi / 6 / q1)

        # Calculating Electromagnetically active mass
        r_r = r_g - g
        # copper volume
        V_Fest = L_t * 2 * p * q1 * m * b_t * h_s  # volume of iron in stator tooth
        V_Fesy = L_t * np.pi * ((r_g + h_s + h_ys) ** 2 - (r_g + h_s) ** 2)  # volume of iron in stator yoke
        V_Fery = L_t * np.pi * ((r_r - h_m) ** 2 - (r_r - h_m - h_yr) ** 2)

        outputs["M_Fest"] = V_Fest * rho_Fe  # Mass of stator tooth
        outputs["M_Fesy"] = V_Fesy * rho_Fe  # Mass of stator yoke
        outputs["M_Fery"] = V_Fery * rho_Fe  # Mass of rotor yoke
        outputs["Iron"] = outputs["M_Fest"] + outputs["M_Fesy"] + outputs["M_Fery"]

        L_m = 2 * m * k_wd**2 * (N_s) ** 2 * mu_0 * tau_p * L_t / np.pi**2 / g_eff / p
        # slot leakage inductance
        L_ssigmas = 2 * mu_0 * l_s * N_s**2 / p / q1 * ((h_s - h_w) / (3 * b_s) + h_w / b_so)
        # end winding leakage inductance
        L_ssigmaew = (2 * mu_0 * l_s * N_s**2 / p / q1) * 0.34 * g * (l_e - 0.64 * tau_p * y_tau_p) / l_s
        # tooth tip leakage inductance#tooth tip leakage inductance
        L_ssigmag = 2 * mu_0 * l_s * N_s**2 / p / q1 * (5 * (g * k_C / b_so) / (5 + 4 * (g * k_C / b_so)))
        L_ssigma = L_ssigmas + L_ssigmaew + L_ssigmag
        outputs["L_s"] = L_m + L_ssigma
        #X_snom = om_e * (L_m + L_ssigma)

        # Calculating magnetic loading
        outputs["B_pm1"] = B_r * h_m / mu_r / (g_eff)
        outputs["B_g"] = B_g = B_r * h_m / mu_r / ((g + h_m / mu_r)) * (4 / np.pi) * np.sin(alpha_p)
        outputs["B_symax"] = B_g * b_m * l_e / (2 * h_ys * l_u)
        outputs["B_rymax"] = B_g * b_m * l_e / (2 * h_yr * l_s)
        outputs["B_tmax"] = B_g * tau_s / b_t

        outputs["Sigma_shear"] = 0.707 * B_g * A_1
        outputs["T_e"] = outputs["Sigma_shear"] * 2 * np.pi * r_g**2 * l_s
        outputs["Sigma_normal"] = (B_g**2) / (2 * mu_0)


class Results(om.ExplicitComponent):
    def setup(self):

        self.add_input("B_g", 0.0, units="T", desc="Peak air gap flux density")
        self.add_input("B_symax", 0.0, units="T", desc="Stator yoke flux density")
        self.add_input("B_tmax", 0.0, units="T", desc="Tooth yoke flux density")
        self.add_input("B_rymax", 0.0, units="T", desc="Rotor yoke flux density")
        self.add_input("M_Fest", 0.0, units="kg", desc="Stator teeth mass")
        self.add_input("M_Fesy", 0.0, units="kg", desc="Stator yoke mass")
        self.add_input("M_Fery", 0.0, units="kg", desc="Rotor yoke mass")
        self.add_input("h_m", 0.0, units="m", desc="magnet thickness")
        self.add_input("P_Fe0h", 4.0, desc="specific hysteresis losses W/kg @ 1.5 T")
        self.add_input("P_Fe0e", 1.0, desc="specific eddy losses W/kg @ 1.5 T")
        self.add_input("P_Cu", 0.0, units="W", desc="Copper losses ")
        self.add_input("P_Ftm", 0.0, units="W", desc="Magnet losses")
        self.add_input("P_rated", 0.0, units="W", desc="Machine rating")
        self.add_output("Losses", 0.0, units="W", desc="Total loss")
        self.add_output("gen_eff", 0.0, desc="Generator efficiency")
        self.add_output("E_p", 0.0, units="V", desc="Stator phase voltage")
        self.add_input("E_p_target", 0.0, units="V", desc="Target voltage")
        self.add_input("r_g", 0.0, units="m", desc="air gap radius")
        self.add_input("I_s", 0.0, units="A", desc="Stator current amplitude")
        self.add_input("l_s", 0.0, units="m", desc="stack length")
        self.add_input("p", desc="No of pole pairs")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("N_s", 0.0, desc="Number of turns in the stator winding")
        self.add_input("N_c", 0.0, desc="Number of coils")
        self.add_input("T_rated", 0.0, units="N*m", desc="Rated torque ")
        self.add_input("T_e", 0.0, units="N*m", desc="Electromagnetic torque ")
        self.add_output("E_p_ratio", 0.0, desc="Voltage constraint")
        self.add_output("torque_ratio", 0.0, desc="torque constraint")
        self.add_output("demag_mmf_ratio", 0.0, desc="torque constraint")
        self.add_input("k_wd", 0.0, desc="winding factor")
        self.add_input("B_r", 0, units="T", desc="remnant flux density")
        self.add_input("mu_0", np.pi * 4e-7, desc="permeability of free space")
        self.add_input("mu_r", 0.0, desc="relative permeability of magnet")
        self.add_input("H_c", 0.0, units="A/m", desc="coercivity")
        self.add_input("g", 0.0, units="m", desc="air gap length")
        self.add_input("k_sfil", 0.65, desc="slot fill factor")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        # Unpack inputs
        B_g = inputs["B_g"]
        B_symax = inputs["B_symax"]
        B_tmax = inputs["B_tmax"]
        B_rymax = inputs["B_rymax"]
        M_Fery = inputs["M_Fery"]
        M_Fesy = inputs["M_Fesy"]
        M_Fest = inputs["M_Fest"]
        P_Fe0h = inputs["P_Fe0h"]
        P_Fe0e = inputs["P_Fe0e"]
        mu_0 = inputs["mu_0"]
        B_r = inputs["B_r"]
        mu_r = inputs["mu_r"]
        H_c = inputs["H_c"]
        N_c = inputs["N_c"]
        P_Cu = inputs["P_Cu"]
        P_rated = inputs["P_rated"]
        P_Ftm = inputs["P_Ftm"]
        E_p_target = inputs["E_p_target"]
        T_rated = inputs["T_rated"]
        N_s = inputs["N_s"]
        k_wd = float(inputs["k_wd"])
        l_s = inputs["l_s"]
        N_nom = float(inputs["N_nom"])
        r_g = inputs["r_g"]
        g = inputs["g"]
        I_s = inputs["I_s"]
        p = inputs["p"]
        h_m = inputs["h_m"]
        T_e = inputs["T_e"]

        # Calculating Losses
        om_m = 2 * np.pi * N_nom / 60
        om_e = om_m * p

        # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys
        P_Hyd = M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator teeth
        P_Ftd = M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd

        P_Hyyr = M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftyr = M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator yoke
        P_Fernom = P_Hyyr + P_Ftyr

        # additional stray losses due to leakage flux
        P_ad = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd + P_Hyyr + P_Ftyr)

        outputs["Losses"] = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm + P_Fernom

        outputs["gen_eff"] = 1 - outputs["Losses"] / P_rated

        I_sc = 4.2 * I_s
        H_demag = -(N_c * I_sc + h_m * B_r / mu_0 * mu_r) / (h_m / mu_r + g)

        outputs["demag_mmf_ratio"] = H_demag / H_c

        outputs["E_p"] = E_p = np.sqrt(3) * N_s * l_s * r_g * k_wd * om_m * B_g * 0.707

        outputs["E_p_ratio"] = E_p / E_p_target
        outputs["torque_ratio"] = T_e / T_rated
