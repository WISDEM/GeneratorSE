# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:45:08 2022

@author: lsethura
"""

"""
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071
"""

import numpy as np
import openmdao.api as om


class Results(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("debug_prints", default=False)

    def setup(self):
        self.add_input("B_g", 0.0, units="T", desc="Peak air gap flux density ")
        self.add_input("Ind", 0.0, desc="Inductance ")
        self.add_input("I_s", 0.0, units="A", desc="Stator current ")
        self.add_input("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("R_s", 0.0, desc="resistance per phase")
        self.add_input("T_rated", 0.0, units="N*m", desc="Rated torque ")
        self.add_input("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_input("P_rated", units="W", desc="Machine rating")
        self.add_input("pp", 0.0, desc="pole pairs")
        self.add_input("P_Cu", units="W", desc="Copper losses")
        self.add_output("P_Ftm", units="W", desc="magnet losses")
        self.add_output("P_Fe", units="W", desc="Iron losses")
        self.add_input("l_s", 0.0, units="m", desc="Stator core length")
        self.add_input("D_a", 0.0, units="m", desc="Armature diameter")
        self.add_input("E_p_target", 0.0, units="V", desc="target terminal voltage")
        self.add_input("h_m", 0.0, units="m", desc="magnet height")
        self.add_input("l_m", 0.0, units="m", desc="magnet length ")
        self.add_input("B_rymax", 0.0, units="T", desc="Peak Rotor yoke flux density")
        self.add_input("B_smax", 0.0, units="T", desc="Peak flux density in the field coils")
        self.add_input("M_Fes", 0.0, units="kg", desc="mstator iron mass ")
        self.add_input("M_Fer", 0.0, units="kg", desc="rotor iron mass ")
        self.add_output("mass_PM", 0.0, units="kg", desc="Iron mass")
        #self.add_input("rho_Fes", 0.0, units="kg/(m**3)", desc="Structural Steel density")
        self.add_input("rho_PM", 0.0, units="kg/(m**3)", desc="Magnet density ")

        self.add_input("P_Fe0h", 4.0, desc="specific hysteresis losses W/kg @ 1.5 T")
        self.add_input("P_Fe0e", 1.0, desc="specific eddy losses W/kg @ 1.5 T")
        #self.add_output("Mass", 0.0, units="kg", desc="Structural Mass")

        self.add_output("E_p", 0.0, units="V", desc="terminal voltage")

        self.add_output("P_Losses", units="W", desc="Total power losses")
        self.add_output("gen_eff", desc="Generator efficiency")
        self.add_output("torque_ratio", desc="Whether torque meets requirement")
        self.add_output("E_p_ratio", desc="Whether terminal voltage meets requirement")
        self.add_input("k_w", 0.0, desc="winding factor ")
        self.add_output("P_ad", 0, units="W", desc="Additional losses")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Unpack inputs
        B_g = float( inputs["B_g"] )
        Ind = float(inputs["Ind"])
        I_s = float(inputs["I_s"])
        N_s = float( inputs["N_s"] )
        N_nom = float( inputs["N_nom"] )
        T_rated = float( inputs["T_rated"] )
        T_e = float( inputs["T_e"] )
        P_rated = float( inputs["P_rated"] )
        pp = float( inputs["pp"] )
        P_Cu = float( inputs["P_Cu"] )
        l_s = float( inputs["l_s"] )
        D_a = float( inputs["D_a"] )
        E_p_target = float( inputs["E_p_target"] )
        h_m = float( inputs["h_m"] )
        l_m = float( inputs["l_m"] )
        B_rymax = float( inputs["B_rymax"] )
        B_smax = float( inputs["B_smax"] )
        M_Fes = float( inputs["M_Fes"] )
        M_Fer = float( inputs["M_Fer"] )
        rho_PM = float( inputs["rho_PM"] )
        P_Fe0h = float( inputs["P_Fe0h"] )
        P_Fe0e = float( inputs["P_Fe0e"] )
        k_w = float( inputs["k_w"] )
        R_s = float(inputs["R_s"])

        # Calculating  voltage per phase
        om_m = 2 * np.pi * N_nom / 60
        om_e = om_m * pp
        om_e2 = om_e / (2 * np.pi * 60)

        #outputs["E_p"] = E_p = l_s * (D_a * 0.5 * k_w * B_g * om_m * N_s) * np.sqrt(1.5)
        outputs["E_p"] = E_p = (l_s * (D_a * 0.5 * k_w * B_g * om_m * N_s) * np.sqrt(1.5) +
                                I_s * np.sqrt(0.5) * np.sqrt(R_s**2 + (om_e/np.pi * Ind)**2) )
        outputs["torque_ratio"] = T_e / T_rated
        outputs["E_p_ratio"] = E_p / E_p_target

        outputs["mass_PM"] = 4 * pp * l_m * h_m * l_s * rho_PM

        # specific magnet loss
        pFtm = 300
        outputs["P_Ftm"] = P_Ftm = pFtm * 2 * pp * l_m * l_s

        # Iron Losses ( from Hysteresis and eddy currents)
        # Hysteresis losses in stator
        P_Hyys = M_Fes * (B_smax / 1.5) ** 2 * (P_Fe0h * om_e2)

        # Eddy losses in stator
        P_Ftys = M_Fes * (B_smax / 1.5) ** 2 * (P_Fe0e * om_e2 ** 2)
        P_Fesnom = P_Hyys + P_Ftys

        # Hysteresis losses in rotor
        P_Hyyr = M_Fer * (B_rymax / 1.5) ** 2 * (P_Fe0h * om_e2)

        # Eddy losses in rotor
        P_Ftyr = M_Fer * (B_rymax / 1.5) ** 2 * (P_Fe0e * om_e2 ** 2)
        P_Fernom = P_Hyyr + P_Ftyr

        outputs["P_Fe"] = P_Fe = P_Fesnom + P_Fernom
        outputs["P_ad"] = P_ad = 0.2 * P_Fe
        outputs["P_Losses"] = P_Losses = P_Cu + P_Fe + P_ad + P_Ftm

        outputs["gen_eff"] = 1 - P_Losses / P_rated
        # additional stray losses due to leakage flux


        if self.options['debug_prints']:
            print('torque_ratio: ', outputs["torque_ratio"])
            print("losses:", P_Cu, P_Fe, P_ad, P_Ftm)
            print('gen_eff: ', outputs["gen_eff"])
            print('E_p_ratio: ', outputs["E_p_ratio"])
            print('B_rymax: ', B_rymax)
            print('B_smax: ', B_smax)
            print('*******')
