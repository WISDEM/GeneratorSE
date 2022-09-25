"""
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071
"""

import numpy as np
import openmdao.api as om


class LTS_active(om.ExplicitComponent):

    """Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator."""

    def setup(self):

        self.add_discrete_input("m", 6, desc="number of phases")
        self.add_discrete_input("q", 2, desc="slots per pole")
        self.add_input("b_s_tau_s", 0.45, desc="??")
        self.add_input("conductor_area", 1.8 * 1.2e-6, desc="??")

        self.add_input("D_a", 0.0, units="m", desc="armature diameter ")
        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("h_s", 0.0, units="m", desc="Slot height ")

        self.add_output("D_sc", 0.0, units="m", desc="field coil diameter ")

        # field coil parameters
        self.add_input("h_sc", 0.0, units="m", desc="SC coil height")
        # self.add_input("alpha_p", 0.0, desc="pole arc coefficient")
        self.add_input("alpha", 0.0, units="deg", desc="Start angle of field coil") 
        self.add_input("dalpha", 0.0, desc="Field coil fraction of available space")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        # self.add_input("I_sc", 0.0, units="A", desc="SC current ")
        self.add_input("N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_input("N_c", 0.0, desc="Number of turns per coil")
        self.add_input("p", 0.0, desc="Pole pairs ")
        self.add_output("p1", 0.0, desc="Pole pairs ")
        self.add_input("delta_em", 0.0, units="m", desc="airgap length ")
        self.add_input("Y", 0.0, desc="coil pitch")

        self.add_output("I_s", 0.0, units="A", desc="Generator output phase current")
        self.add_output("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_output("N_l", 0.0, desc="Number of layers of the SC field coil")
        # self.add_output("Dia_sc", 0.0, units="m", desc="field coil diameter")
        # self.add_input("Outer_width", 0.0, units="m", desc="Coil outer width")
        self.add_output("beta", 0.0, units="deg", desc="End angle of field coil")
        self.add_output("theta_p", 0.0, units="deg", desc="Pole pitch angle in degrees")

        # Material properties
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        self.add_input("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        self.add_input("rho_NbTi", 0.0, units="kg/(m**3)", desc="SC conductor mass density ")
        self.add_input("resisitivty_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        self.add_input("U_b", 0.0, units="V", desc="brush voltage ")

        self.add_input("P_rated", units="W", desc="Machine rating")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        # self.add_input("r_strand", 0.0, units="mm", desc="radius of the SC wire strand")
        # self.add_input("k_pf_sc", 0.0, units="mm", desc="packing factor for SC wires")
        self.add_input("J_s", 0.0, units="A/(mm*mm)", desc="Stator winding current density")
        # self.add_input("J_c", 0.0, units="A/(mm*mm)", desc="SC critical current density")

        # Field coil geometry
        self.add_output("a_m", 0.0, units="m", desc="Coil separation distance")
        self.add_output("W_sc", 0.0, units="m", desc="SC coil width")
        self.add_output("Outer_width", 0.0, units="m", desc="Coil outer width")
        self.add_output("field_coil_x", np.zeros(8), desc="Field coil points")
        self.add_output("field_coil_y", np.zeros(8), desc="Field coil points")
        self.add_output("field_coil_xlabel", np.zeros(2), desc="Field coil label points")
        self.add_output("field_coil_ylabel", np.zeros(2), desc="Field coil label points")

        # Magnetic loading
        self.add_output("tau_p", 0.0, units="m", desc="Pole pitch")
        # self.add_output("b_p", 0.0, units="m", desc="distance between positive and negative side of field coil")
        self.add_output("alpha_u", 0.0, units="rad", desc="slot angle")
        self.add_output("tau_v", 0.0, units="m", desc="Phase zone span")
        self.add_output("zones", 0.0, desc="Phase zones")
        self.add_output("delta", 0.0, units="rad", desc="short-pitch angle")
        self.add_output("k_p1", 0.0, desc="Pitch factor-fundamental harmonic")
        self.add_output("k_d1", 0.0, desc="Distribution factor-fundamental harmonic")
        self.add_output("k_w1", 0.0, desc="Winding factor- fundamental harmonic")
        # self.add_output("Iscn", 0.0, units="A", desc="SC current")
        # self.add_output("Iscp", 0.0, units="A", desc="SC current")
        # self.add_output("g", 0.0, units="m", desc="Air gap length")

        self.add_output("l_sc", 0.0, units="m", desc="SC coil length")
        self.add_output("l_eff_rotor", 0.0, units="m", desc="effective rotor length with end windings")
        self.add_output("l_eff_stator", 0.0, units="m", desc="effective stator length with end field windings")
        self.add_output("l_Cus", 0.0, units="m", desc="copper winding length")
        self.add_output("R_sc", 0.0, units="m", desc="Radius of the SC coils")

        # Stator design
        # self.add_output("h41", 0.0, units="m", desc="Bottom coil height")
        # self.add_output("h42", 0.0, units="m", desc="Top coil height")
        self.add_output("b_s", 0.0, units="m", desc="slot width")
        self.add_output("b_t", 0.0, units="m", desc="tooth width")
        # self.add_output("h_t", 0.0, units="m", desc="tooth height")
        self.add_output("A_Cuscalc", 0.0, units="mm**2", desc="Conductor cross-section")
        self.add_output("tau_s", 0.0, units="m", desc="Slot pitch ")

        # Electrical performance
        self.add_output("f", 0.0, units="Hz", desc="Generator output frequency")
        self.add_output("R_s", 0.0, units="ohm", desc="Stator resistance")
        self.add_output("A_1", 0.0, units="A/m", desc="Electrical loading")
        # self.add_output("J_actual", 0.0, units="A/m**2", desc="Current density")
        # self.add_output("T_e", 0.0, units="N*m", desc="Electromagnetic torque")
        # self.add_output("Torque_constraint", 0.0, units="N/(m*m)", desc="Shear stress contraint")

        # Objective functions
        self.add_output("K_rad", desc="Aspect ratio")
        self.add_output("Cu_losses", units="W", desc="Copper losses")
        self.add_output("P_add", units="W", desc="Additional losses")
        self.add_output("P_brushes", units="W", desc="brush losses")

        # Other parameters
        # self.add_output("R_out", 0.0, units="m", desc="Outer radius")
        self.add_output("Slots", 0.0, desc="Stator slots")
        self.add_output("Slot_aspect_ratio", 0.0, desc="Slot aspect ratio")
        self.add_output("y_Q", 0.0, desc="Slots per pole also pole pitch")

        # Mass Outputs
        self.add_output("mass_NbTi_racetrack", 0.0, units="kg", desc="SC conductor mass per racetrack")
        self.add_output("mass_NbTi", 0.0, units="kg", desc=" Total SC conductor mass")
        self.add_output("mass_copper", 0.0, units="kg", desc="Copper Mass")
        self.add_output("mass_iron", 0.0, units="kg", desc="Electrical Steel Mass")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        b_s_tau_s = float(inputs["b_s_tau_s"])
        conductor_area = float(inputs["conductor_area"])
        m = float(discrete_inputs["m"])
        q = float(discrete_inputs["q"])
        D_a = float(inputs["D_a"])
        l_s = float(inputs["l_s"])
        h_s = float(inputs["h_s"])
        h_sc = float(inputs["h_sc"])
        # alpha_p = float(inputs["alpha_p"])
        alpha_d = float(inputs["alpha"])
        alpha_r = np.deg2rad(alpha_d)
        dalpha = float(inputs["dalpha"])
        h_yr = float(inputs["h_yr"])
        N_sc = float(inputs["N_sc"])
        N_c = float(inputs["N_c"])
        p = float(inputs["p"])
        delta_em = float(inputs["delta_em"])
        Y = float(inputs["Y"])
        rho_Fe = float(inputs["rho_Fe"])
        rho_Copper = float(inputs["rho_Copper"])
        rho_NbTi = float(inputs["rho_NbTi"])
        resisitivty_Cu = float(inputs["resisitivty_Cu"])
        U_b = float(inputs["U_b"])
        P_rated = float(inputs["P_rated"])
        N_nom = float(inputs["N_nom"])
        J_s = float(inputs["J_s"])

        ###################################################### Electromagnetic design#############################################

        outputs["K_rad"] = l_s / (D_a)  # Aspect ratio
        outputs["D_sc"] = D_sc = D_a + 2 * delta_em
        outputs["R_sc"] = R_sc = D_sc * 0.5
        outputs["p1"] = p1 = p  # np.round(p)

        # Calculating pole pitch
        # r_s = 0.5 * D_a  # Stator outer radius # UNUSED
        outputs["tau_p"] = tau_p = np.pi * (R_sc + h_sc) / p1
        # outputs["b_p"] = alpha_p * tau_p

        # Calculating winding factor
        outputs["Slots"] = S = q * 2 * p1 * m

        outputs["tau_s"] = tau_s = np.pi * (D_a) / S  # Slot pitch
        outputs["alpha_u"] = alpha_u = p1 * 2 * np.pi / S  # slot angle
        outputs["tau_v"] = tau_p / m
        outputs["zones"] = 2 * p1 * m
        outputs["y_Q"] = y_Q = S / (2 * p1)  # coil span
        outputs["delta"] = Y * np.pi * 0.5 / y_Q  # short pitch angle     #coil span
        outputs["k_p1"] = k_p1 = np.sin(np.pi * 0.5 * Y / y_Q)
        outputs["k_d1"] = k_d1 = np.sin(q * alpha_u * 0.5) / (q * np.sin(alpha_u * 0.5))
        outputs["k_w1"] = k_p1 * k_d1

        # magnet width
        # alpha_p = np.pi / 2 * ratio # COMMENTING OUT BECAUSE ALPHA_P IS AN INPUT
        outputs["b_s"] = b_s = b_s_tau_s * tau_s
        outputs["b_t"] = tau_s - b_s  # slot width
        # UNUSED
        # gamma =  2 / np.pi * (
        #    np.atan(b_s * 0.5 / delta_em)
        #    - (2 * delta_em / b_s)
        #    * log(((1 + (b_s * 0.5 / delta_em) ** 2)) ** 0.5)
        # )

        # k_C = tau_s / (tau_s - gamma * (b_s))  # carter coefficient UNUSED
        # g_eff = k_C * delta_em # UNUSED

        # angular frequency in radians
        om_m = 2 * np.pi * N_nom / 60
        om_e = p1 * om_m
        outputs["f"] = om_e / 2 / np.pi  # outout frequency
        # outputs["N_s   N_s =        =   p1*Slot_pole*N_c*3                  #2*m*p*q
        # cos_theta_end = 1 - (b_s / (b_t + b_s) ** 2) ** 0.5
        # l_end = 4 * tau_s * 0.5 / cos_theta_end
        l_end = 10 * tau_s / 2 * np.tan(30 * np.pi / 180)

        outputs["l_eff_rotor"] = l_s + 2 * l_end
        # l_end                  =     np.sqrt(3)/6*(10*tau_s+)

        # Stator winding length ,cross-section and resistance
        outputs["l_Cus"] = l_Cus = 8 * l_end + 2 * l_s  # length of a turn
        z = S  # Number of coils
        A_slot = 0.5 * h_s * b_s
        # d_cu = 2 * np.sqrt(A_Cuscalc / pi) # UNUSED
        outputs["A_Cuscalc"] = A_Cuscalc = A_slot * 0.65 / N_c # factor of 0.5 for 2 layers, 0.65 is fill density
        outputs["I_s"] = I_s = 1e6 * J_s * A_Cuscalc # 1e6 to convert m^2 to mm^2
        # k_fill = A_slot / (2 * N_c * A_Cuscalc) # UNUSED

        outputs["N_s"] = N_s = N_c * z / (m)  # turns per phase int(N_c)
        outputs["R_s"] = R_s = resisitivty_Cu * (1 + 20 * 0.00393) * l_Cus * N_s / (2*A_Cuscalc)


        # print ("Resitance per phase:" ,R_s)
        # r_strand                =0.425e-3
        theta_p_r = tau_p / (R_sc + h_sc)
        outputs["theta_p"] = theta_p_d = np.rad2deg(theta_p_r)
        dalpha_d = dalpha * (0.5 * theta_p_d - alpha_d)
        outputs["beta"] = beta_d = alpha_d + dalpha_d
        beta_r = np.deg2rad(beta_d)
        # outputs["beta   beta = = (theta_p - 2*alpha)*0.5-2
        # random_degree = np.random.uniform(1.0, theta_p * 0.5 - alpha - 0.25)
        # random_degree = np.mean([1.0, float(theta_p) * 0.5 - float(alpha) - 0.25])
        # outputs["beta"] = beta = theta_p * 0.5 - random_degree

        # Field coil geometry prep for FEMM
        m_1 = -1 / (np.tan(theta_p_r * 0.5))  # slope of the tangent line
        x_coord = R_sc * np.cos(theta_p_r * 0.5)
        y_coord = R_sc * np.sin(theta_p_r * 0.5)
        c1 = y_coord - m_1 * x_coord

        if alpha_r <= 1:
            angle = alpha_r
        else:
            angle = np.tan(alpha_r)
        m_2 = angle
        c2 = 0

        m_3 = m_1  # tangent offset
        x_coord3 = (R_sc + h_sc) * np.cos(theta_p_r * 0.5)
        y_coord3 = (R_sc + h_sc) * np.sin(theta_p_r * 0.5)
        c3 = y_coord3 - m_3 * x_coord3

        mlabel = m_1

        x_label = (R_sc + h_sc * 0.5) * np.cos(theta_p_r * 0.5)
        y_label = (R_sc + h_sc * 0.5) * np.sin(theta_p_r * 0.5)
        clabel = y_label - mlabel * x_label

        mlabel2 = np.tan(alpha_r + (beta_r - alpha_r) * 0.5)
        clabel2 = 0

        mlabel3 = np.tan(theta_p_r - alpha_r - (beta_r - alpha_r) * 0.5)
        clabel3 = 0

        m_6 = np.tan(beta_r)
        c6 = 0

        x1 = (c1 - c2) / (m_2 - m_1)
        y1 = m_1 * x1 + c1

        m_4 = np.tan(theta_p_r * 0.5)

        c4 = y1 - m_4 * x1

        x2 = (c3 - c4) / (m_4 - m_3)
        y2 = m_4 * x2 + c4

        x4 = (c1 - c6) / (m_6 - m_1)
        y4 = m_1 * x4 + c1

        m_5 = np.tan(theta_p_r * 0.5)
        c5 = y4 - m_5 * x4

        x3 = (c3 - c5) / (m_5 - m_3)
        y3 = m_3 * x3 + c3

        m_7 = np.tan(theta_p_r - alpha_r)
        c7 = 0

        x5 = (c1 - c7) / (m_7 - m_1)
        y5 = m_1 * x5 + c1

        m_8 = m_4
        c8 = y5 - m_8 * x5

        x6 = (c3 - c8) / (m_8 - m_3)
        y6 = m_3 * x6 + c3

        m_9 = np.tan(theta_p_r - beta_r)
        c9 = 0

        x8 = (c1 - c9) / (m_9 - m_1)
        y8 = m_1 * x8 + c1

        m_10 = m_5
        c10 = y8 - m_10 * x8

        x7 = (c3 - c10) / (m_10 - m_3)
        y7 = m_3 * x7 + c3

        xlabel1 = (clabel - clabel2) / (mlabel2 - mlabel)
        ylabel1 = mlabel * xlabel1 + clabel

        xlabel2 = (clabel - clabel3) / (mlabel3 - mlabel)
        ylabel2 = mlabel * xlabel2 + clabel

        outputs["a_m"] = a_m = 2 * (
            np.sqrt((R_sc * np.cos(theta_p_r * 0.5) - x4) ** 2 + ((R_sc * np.sin(theta_p_r * 0.5) - y4) ** 2))
        )
        outputs["W_sc"] = W_sc = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
        outputs["Outer_width"] = a_m + 2 * W_sc
        outputs["field_coil_x"] = np.r_[x1, x2, x3, x4, x5, x6, x7, x8]
        outputs["field_coil_y"] = np.r_[y1, y2, y3, y4, y5, y6, y7, y8]
        outputs["field_coil_xlabel"] = np.r_[xlabel1, xlabel2]
        outputs["field_coil_ylabel"] = np.r_[ylabel1, ylabel2]

        # N_sc = k_pf_sc*W_sc*h_sc/np.pi*r_strand**2
        outputs["l_sc"] = l_sc = N_sc * (2 * l_s + np.pi * (a_m + W_sc))

        outputs["l_eff_stator"] = l_s + (a_m + W_sc)

        outputs["Slot_aspect_ratio"] = h_s / b_s

        # Calculating stator current and electrical loading
        # +I_s              = sqrt(Z**2+(((E_p-G**0.5)/(om_e*L_s)**2)**2))
        # Calculating volumes and masses
        # V_Cus 	                    =   m*L_Cus*(A_Cuscalc*(10**-6))     # copper volume
        # outputs["h_t   h_t =            =   (h_s+h_1+h_0)
        # volume of iron in stator tooth
        V_Fery = 0.25 * np.pi * l_s * ((D_a - 2 * h_s) ** 2 - (D_a - 2 * h_s - 2 * h_yr) ** 2)
        # outputs["Copper		 Copper =    =   V_Cus*rho_Copper
        outputs["mass_iron"] = V_Fery * rho_Fe  # Mass of stator yoke

        outputs["N_l"] = h_sc / (1.2e-3)  # round later!

        outputs["mass_NbTi_racetrack"] = mass_NbTi = l_sc * conductor_area * rho_NbTi
        outputs["mass_NbTi"] = p1 * mass_NbTi
        V_Cus = m * l_Cus * N_s * 2 * A_Cuscalc # copper volume, factor of 2 for 2 layers
        outputs["mass_copper"] = V_Cus * rho_Copper

        outputs["A_1"] = (2 * I_s * N_s * m) / (np.pi * (D_a))

        outputs["Cu_losses"] = 0.5 * m * I_s ** 2 * R_s
        outputs["P_add"] = 0.01 * P_rated
        outputs["P_brushes"] = 6 * U_b * I_s * 0.707



class Results(om.ExplicitComponent):
    def setup(self):
        self.add_input("K_h", 2.0, desc="??")
        self.add_input("K_e", 0.5, desc="??")

        #self.add_input("I_sc_out", 0.0, units="A", desc="SC current ")
        self.add_input("N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_input("N_l", 0.0, desc="Number of layers of the SC field coil")
        self.add_input("D_a", 0.0, units="m", desc="Armature diameter ")
        self.add_input("k_w1", 0.0, desc="Winding factor- fundamental harmonic")
        self.add_input("B_rymax", 0.0, units='T', desc="Peak Rotor yoke flux density")
        self.add_input("B_g", 0.0, units='T', desc="Peak air gap flux density ")
        self.add_input("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("p1", 0.0, desc="Pole pairs ")
        self.add_input("mass_iron", 0.0, units="kg", desc="Electrical Steel Mass")
        self.add_input("T_rated", 0.0, units="N*m", desc="Rated torque ")
        self.add_input("Torque_actual", 0.0, units="N*m", desc="Shear stress actual")
        self.add_input("P_rated", units="W", desc="Machine rating")
        self.add_input("Cu_losses", units="W", desc="Copper losses")
        self.add_input("P_add", units="W", desc="Additional losses")
        self.add_input("P_brushes", units="W", desc="brush losses")
        self.add_input("l_s", 0.0, units="m", desc="Stator core length")
        self.add_input("E_p_target", 0.0, units="V", desc="target terminal voltage")
        # self.add_output("con_I_sc", 0.0, units="A/(mm*mm)", desc="SC current ")
        # self.add_output("con_N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_output("E_p", 0.0, units="V", desc="terminal voltage")
        self.add_output("N_sc_layer", 0.0, desc="Number of turns per layer")
        self.add_output("P_Fe", units="W", desc="Iron losses")
        self.add_output("P_Losses", units="W", desc="Total power losses")
        self.add_output("gen_eff", desc="Generator efficiency")
        self.add_output("torque_ratio", desc="Whether torque meets requirement")
        self.add_output("E_p_ratio", desc="Whether terminal voltage meets requirement")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Unpack inputs
        K_h = inputs["K_h"]
        K_e = inputs["K_e"]
        N_sc = float(inputs["N_sc"])
        N_l = float(inputs["N_l"])
        D_a = float(inputs["D_a"])
        k_w1 = float(inputs["k_w1"])
        B_rymax = float(inputs["B_rymax"])
        B_g = float(inputs["B_g"])
        N_s = float(inputs["N_s"])
        N_nom = float(inputs["N_nom"])
        p1 = float(inputs["p1"])
        Iron = float(inputs["mass_iron"])
        P_rated = float(inputs["P_rated"])
        Cu_losses = float(inputs["Cu_losses"])
        P_add = float(inputs["P_add"])
        P_brushes = float(inputs["P_brushes"])
        l_s = float(inputs["l_s"])
        T_rated = float(inputs["T_rated"])
        T_actual = float(inputs["Torque_actual"])
        E_p_target = float(inputs["E_p_target"])
        # outputs["con_N_sc"] = con_N_sc = N_sc - N_sc_out
        # outputs["con_I_sc"] = con_I_sc = I_sc - I_sc_out
        outputs["N_sc_layer"] = int(N_sc / N_l)

        # Calculating  voltage per phase
        om_m = 2 * np.pi * N_nom / 60
        outputs["E_p"] = E_p = l_s * (D_a * 0.5 * k_w1 * B_g * om_m * N_s) * np.sqrt(1.5) * 1.12253

        # print ("Voltage and lengths are:",outputs["E_p,l_s )

        f_e = 2 * p1 * N_nom / 120
        outputs["P_Fe"] = P_Fe = (
            2 * K_h * (f_e / 60) * (B_rymax / 1.5) ** 2 + 2 * K_e * (f_e / 60) ** 2 * (B_rymax / 1.5) ** 2
        ) * Iron

        outputs["P_Losses"] = P_Losses = Cu_losses + P_Fe + P_add + P_brushes
        outputs["gen_eff"] = 1 - P_Losses / P_rated
        outputs["torque_ratio"] = T_actual / T_rated
        outputs["E_p_ratio"] = E_p / E_p_target

        # print(outputs["gen_eff"],outputs["torque_ratio"],outputs["E_p_ratio"])
