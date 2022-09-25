"""
Created on Fri Dec 31 12:28:24 2021

@author: lsethura
"""

import femm
import numpy as np
import openmdao.api as om
from generatorse.common.femm_util import myopen, cleanup_femm_files

mu0 = 4 * np.pi * 1e-7

def bad_inputs(outputs):
    outputs["B_g"] = 7
    outputs["B_coil_max"] = 1.0
    outputs["B_rymax"] = 5
    outputs["Torque_actual"] = 1.0e6  # 50e6
    outputs["Sigma_shear"] = 1.0e3  # 300000
    outputs["Sigma_normal"] = 1.0e3  # 200000
    return outputs


def run_post_process(D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n):
    # After looking at the femm results, this function post-processes them, no electrical load condition
    R_a = 0.5 * D_a
    theta_p_d = np.rad2deg(theta_p_r)

    femm.mo_addcontour(
        (R_a + (radius_sc - R_a) * 0.5) * np.cos(0), (R_a + (radius_sc - R_a) * 0.5) * np.sin(0)
    )
    femm.mo_addcontour(
        (R_a + (radius_sc - R_a) * 0.5) * np.cos(theta_p_r),
        (R_a + (radius_sc - R_a) * 0.5) * np.sin(theta_p_r),
    )
    femm.mo_bendcontour(theta_p_d, 0.25)
    femm.mo_makeplot(1, 1500, "gap_" + str(n) + ".csv", 1)
    femm.mo_makeplot(2, 1000, "B_r_normal" + str(n) + ".csv", 1)
    femm.mo_makeplot(3, 1000, "B_t_normal" + str(n) + ".csv", 1)
    femm.mo_clearcontour()

    # Approximate peak field
    B_coil_max = 0.0
    for k in np.linspace(slot_radius, radius_sc+2*h_sc, 20):
        femm.mo_addcontour(k * np.cos(0), k * np.sin(0))
        femm.mo_addcontour(k * np.cos(theta_p_r), k * np.sin(theta_p_r))
        femm.mo_bendcontour(theta_p_d, 0.25)
        femm.mo_makeplot(1, 100, "B_mag.csv", 1)
        femm.mo_clearcontour()
        B_mag = np.loadtxt("B_mag.csv")
        B_coil_max = np.maximum(B_coil_max, B_mag.max())

    radius_eps = 1e-3
    femm.mo_addcontour((slot_radius - radius_eps) * np.cos(0), (slot_radius - radius_eps) * np.sin(0))
    femm.mo_addcontour((slot_radius - radius_eps) * np.cos(theta_p_r), (slot_radius - radius_eps) * np.sin(theta_p_r))
    femm.mo_bendcontour(theta_p_d, 0.25)

    femm.mo_makeplot(1, 1500, "core_" + str(n) + ".csv", 1)
    femm.mo_clearcontour()

    B_rymax = np.loadtxt("core_" + str(n) + ".csv")[:, 1].max()
    B_g_peak = np.loadtxt("gap_" + str(n) + ".csv")[:, 1].max()
    B_r_normal = np.loadtxt("B_r_normal" + str(n) + ".csv")
    B_t_normal = np.loadtxt("B_t_normal" + str(n) + ".csv")

    circ = B_r_normal[-1, 0]
    force = np.trapz(B_r_normal[:, 1] ** 2 - B_t_normal[:, 1] ** 2, B_r_normal[:, 0])
    sigma_n = abs(force / (2*mu0)) / circ
    # B_g_peak: peak air-gap flux density
    # B_rymax: peak rotor yoke flux density
    # B_coil_max: peak flux density in the superconducting coil
    # sigma_n: max normal stress
    return B_g_peak, B_rymax, B_coil_max, sigma_n


def B_r_B_t(Theta_elec, D_a, l_s, p1, delta_em, theta_p_r, I_s,
            theta_b_t, theta_b_s, layer_1, layer_2, Y_q, N_c, tau_p):
    # This function loads the machine with electrical currents
    theta_p_d = np.rad2deg(theta_p_r)
    R_a = 0.5 * D_a

    #myopen()
    #femm.opendocument("coil_design_new.fem")
    femm.mi_modifycircprop("A+", 1, I_s * np.sin(0))
    femm.mi_modifycircprop("D+", 1, I_s * np.sin(1 * np.pi / 6))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(-4 * np.pi / 6))
    femm.mi_modifycircprop("F-", 1, -I_s * np.sin(-3 * np.pi / 6))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(-8 * np.pi / 6))
    femm.mi_modifycircprop("E+", 1, I_s * np.sin(-7 * np.pi / 6))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(0))
    femm.mi_modifycircprop("D-", 1, -I_s * np.sin(np.pi / 6))
    femm.mi_saveas("coil_design_new_I1.fem")
    femm.mi_analyze()
    femm.mi_loadsolution()

    femm.mo_addcontour((R_a + delta_em * 0.5) * np.cos(0), (R_a + delta_em * 0.5) * np.sin(0))
    femm.mo_addcontour((R_a + delta_em * 0.5) * np.cos(theta_p_r), (R_a + delta_em * 0.5) * np.sin(theta_p_r))
    femm.mo_bendcontour(theta_p_d, 0.25)
    #sigma_t_sec, _ = femm.mo_lineintegral(3)
    torque_sec, _ = femm.mo_lineintegral(4)
    #sigma_t = sigma_t_sec*2*np.pi/theta_p_r
    torque = torque_sec*2*np.pi/theta_p_r
    sigma_t = torque / (2*np.pi * (R_a + delta_em * 0.5)**2 * l_s)
    #femm.mo_makeplot(2, 1000, "B_r_1.csv", 1)
    #femm.mo_makeplot(3, 1000, "B_t_1.csv", 1)
    #B_r_1 = np.loadtxt("B_r_1.csv")
    #B_t_1 = np.loadtxt("B_t_1.csv")
    #circ = B_r_1[-1, 0]
    femm.mo_clearcontour()
    #femm.mo_selectblock((R_a + delta_em * 0.5) * np.cos(0.5*theta_p_r), (R_a + delta_em * 0.5) * np.sin(0.5*theta_p_r))
    #temp = femm.mo_blockintegral(22)
    femm.mo_close()
    #myopen()
    #femm.opendocument("coil_design_new_I1.fem")

    '''
    Phases = ["D+", "C-", "F-", "B+", "E+", "A-", "D-"]
    # Phases = ["F+", "B+", "E+", "A+", "D+", "C+", "F-"]
    # N_c_a1 = [2 * N_c, 4 * N_c, 4 * N_c, 4 * N_c, 4 * N_c, 4 * N_c, 2 * N_c]

    pitch = 1
    count = 0
    angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
    delta_theta = theta_b_t + theta_b_s
    for pitch in range(1, int(np.ceil(Y_q)), 2):
        try:
            femm.mi_selectlabel(
                layer_2 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
                layer_2 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
            )
            femm.mi_selectlabel(
                layer_2 * np.cos(angle_r + pitch * delta_theta), layer_2 * np.sin(angle_r + pitch * delta_theta)
            )
            femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c)
            # femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c_a1[count])
            femm.mi_clearselected()
            count = count + 1
        except:
            continue

    count = 0
    angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
    delta_theta = theta_b_t + theta_b_s
    for pitch in range(1, int(np.ceil(Y_q)), 2):
        try:
            femm.mi_selectlabel(
                layer_1 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
                layer_1 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
            )
            femm.mi_selectlabel(
                layer_1 * np.cos(angle_r + pitch * delta_theta), layer_1 * np.sin(angle_r + pitch * delta_theta)
            )
            femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c)
            # femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c_a1[count+1])
            femm.mi_clearselected()
            count = count + 1
        except:
            continue
    Theta_elec=-Theta_elec
    femm.mi_modifycircprop("D+", 1, I_s * np.sin(Theta_elec + np.pi / 6))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(Theta_elec - 4 * np.pi / 6))
    femm.mi_modifycircprop("F-", 1, -I_s * np.sin(Theta_elec - 3 * np.pi / 6))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(Theta_elec - 8 * np.pi / 6))
    femm.mi_modifycircprop("E+", 1, I_s * np.sin(Theta_elec - 7 * np.pi / 6))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(Theta_elec))
    femm.mi_modifycircprop("D-", 1, -I_s * np.sin(Theta_elec + np.pi / 6))

    femm.mi_saveas("coil_design_new_I2.fem")
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_addcontour((R_a + delta_em * 0.5) * np.cos(0), (R_a + delta_em * 0.5) * np.sin(0))
    femm.mo_addcontour((R_a + delta_em * 0.5) * np.cos(theta_p_r), (R_a + delta_em * 0.5) * np.sin(theta_p_r))
    femm.mo_bendcontour(theta_p_d, 0.25)
    femm.mo_makeplot(2, 1000, "B_r_2.csv", 1)
    femm.mo_makeplot(3, 1000, "B_t_2.csv", 1)
    B_r_2 = np.loadtxt("B_r_2.csv")
    B_t_2 = np.loadtxt("B_t_2.csv")
    femm.mo_clearcontour()
    femm.mo_close()
    '''
    #B_r_2 = B_r_1
    #B_t_2 = B_t_1

    #force = np.array(
    #    [np.trapz(B_r_1[:, 1] * B_t_1[:, 1], B_r_1[:, 0]), np.trapz(B_r_2[:, 1] * B_t_2[:, 1], B_r_2[:, 0])]
    #)
    #sigma_t = abs(force / mu0) / circ
    #torque = np.pi / 2 * sigma_t * D_a ** 2 * l_s
    # Air gap electro-magnetic torque for the full machine
    # Average shear stress for the full machine
    #return torque.mean(), sigma_t.mean()
    return np.abs(torque), np.abs(sigma_t)


class FEMM_Geometry(om.ExplicitComponent):
    # This openmdao component builds the geometry of one sector of the LTS generator in pyfemm and runs the analysis

    def setup(self):

        # Discrete inputs
        self.add_discrete_input("q", 2, desc="slots_per_pole")
        self.add_discrete_input("m", 6, desc="number of phases")

        # Float nputs
        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("alpha", 0.0, units="deg", desc="Start angle of field coil")
        self.add_input("beta", 0.0, units="deg", desc="End angle of field coil")
        self.add_input("h_sc", 0.0, units="m", desc="SC coil height")
        self.add_input("p1", 0.0, desc="Pole pairs ")
        self.add_input("D_a", 0.0, units="m", desc="armature diameter ")
        self.add_input("h_s", 0.0, units="m", desc="Slot height ")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("D_sc", 0.0, units="m", desc="field coil diameter ")
        self.add_input("I_sc_in", 0.0, units="A", desc="Initial current in the superconducting coils")
        self.add_output("I_sc_out", 0.0, units="A", desc="Actual current in the superconducting coils")
        self.add_input("N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_input("N_c", 0.0, desc="Number of turns per coil")
        self.add_input("load_margin", 0.0, desc="SC coil current loading margin %")
        self.add_input("N_nom", 0.0,units='rpm', desc="Number of turns per coil")
        self.add_input("delta_em", 0.0, units="m", desc="airgap length ")
        self.add_input("I_s", 0.0, units="A", desc="Generator output phase current")
        self.add_input("con_angle", 0.0, units="deg", desc="Geometry constraint status")
        self.add_input("a_m", 0.0, units="m", desc="Coil separation distance")
        self.add_input("W_sc", 0.0, units="m", desc="SC coil width")
        self.add_input("Outer_width", 0.0, units="m", desc="Coil outer width")
        self.add_input("field_coil_x", np.zeros(8), desc="Field coil points")
        self.add_input("field_coil_y", np.zeros(8), desc="Field coil points")
        self.add_input("field_coil_xlabel", np.zeros(2), desc="Field coil label points")
        self.add_input("field_coil_ylabel", np.zeros(2), desc="Field coil label points")
        self.add_input("Slots", 0.0, desc="Stator slots")
        self.add_input("y_Q", 0.0, desc="Slots per pole also pole pitch")

        # Outputs
        self.add_output("B_g", 0.0, units='T',desc="Peak air gap flux density ")
        self.add_output("B_rymax", 0.0, units='T', desc="Peak Rotor yoke flux density")
        self.add_output("B_coil_max", 0.0, units='T', desc="Peak flux density in the field coils")
        self.add_output("Torque_actual", 0.0, units="N*m", desc="Shear stress actual")
        self.add_output("Sigma_shear", 0.0, units="Pa", desc="Shear stress")
        self.add_output("Sigma_normal", 0.0, units="Pa", desc="Normal stress")
        self.add_output("margin_I_c", 0.0, units="A", desc="Critical current margin")
        #self.add_output("Critical_current_ratio", 0.0, units="A", desc="Ratio of critical to max current")
        self.add_output(
            "Coil_max_ratio",
            0.0,
            desc="Ratio of actual to critical coil flux density, usually constrained to be smaller than 1",
        )
        self.add_output("constr_B_g_coil", 0.0, desc="Ratio of B_g to B_coil_max, should be <1.0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        l_s = float(inputs["l_s"])
        alpha_d = float(inputs["alpha"])
        beta_d = float(inputs["beta"])
        alpha_r = np.deg2rad(alpha_d)
        beta_r = np.deg2rad(beta_d)
        h_sc = float(inputs["h_sc"])
        p1 = float(inputs["p1"])
        D_a = float(inputs["D_a"])
        R_a = 0.5 * D_a
        h_yr = float(inputs["h_yr"])
        h_s = float(inputs["h_s"])
        q = discrete_inputs["q"]
        m = discrete_inputs["m"]
        D_sc = float(inputs["D_sc"])
        N_sc = float(inputs["N_sc"])
        I_sc = float(inputs["I_sc_in"])
        N_c = float(inputs["N_c"])
        delta_em = float(inputs["delta_em"])
        I_s = float(inputs["I_s"])
        Slots = float(inputs["Slots"])
        Y_q = float(inputs["y_Q"])
        N_nom = float(inputs["N_nom"])
        radius_sc = D_sc / 2
        tau_p = np.pi * (radius_sc * 2 + 2 * h_sc) / (2 * p1)
        theta_p_r = tau_p / (radius_sc + h_sc)
        theta_p_d = np.rad2deg(theta_p_r)
        x1, x2, x3, x4, x5, x6, x7, x8 = inputs["field_coil_x"]
        y1, y2, y3, y4, y5, y6, y7, y8 = inputs["field_coil_y"]
        xlabel1, xlabel2 = inputs["field_coil_xlabel"]
        ylabel1, ylabel2 = inputs["field_coil_ylabel"]
        load_margin=float(inputs["load_margin"])

        # Build the geometry of the generator sector
        if (alpha_d <= 0) or (float(inputs["con_angle"]) < 0):
            outputs = bad_inputs(outputs)
        else:
            slot_radius = R_a - h_s
            yoke_radius = slot_radius - h_yr
            h42 = R_a - 0.5 * h_s
            Slots_pp = float(q * m)
            tau_s = np.pi * D_a / Slots

            # bs_taus = 0.45 #UNUSED
            b_s = 0.45 * tau_s
            b_t = tau_s - b_s
            theta_b_s = b_s / (D_a * 0.5)
            theta_tau_s = tau_s / (D_a * 0.5) # UNUSED
            theta_b_t = (b_t) / (D_a * 0.5)
            tau_p = np.pi * (radius_sc * 2 + 2 * h_sc) / (2 * p1)
            Current = 0
            theta_p_d = np.rad2deg(theta_p_r)
            # alphap_taup_r = theta_p_r - 2 * beta_r
            # alphap_taup_angle_r = beta_r + alphap_taup_r - alpha_r
            # alphap_taup_angle_d = np.rad2deg(alphap_taup_angle_r)

            # Draw the field coil
            myopen()
            femm.newdocument(0)
            femm.mi_probdef(0, "meters", "planar", 1.0e-8, l_s, 30)

            femm.mi_addnode(x1, y1)
            femm.mi_addnode(x2, y2)
            femm.mi_addnode(x3, y3)
            femm.mi_addnode(x4, y4)
            femm.mi_addnode(x5, y5)
            femm.mi_addnode(x6, y6)
            femm.mi_addnode(x7, y7)
            femm.mi_addnode(x8, y8)
            femm.mi_addsegment(x1, y1, x2, y2)
            femm.mi_addsegment(x2, y2, x3, y3)
            femm.mi_addsegment(x3, y3, x4, y4)
            femm.mi_addsegment(x4, y4, x1, y1)
            femm.mi_addsegment(x5, y5, x6, y6)
            femm.mi_addsegment(x8, y8, x7, y7)
            femm.mi_addsegment(x5, y5, x8, y8)
            femm.mi_addsegment(x6, y6, x7, y7)
            # femm.mi_addnode(0, 0)

            # Draw the stator slots and Stator coils
            femm.mi_addnode(R_a * np.cos(0), R_a * np.sin(0))
            femm.mi_selectnode(R_a * np.cos(0), R_a * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_addnode(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_selectnode(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_addnode(h42 * np.cos(0), h42 * np.sin(0))
            femm.mi_selectnode(h42 * np.cos(0), h42 * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_addsegment(R_a * np.cos(0), R_a * np.sin(0), h42 * np.cos(0), h42 * np.sin(0))
            femm.mi_addsegment(h42 * np.cos(0), h42 * np.sin(0), slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_selectsegment(h42 * np.cos(0), h42 * np.sin(0))

            femm.mi_selectsegment(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_setgroup(1)

            femm.mi_addarc(
                slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5), slot_radius, 0, 5, 1
            )
            femm.mi_selectarcsegment(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_selectsegment(R_a * np.cos(0), R_a * np.sin(0))
            femm.mi_setgroup(1)

            femm.mi_addnode(D_a * 0.5 * np.cos(theta_b_t * 0.5), D_a * 0.5 * np.sin(theta_b_t * 0.5))
            femm.mi_selectnode(D_a * 0.5 * np.cos(theta_b_t * 0.5), D_a * 0.5 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            femm.mi_addnode(slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            femm.mi_addnode(h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            femm.mi_addsegment(
                D_a * 0.5 * np.cos(theta_b_t * 0.5),
                D_a * 0.5 * np.sin(theta_b_t * 0.5),
                h42 * np.cos(theta_b_t * 0.5),
                h42 * np.sin(theta_b_t * 0.5),
            )

            femm.mi_addsegment(
                h42 * np.cos(theta_b_t * 0.5),
                h42 * np.sin(theta_b_t * 0.5),
                slot_radius * np.cos(theta_b_t * 0.5),
                slot_radius * np.sin(theta_b_t * 0.5),
            )
            femm.mi_selectarcsegment(h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            femm.mi_selectsegment(h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            femm.mi_selectsegment(D_a * 0.5 * np.cos(theta_b_t * 0.5), D_a * 0.5 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            femm.mi_selectsegment(slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)

            theta_b_s_new = (b_t * 0.5 + b_s) / (D_a * 0.5)
            femm.mi_addnode(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_selectnode(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addnode(h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new))
            femm.mi_selectnode(h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addsegment(
                D_a * 0.5 * np.cos(theta_b_s_new),
                D_a * 0.5 * np.sin(theta_b_s_new),
                h42 * np.cos(theta_b_s_new),
                h42 * np.sin(theta_b_s_new),
            )
            femm.mi_selectsegment(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addnode(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_selectnode(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addsegment(
                slot_radius * np.cos(theta_b_s_new),
                slot_radius * np.sin(theta_b_s_new),
                h42 * np.cos(theta_b_s_new),
                h42 * np.sin(theta_b_s_new),
            )
            femm.mi_selectsegment(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addarc(
                D_a * 0.5 * np.cos(theta_b_s_new),
                D_a * 0.5 * np.sin(theta_b_s_new),
                D_a * 0.5 * np.cos(theta_b_t * 0.5),
                D_a * 0.5 * np.sin(theta_b_t * 0.5),
                5,
                1,
            )
            femm.mi_selectarcsegment(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addarc(
                h42 * np.cos(theta_b_s_new),
                h42 * np.sin(theta_b_s_new),
                h42 * np.cos(theta_b_t * 0.5),
                h42 * np.sin(theta_b_t * 0.5),
                5,
                1,
            )
            femm.mi_selectarcsegment(h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_addarc(
                slot_radius * np.cos(theta_b_t * 0.5),
                slot_radius * np.sin(theta_b_t * 0.5),
                slot_radius * np.cos(theta_b_s_new),
                slot_radius * np.sin(theta_b_s_new),
                2,
                1,
            )
            femm.mi_selectarcsegment(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)

            femm.mi_selectgroup(2)
            angle_d = np.rad2deg(tau_s / (D_a * 0.5))
            femm.mi_copyrotate(0, 0, angle_d, Slots_pp - 1)

            # b_t_new = slot_radius * b_t / ((D_a * 0.5)) #UNUSED

            femm.mi_addnode(yoke_radius * np.cos(0), yoke_radius * np.sin(0))

            # femm.mi_addnode(yoke_radius*np.cos(alpha_r),yoke_radius*np.sin(alpha_r))
            femm.mi_addnode(yoke_radius * np.cos(theta_p_r), yoke_radius * np.sin(theta_p_r))

            # femm.mi_addnode(slot_radius*np.cos(theta_p_r),slot_radius*np.sin(theta_p_r))

            femm.mi_addarc(
                yoke_radius * np.cos(0),
                yoke_radius * np.sin(0),
                yoke_radius * np.cos(theta_p_r),
                yoke_radius * np.sin(theta_p_r),
                theta_p_d,
                1,
            )

            #    femm.mi_addarc(D_a*0.5*np.cos(0),D_a*0.5*np.sin(0),D_a*0.5*np.cos(theta_b_t*0.5),D_a*0.5*np.sin(theta_b_t*0.5),5,1)
            #    femm.mi_selectarcsegment(D_a*0.5*np.cos(0),D_a*0.5*np.sin(0))
            #    femm.mi_setgroup(1)
            #
            #
            femm.mi_addarc(
                slot_radius, 0, slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5), 5, 1
            )
            femm.mi_selectarcsegment(slot_radius * np.cos(0), 0)
            femm.mi_setgroup(1)

            femm.mi_selectgroup(1)
            angle1_d = np.rad2deg(tau_p / (radius_sc + h_sc) - theta_b_t * 0.5)
            femm.mi_copyrotate(0, 0, angle1_d, 1)

            # femm.mi_addarc(slot_radius*np.cos(theta_b_t_new),slot_radius*np.sin(theta_b_t_new),slot_radius*np.cos(theta_p_r),slot_radius*np.sin(theta_p_r),5,1)

            # femm.mi_addsegment(slot_radius*np.cos(theta_p_r),slot_radius*np.sin(theta_p_r),D_a/2*np.cos(theta_p_r),D_a/2*np.sin(theta_p_r))

            r_o = (radius_sc + h_sc) * 2

            # femm.mi_addsegment(D_a/2*np.cos(0),D_a/2*np.sin(0),radius_sc*np.cos(0),radius_sc*np.sin(0))

            femm.mi_addnode(r_o * np.cos(0), r_o * np.sin(0))
            femm.mi_addnode(r_o * np.cos(theta_p_r), r_o * np.sin(theta_p_r))
            #
            ## Add some block labels materials properties
            femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            femm.mi_addmaterial(
                "NbTi", 0.6969698190303028, 0.6969698190303028, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            )
            femm.mi_getmaterial("M-36 Steel")
            femm.mi_getmaterial("20 SWG")
            femm.mi_addcircprop("A1+", I_sc, 1)
            femm.mi_addcircprop("A1-", -1 * I_sc, 1)
            femm.mi_addcircprop("A+", Current, 1)
            femm.mi_addcircprop("A-", Current, 1)
            femm.mi_addcircprop("B+", Current, 1)
            femm.mi_addcircprop("B-", Current, 1)
            femm.mi_addcircprop("C+", Current, 1)
            femm.mi_addcircprop("C-", Current, 1)
            femm.mi_addcircprop("D+", Current, 1)
            femm.mi_addcircprop("D-", Current, 1)
            femm.mi_addcircprop("E+", Current, 1)
            femm.mi_addcircprop("E-", Current, 1)
            femm.mi_addcircprop("F+", Current, 1)
            femm.mi_addcircprop("F-", Current, 1)

            femm.mi_addboundprop("Dirichlet", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            femm.mi_addboundprop("apbc1", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc2", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc3", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc4", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc5", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)

            femm.mi_addarc(
                r_o * np.cos(0), r_o * np.sin(0), r_o * np.cos(theta_p_r), r_o * np.sin(theta_p_r), theta_p_d, 1
            )
            femm.mi_selectarcsegment(r_o * np.cos(0), r_o * np.sin(0))
            femm.mi_setarcsegmentprop(5, "Dirichlet", 0, 5)

            femm.mi_addsegment(
                yoke_radius * np.cos(0), yoke_radius * np.sin(0), slot_radius * np.cos(0), slot_radius * np.sin(0)
            )
            femm.mi_addsegment(D_a * 0.5 * np.cos(0), D_a * 0.5 * np.sin(0), r_o * np.cos(0), r_o * np.sin(0))

            femm.mi_selectarcsegment(yoke_radius * np.cos(0), yoke_radius * np.sin(0))
            femm.mi_setarcsegmentprop(5, "Dirichlet", 0, 5)

            #            femm.mi_selectsegment(0.0, 0)
            #            femm.mi_setsegmentprop("apbc1", 100, 0, 0, 6)
            #            femm.mi_clearselected()

            femm.mi_selectsegment(slot_radius * 0.999 * np.cos(0), slot_radius * 0.999 * np.sin(0))
            femm.mi_setsegmentprop("apbc1", 100, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(h42 * 0.99 * np.cos(0), h42 * 0.99 * np.sin(0))
            femm.mi_setsegmentprop("apbc2", 100, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(0.99 * D_a * 0.5 * np.cos(0), 0.99 * D_a * 0.5 * np.sin(0))
            femm.mi_setsegmentprop("apbc3", 1, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(r_o * 0.95 * np.cos(0), r_o * 0.95 * np.sin(0))
            femm.mi_setsegmentprop("apbc4", 100, 0, 0, 6)
            femm.mi_clearselected()

            femm.mi_selectgroup(6)
            femm.mi_copyrotate(0, 0, theta_p_d, 1)

            #            femm.mi_selectsegment(h42 * 0.99 * np.cos(theta_p_r), h42 * 0.99 * np.sin(theta_p_r))
            #            femm.mi_setsegmentprop("apbc3", 100, 0, 0, 6)
            #            femm.mi_clearselected()
            #            femm.mi_selectsegment(D_a * 0.5 * 0.99 * np.cos(theta_p_r), D_a * 0.5 * 0.99 * np.sin(theta_p_r))
            #            femm.mi_setsegmentprop("apbc4", 100, 0, 0, 6)
            #            femm.mi_clearselected()

            iron_label = yoke_radius + (slot_radius - yoke_radius) * 0.5

            femm.mi_addblocklabel(iron_label * np.cos(theta_p_r * 0.5), iron_label * np.sin(theta_p_r * 0.5))
            femm.mi_selectlabel(iron_label * np.cos(theta_p_r * 0.5), iron_label * np.sin(theta_p_r * 0.5))
            femm.mi_setblockprop("M-36 Steel", 1, 1, "incircuit", 0, 7, 0)
            femm.mi_clearselected()

            select_angle_r = theta_b_s_new + theta_b_t * 0.9

            femm.mi_addarc(
                slot_radius * np.cos(theta_b_s_new),
                slot_radius * np.sin(theta_b_s_new),
                slot_radius * np.cos(theta_b_s_new + theta_b_t),
                slot_radius * np.sin(theta_b_s_new + theta_b_t),
                5,
                1,
            )
            femm.mi_selectarcsegment(slot_radius * np.cos(select_angle_r), slot_radius * np.sin(select_angle_r))
            femm.mi_setgroup(10)

            #    femm.mi_addarc(D_a*0.5*np.cos(theta_b_s_new),D_a*0.5*np.sin(theta_b_s_new),D_a*0.5*np.cos(theta_b_s_new+theta_b_t),D_a*0.5*np.sin(theta_b_s_new+theta_b_t),5,1)
            #    femm.mi_selectarcsegment(D_a*0.5*np.cos(select_angle_r*0.9),D_a*0.5*np.sin(select_angle_r*0.9))
            #    femm.mi_setgroup(3)

            femm.mi_selectgroup(10)
            femm.mi_copyrotate(0, 0, (theta_b_s_new + theta_b_t * 0.5) * 180 / np.pi, Slots_pp - 2)

            femm.mi_clearselected()

            femm.mi_addblocklabel(xlabel1, ylabel1)
            femm.mi_addblocklabel(xlabel2, ylabel2)
            femm.mi_selectlabel(xlabel1, ylabel1)
            femm.mi_setblockprop("NbTi", 1, 1, "A1+", 0, 10, N_sc)
            femm.mi_clearselected()
            femm.mi_selectlabel(xlabel2, ylabel2)
            femm.mi_setblockprop("NbTi", 1, 1, "A1-", 0, 7, N_sc)
            femm.mi_clearselected()

            layer_1 = slot_radius + (h42 - slot_radius) * 0.5

            femm.mi_addblocklabel(
                layer_1 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_1 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5)
            )
            femm.mi_selectlabel(
                layer_1 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_1 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5)
            )
            femm.mi_copyrotate(0, 0, (theta_b_s + theta_b_t) * 180 / np.pi, Slots_pp - 1)

            layer_2 = h42 + (D_a * 0.5 - h42) * 0.5

            femm.mi_addblocklabel(
                layer_2 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_2 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5)
            )
            femm.mi_selectlabel(
                layer_2 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_2 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5)
            )
            femm.mi_copyrotate(0, 0, (theta_b_s + theta_b_t) * 180 / np.pi, Slots_pp - 1)

            #    femm.mi_addblocklabel(h42*np.cos(theta_b_t*0.25),h42*np.sin(theta_b_t*0.25))
            #    femm.mi_selectlabel(h42*np.cos(theta_b_t*0.25),h42*np.sin(theta_b_t*0.25))
            #    femm.mi_copyrotate(0,0,(theta_p_r-theta_b_t*0.5)*180/np.pi,1)
            #
            #    femm.mi_addblocklabel(layer_2*np.cos(theta_b_s*0.25),layer_2*np.sin(theta_b_s*0.25))
            #    femm.mi_selectlabel(layer_2*np.cos(theta_b_s*0.25),layer_2*np.sin(theta_b_s*0.25))
            #    femm.mi_copyrotate(0,0,(theta_p_r-theta_b_t*0.5)*180/np.pi,1)

            #            femm.mi_addblocklabel(
            #                yoke_radius * 0.5 * np.cos(theta_p_r * 0.5), yoke_radius * 0.5 * np.sin(theta_p_r * 0.5)
            #            )
            #            femm.mi_selectlabel(
            #                yoke_radius * 0.5 * np.cos(theta_p_r * 0.5), yoke_radius * 0.5 * np.sin(theta_p_r * 0.5)
            #            )
            #            femm.mi_setblockprop("Air", 1, 1, "incircuit", 0, 7, 0)
            #            femm.mi_clearselected()

            femm.mi_addblocklabel(D_sc * 0.5 * np.cos(theta_p_r * 0.5), D_sc * 0.5 * np.sin(theta_p_r * 0.5))
            femm.mi_selectlabel(D_sc * 0.5 * np.cos(theta_p_r * 0.5), D_sc * 0.5 * np.sin(theta_p_r * 0.5))
            femm.mi_setblockprop("Air", 1, 1, "incircuit", 0, 7, 0)
            femm.mi_clearselected()

            pitch = 1

            Phases = ["A+", "D+", "C-", "F-", "B+", "E+", "A-", "D-", "C-"]
            # N_c_a = [2 * N_c, 4 * N_c, 4 * N_c, 4 * N_c, 4 * N_c, 4 * N_c, 2 * N_c, 2 * N_c, 2 * N_c]

            count = 0
            angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
            delta_theta = theta_b_t + theta_b_s

            for pitch in range(1, int(np.ceil(Y_q)), 2):
                try:
                    femm.mi_selectlabel(
                        layer_2 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
                        layer_2 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
                    )
                    femm.mi_selectlabel(
                        layer_2 * np.cos(angle_r + pitch * delta_theta), layer_2 * np.sin(angle_r + pitch * delta_theta)
                    )
                    femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c)
                    # femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c_a[count])
                    femm.mi_clearselected()
                    count = count + 1
                except:
                    continue

            count = 0
            angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
            delta_theta = theta_b_t + theta_b_s
            for pitch in range(1, int(np.ceil(Y_q)), 2):
                try:
                    femm.mi_selectlabel(
                        layer_1 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
                        layer_1 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
                    )
                    femm.mi_selectlabel(
                        layer_1 * np.cos(angle_r + pitch * delta_theta), layer_1 * np.sin(angle_r + pitch * delta_theta)
                    )
                    femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c)
                    # femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c_a[count+1])
                    femm.mi_clearselected()
                    count = count + 1
                except:
                    continue

            # Now, the finished input geometry can be displayed.
            femm.mi_zoomnatural()
            ## We have to give the geometry a name before we can analyze it.
            femm.mi_saveas("coil_design_new.fem")

            # Analyze geometry with pyfemm
            # femm.mi_analyze()
            f=2*p1*N_nom/120
            Time =60/(f*2*np.pi)

            Theta_elec=(theta_tau_s*Time)*2*np.pi*f

            try:

                femm.mi_analyze()

                # Load and post-process results
                femm.mi_loadsolution()
                n = 0
                _, _, B_coil_max, _ = run_post_process(
                    D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n
                )
                Load_line_slope = I_sc / B_coil_max
                # SHOULD STRONGLY CONSIDER A USER DEFINED LIMIT INSTEAD
                a = 5.8929
                b = -(Load_line_slope + 241.32)
                c = 1859.9
                B_o = (-b - np.sqrt(b ** 2.0 - 4.0 * a * c)) / 2.0 / a
                # Populate openmdao outputs
                # Max current from manufacturer of superconducting coils, quadratic fit
                # outputs["margin_I_c"] = 3.5357 * B_coil_max ** 2.0 - 144.79 * B_coil_max + 1116.0

                outputs["margin_I_c"] = outputs["I_sc_out"] = I_sc_out = B_o * Load_line_slope - 1e3*(1-load_margin)

                myopen()
                femm.opendocument("coil_design_new.fem")
                femm.mi_modifycircprop("A1+",  1, I_sc_out)
                femm.mi_modifycircprop("A1-",  1, -1 * I_sc_out)
                femm.mi_saveas("coil_design_new.fem")

                femm.mi_analyze()

                # Load and post-process results
                femm.mi_loadsolution()

                outputs["B_g"], outputs["B_rymax"], B_coil_max, outputs["Sigma_normal"] = run_post_process(
                    D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n
                )
                # B_o is the max allowable flux density at the coil, B_coil_max is the max value from femm
                outputs["B_coil_max"] = B_coil_max
                outputs["constr_B_g_coil"] = outputs["B_g"] / B_coil_max
                outputs["Coil_max_ratio"] = B_coil_max / B_o
                outputs["Torque_actual"], outputs["Sigma_shear"] = B_r_B_t(Theta_elec,
                    D_a, l_s, p1, delta_em, theta_p_r, I_s, theta_b_t, theta_b_s, layer_1, layer_2, Y_q, N_c, tau_p
                )
            except Exception as e:
                #raise(e)
                outputs = bad_inputs(outputs)

            femm.closefemm()
            #cleanup_femm_files()
