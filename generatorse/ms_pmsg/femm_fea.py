# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:28:2 2021

@author: lsethura
"""
import femm
import numpy as np
import openmdao.api as om
from sympy import Point, Segment
#from sympy.geometry.util import centroid
import random
#import csv
import os
import platform


def myopen():
    if platform.system().lower() == "windows":
        femm.openfemm(1)
    else:
        femm.openfemm(winepath=os.environ["WINEPATH"], femmpath=os.environ["FEMMPATH"])
    femm.smartmesh(0)


# The package must be initialized with the openfemm command.


def cleanup_femm_files():
    clean_dir = os.getcwd()
    files = os.listdir(clean_dir)
    for file in files:
        if file.endswith(".ans") or file.endswith(".fem") or file.endswith(".csv"):
            os.remove(os.path.join(clean_dir, file))


def bad_inputs(outputs):
    print("Bad inputs for geometry")
    outputs["B_g"] = 2.5
    outputs["B_rymax"] = 2.5
    outputs["B_symax"] = 2.5
    outputs["Sigma_normal"] = 1e9
    outputs["M_Fes"] = 100000
    outputs["M_Fery"] = 100000
    outputs["M_Fesy"] = 100000
    outputs["M_Fest"] = 100000
    outputs["Iron"] = 1e8
    outputs["T_e"] = 1e9
    outputs["Sigma_shear"] = 1e9
    femm.mi_saveas("IPM_new_bad_geomtery.fem")
    return outputs


def run_post_process(r_g, g, r_outer, h_yr, h_ys, r_m, yoke_radius, theta_p_r):

    theta_p_d = np.rad2deg(theta_p_r)
    femm.mi_loadsolution()
    femm.mo_smooth("off")

    femm.mo_selectblock(
        (yoke_radius + h_yr * 0.5) * np.cos(theta_p_r * 0.5),
        (yoke_radius + h_yr * 0.5) * np.sin(theta_p_r * 0.5),
    )

    V_rotor = femm.mo_blockintegral(10)
    femm.mo_clearblock()
    sy_area = []
    ry_area = []
    t_area = []
    numelm2 = femm.mo_numelements()
    for r in range(1, numelm2):
        p3, p4, p5, x1, y1, a1, g1 = femm.mo_getelement(r)
        mag = np.sqrt(x1**2 + y1**2)
        if mag <= r_m:
            a = femm.mo_getb(x1, y1)
            ry_area.append(np.sqrt(a[0] ** 2 + a[1] ** 2))
        elif mag >= (r_outer - h_ys):
            b = femm.mo_getb(x1, y1)
            sy_area.append(np.sqrt(b[0] ** 2 + b[1] ** 2))
        elif (mag >= r_g) and (mag <= (r_outer - h_ys)):
            b = femm.mo_getb(x1, y1)
            t_area.append(np.sqrt(b[0] ** 2 + b[1] ** 2))

    B_symax = max(sy_area)
    B_rymax = max(ry_area)
    B_tmax = max(t_area)

    femm.mo_addcontour((r_g - g * 0.5) * np.cos(0), (r_g - g * 0.5) * np.sin(0))
    femm.mo_addcontour((r_g - g * 0.5) * np.cos(theta_p_r * 1), (r_g - g * 0.5) * np.sin(theta_p_r * 1))
    femm.mo_bendcontour(theta_p_d * 1, 1)
    femm.mo_makeplot(1, 1000, "gap.csv", 1)
    femm.mo_makeplot(2, 100, "B_r.csv", 1)
    femm.mo_makeplot(3, 100, "B_t.csv", 1)
    femm.mo_clearcontour()
    # #
    femm.mo_selectblock(
        (r_outer - h_ys * 0.5) * np.cos(theta_p_r * 0.5),
        (r_outer - h_ys * 0.5) * np.sin(theta_p_r * 0.5),
    )
    V_stator = femm.mo_blockintegral(10)

    B_g_peak = np.loadtxt("gap.csv")[:, 1].max()
    B_r_normal = np.loadtxt("B_r.csv")
    B_t_normal = np.loadtxt("B_t.csv")

    circ = B_r_normal[-1, 0]
    delta_L = np.diff(B_r_normal[:, 0])[0]

    force = np.sum((B_r_normal[:, 1]) ** 2 - (B_t_normal[:, 1]) ** 2) * delta_L
    sigma_n = abs(1 / (8 * np.pi * 1e-7) * force) / circ

    return B_g_peak, B_rymax, B_symax, B_tmax, sigma_n, V_rotor, V_stator


def B_r_B_t(Theta_elec, r_g, l_s, p, g, theta_p_r, I_s, theta_tau_s, layer_1, layer_2, N_c):

    theta_p_d = np.rad2deg(theta_p_r)

    femm.openfemm(1)
    femm.opendocument("MS_PMSG.fem")
    femm.mi_modifycircprop("A+", 1, I_s * np.sin(-2 * np.pi / 3))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(0 * np.pi / 3))
    femm.mi_modifycircprop("C+", 1, I_s * np.sin(2 * np.pi / 3))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(-2 * np.pi / 3))
    femm.mi_modifycircprop("B-", 1, -I_s * np.sin(0 * np.pi / 3))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(2 * np.pi / 3))

    femm.mi_saveas("MS_PMSG1.fem")
    femm.smartmesh(0)
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.mi_loadsolution()

    femm.mo_addcontour((r_g - g * 0.5) * np.cos(0), (r_g - g * 0.5) * np.sin(0))
    femm.mo_addcontour((r_g - g * 0.5) * np.cos(theta_p_r * 1), (r_g - g * 0.5) * np.sin(theta_p_r * 1))
    femm.mo_bendcontour(theta_p_d * 1, 1)
    femm.mo_makeplot(2, 100, "B_r_1.csv", 1)
    femm.mo_makeplot(3, 100, "B_t_1.csv", 1)
    femm.mo_clearcontour()
    femm.mo_close()
    femm.openfemm(1)
    femm.opendocument("MS_PMSG1.fem")

    Phases1 = ["A+", "C-", "B+"]
    Phases2 = ["C-", "B+", "A-"]

    angle_r = theta_tau_s * 0.5
    delta_theta = theta_tau_s
    for pitch in range(1, 4):
        femm.mi_selectlabel(
            layer_2 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
            layer_2 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
        )

        femm.mi_setblockprop("20 SWG", 1, 0, Phases1[pitch - 1], 0, 15, N_c)
        femm.mi_clearselected()
        femm.mi_selectlabel(
            layer_1 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
            layer_1 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
        )
        femm.mi_setblockprop("20 SWG", 1, 0, Phases2[pitch - 1], 0, 15, N_c)
        femm.mi_clearselected()

    femm.mi_modifycircprop("A+", 1, I_s * np.sin(Theta_elec - 2 * np.pi / 3))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(Theta_elec + 0 * np.pi / 3))
    femm.mi_modifycircprop("C+", 1, I_s * np.sin(Theta_elec + 2 * np.pi / 3))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(Theta_elec - 2 * np.pi / 3))
    femm.mi_modifycircprop("B-", 1, -I_s * np.sin(Theta_elec + 0 * np.pi / 3))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(Theta_elec + 2 * np.pi / 3))

    femm.mi_saveas("MS_PMSG2.fem")
    femm.smartmesh(0)
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_addcontour((r_g - g * 0.5) * np.cos(0), (r_g - g * 0.5) * np.sin(0))
    femm.mo_addcontour((r_g - g * 0.5) * np.cos(theta_p_r * 1), (r_g - g * 0.5) * np.sin(theta_p_r * 1))
    femm.mo_bendcontour(theta_p_d * 1, 1)
    femm.mo_makeplot(2, 100, "B_r_2.csv", 1)
    femm.mo_makeplot(3, 100, "B_t_2.csv", 1)
    femm.mo_clearcontour()
    femm.mo_close()

    B_r_1 = np.loadtxt("B_r_1.csv")
    B_t_1 = np.loadtxt("B_t_1.csv")
    B_r_2 = np.loadtxt("B_r_2.csv")
    B_t_2 = np.loadtxt("B_t_2.csv")
    delta_L = np.diff(B_r_1[:, 0])[0]
    circ = B_r_1[-1, 0]

    force = np.array([np.sum(B_r_1[:, 1] * B_t_1[:, 1]), np.sum(B_r_2[:, 1] * B_t_2[:, 1])]) * delta_L
    sigma_t = abs(1 / (4 * np.pi * 1e-7) * force) / circ
    torque = np.pi / 2 * sigma_t * (2 * r_g) ** 2 * l_s
    torque_ripple = (torque[0] - torque[1]) * 100 / torque.mean()
    #print(torque[0], torque[1], torque.mean())
    return torque.mean(), sigma_t.mean()


#
class FEMM_Geometry(om.ExplicitComponent):
    def setup(self):
        self.add_input("m", 3, desc="number of phases")
        self.add_input("k_sfil", 0.65, desc="slot fill")
        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("L_t", 0.0, units="m", desc="effective stator length for support structure")

        self.add_input("p", 0.0, desc="pole pairs ")
        self.add_input("g", 0.0, units="m", desc="air gap length ")
        self.add_input("b_s", 0.0, units="m", desc="slot width")
        self.add_input("tau_s", 0.0, units="m", desc="slot pitch")

        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("f", 0.0, units="Hz", desc="frequency")
        self.add_input("h_s1", 0.010, desc="Slot Opening height")
        self.add_input("h_s2", 0.010, desc="Wedge Opening height")
        self.add_input("ratio", 0.00, desc="Wpole width to pole pitch")

        self.add_output("h_t", 0.0, units="m", desc="tooth height")

        self.add_input("h_m", 0.0, units="m", desc="magnet height")
        self.add_input("b_m", 0.0, units="m", desc="magnet width ")
        self.add_input("tau_p", 0.0, units="m", desc="pole pitch")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("h_ys", 0.0, units="m", desc="stator yoke height")
        self.add_input("h_s", 0.0, units="m", desc="stator slot height")
        self.add_input("r_g", 0.0, units="m", desc="air gap radius ")
        self.add_input("I_s", 0.0, units="A", desc="Stator current ")
        self.add_input("N_c", 0.0, desc="Number of turns per coil in series")
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        self.add_output("B_pm1", 0.0, units="T", desc="Peak air gap flux density ")
        self.add_output("B_rymax", 0.0, units="T", desc="Peak Rotor yoke flux density")
        self.add_output("B_symax", 0.0, units="T", desc="Peak flux density in thestator yoke")
        self.add_output("B_tmax", 0.0, units="T", desc="Peak tooth flux density")
        self.add_output("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_output("Sigma_shear", 0.0, units="Pa", desc="Shear stress")
        self.add_output("Sigma_normal", 0.0, units="Pa", desc="Normal stress")
        self.add_output("Iron", 0.0, units="kg", desc="magnet length ")
        self.add_output("M_Fes", 0.0, units="kg", desc="mstator iron mass ")
        self.add_output("M_Fery", 0.0, units="kg", desc="rotor iron mass ")
        self.add_output("M_Fest", 0.0, units="kg", desc="Stator teeth mass ")
        self.add_output("M_Fesy", 0.0, units="kg", desc="Stator yoke mass")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        # Unpack variables
        l_s = float(inputs["l_s"])
        g = float(inputs["g"])
        h_yr = float(inputs["h_yr"])
        h_s = float(inputs["h_s"])
        L_t = float(inputs["L_t"])

        h_ys = float(inputs["h_ys"])
        tau_s = float(inputs["tau_s"])
        m = float(inputs["m"])

        m = float(inputs["m"])
        h_s1 = float(inputs["h_s1"])
        h_s2 = float(inputs["h_s2"])
        ratio = float(inputs["ratio"])
        r_g = float(inputs["r_g"])
        N_c = float(inputs["N_c"])
        I_s = float(inputs["I_s"])
        rho_Fe = float(inputs["rho_Fe"])
        f = float(inputs["f"])
        h_m = float(inputs["h_m"])
        b_m = float(inputs["b_m"])
        b_s = float(inputs["b_s"])
        f = float(inputs["f"])
        N_nom = float(inputs["N_nom"])
        tau_s = float(inputs["tau_s"])

        q = 1
        p = float(inputs["p"])
        Slots = 2 * q * m * p

        # Assign values to design constants

        Slots_pp = 2 * q * m

        theta_p_d = 180.0 / p

        b_so = 2 * g

        theta_p_r = np.deg2rad(theta_p_d)

        r_m = r_g - g

        theta_b_s = np.arctan(b_s / (r_g))

        theta_tau_s = theta_p_r * 2 / Slots_pp

        theta_tau_s_new = np.arctan((tau_s * 0.5 - b_so * 0.5) / (r_g))

        theta_tau_s_new2 = np.arctan((tau_s * 0.5 - b_s * 0.5) / (r_g))

        theta_tau_s_new3 = np.arctan((tau_s * 0.5 + b_so * 0.5) / (r_g))

        theta_tau_s_new4 = np.arctan((tau_s * 0.5 + b_s * 0.5) / (r_g))

        theta_b_so = b_so / (r_g)

        outputs["h_t"] = h_s + h_s1 + h_s2

        femm.openfemm(1)
        femm.newdocument(0)
        femm.mi_probdef(0, "meters", "planar", 1.0e-8, l_s, 30, 0)
        Current = 0

        # Define the problem type.  Magnetostatic; Units of mm; Axisymmetric;
        # Precision of 10^(-8) for the linear solver; a placeholder of 0 for
        # the depth dimension, and an angle constraint of 30 degrees
        # Add some block labels materials properties
        femm.mi_addmaterial("Air")
        femm.mi_getmaterial("M-36 Steel")
        femm.mi_getmaterial("20 SWG")
        femm.mi_getmaterial("N48")
        femm.mi_modifymaterial("N48", 0, "N48SH")
        b = [0, 1.20, 1.22, 1.23, 1.24, 1.26, 1.27, 1.29, 1.30, 1.31]
        h = [
            0.000000,
            82000.000000,
            232000.000000,
            392000.000000,
            557000.000000,
            717000.000000,
            872000.000000,
            1037000.000000,
            1352000.000000,
            1512000.000000,
        ]
        # h=[-1.512E+06,-1.43E+06,-1.28E+06,-1.12E+06,-9.55E+05,-7.95E+05,-6.40E+05,-4.75E+05,-1.60E+05,0]

        femm.mi_modifymaterial("N48SH", 5, 0.7142857)

        femm.mi_clearbhpoints("N48SH")
        for i in range(0, 10):
            femm.mi_addbhpoint("N48SH", b[i], h[i])

        femm.mi_modifymaterial("N48SH", 9, 0)
        femm.mi_modifymaterial("N48SH", 3, 1512000)

        femm.mi_addcircprop("A+", Current, 1)
        femm.mi_addcircprop("A-", Current, 1)
        femm.mi_addcircprop("B+", Current, 1)
        femm.mi_addcircprop("B-", Current, 1)
        femm.mi_addcircprop("C+", Current, 1)
        femm.mi_addcircprop("C-", Current, 1)

        femm.mi_addboundprop("Dirichlet", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_addboundprop("apbc1", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
        femm.mi_addboundprop("apbc2", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
        femm.mi_addboundprop("apbc3", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
        femm.mi_addboundprop("apbc4", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
        femm.mi_addboundprop("apbc5", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)

        # draw magnets and rotor

        mag_angle = np.arctan(b_m / r_m)
        theta_p_m_r = np.deg2rad((theta_p_d - np.rad2deg(mag_angle)) / 2)
        femm.mi_addnode(r_m * np.cos(theta_p_m_r), r_m * np.sin(theta_p_m_r))
        femm.mi_addnode((r_m - h_m) * np.cos(theta_p_m_r), (r_m - h_m) * np.sin(theta_p_m_r))

        X_1, Y_1 = r_m * np.cos(mag_angle + theta_p_m_r), r_m * np.sin(mag_angle + theta_p_m_r)
        X_2, Y_2 = (r_m - h_m) * np.cos(mag_angle + theta_p_m_r), (r_m - h_m) * np.sin(mag_angle + theta_p_m_r)
        X_3, Y_3 = r_m * np.cos(theta_p_m_r), r_m * np.sin(theta_p_m_r)
        X_4, Y_4 = (r_m - h_m) * np.cos(theta_p_m_r), (r_m - h_m) * np.sin(theta_p_m_r)
        X_5, Y_5 = (r_m - h_m) * np.cos(theta_p_r), (r_m - h_m) * np.sin(theta_p_r)
        X_6, Y_6 = r_m - h_m, 0
        X_7, Y_7 = r_m - h_m - h_yr, 0
        X_8, Y_8 = (r_m - h_m - h_yr) * np.cos(theta_p_r * 1), (r_m - h_m - h_yr) * np.sin(theta_p_r * 1)

        x0, y0 = 0.0, 0.0
        xc, yc = r_m * np.cos(theta_p_r), r_m * np.sin(theta_p_r)
        femm.mi_addnode(X_1, Y_1)
        femm.mi_selectnode(X_1, Y_1)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addnode(X_2, Y_2)
        femm.mi_selectnode(X_2, Y_2)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addnode(X_3, Y_3)
        femm.mi_selectnode(X_3, Y_3)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addnode(X_4, Y_4)
        femm.mi_selectnode(X_4, Y_4)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addnode(X_5, Y_5)
        femm.mi_selectnode(X_5, Y_5)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addarc(X_3, Y_3, X_1, Y_1, np.rad2deg(mag_angle), 1)
        femm.mi_selectarcsegment(X_3, Y_3)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addarc(X_3, Y_3, X_1, Y_1, np.rad2deg(mag_angle), 1)
        femm.mi_selectarcsegment(X_3, Y_3)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addarc(X_4, Y_4, X_2, Y_2, np.rad2deg(mag_angle), 1)
        femm.mi_selectarcsegment(X_2, Y_2)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addsegment(X_3, Y_3, X_4, Y_4)
        femm.mi_selectsegment(X_3, Y_3)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addsegment(X_1, Y_1, X_2, Y_2)
        femm.mi_selectsegment(X_1, Y_1)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addnode(X_6, Y_6)
        femm.mi_selectnode(X_6, Y_6)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addarc(X_6, Y_6, X_4, Y_4, np.rad2deg(theta_p_m_r), 1)
        femm.mi_selectarcsegment(X_6, Y_6)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addarc(X_2, Y_2, X_5, Y_5, np.rad2deg(theta_p_m_r), 1)
        femm.mi_selectarcsegment(X_5, Y_5)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        femm.mi_addnode(X_7, Y_7)
        femm.mi_addnode(X_8, Y_8)
        femm.mi_addarc(X_7, Y_7, X_8, Y_8, theta_p_d * 1, 1)
        femm.mi_selectarcsegment(X_8, Y_8)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        mag_dir = np.rad2deg(theta_p_m_r + mag_angle / 2)
        mag_dir2 = np.rad2deg(theta_p_r + mag_angle / 2 + theta_p_m_r) + 180

        line1 = Segment(Point(X_1, Y_1, evaluate=False), Point(X_4, Y_4, evaluate=False))
        l1 = line1.midpoint.evalf()
        femm.mi_addblocklabel(l1.x, l1.y)
        femm.mi_selectlabel(l1.x, l1.y)
        femm.mi_setblockprop("N48SH", 1, 1, 0, mag_dir, 1, 0)
        femm.mi_clearselected()

        #        femm.mi_selectgroup(1)
        #        femm.mi_copyrotate(x0,y0,theta_p_d,1)
        #        femm.mi_clearselected()

        #        r_mag_label=np.sqrt(float(l1.x)**2+float(l1.y)**2)
        #        femm.mi_selectlabel(r_mag_label*np.cos(theta_p_r+theta_p_m_r+mag_angle*0.5),r_mag_label*np.sin(theta_p_r+theta_p_m_r*+mag_angle*0.5))
        #        femm.mi_setblockprop("N48SH", 1, 1, 0,mag_dir2,1,0 )
        #        femm.mi_clearselected()

        femm.mi_addsegment(X_6, Y_6, X_7, Y_7)
        femm.mi_selectsegment(X_7, Y_7)
        femm.mi_setgroup(13)
        femm.mi_selectsegment(X_7, Y_7)
        femm.mi_setsegmentprop("apbc1", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_addblocklabel(
            (r_m - h_m - h_yr * 0.5) * np.cos(theta_p_r * 0.5), (r_m - h_m - h_yr * 0.5) * np.sin(theta_p_r * 0.5)
        )
        femm.mi_selectlabel(
            (r_m - h_m - h_yr * 0.5) * np.cos(theta_p_r * 0.5), (r_m - h_m - h_yr * 0.5) * np.sin(theta_p_r * 0.5)
        )
        femm.mi_setblockprop("M-36 Steel")
        femm.mi_clearselected()

        femm.mi_selectarcsegment((r_m - h_m - h_yr) * np.cos(theta_p_r), (r_m - h_m - h_yr) * np.sin(theta_p_r))
        femm.mi_setarcsegmentprop(10, "Dirichlet", 0, 50)
        femm.mi_clearselected()

        #### #  *************
        ####       # Draw the stator slots and Stator coils
        ####
        X_9, Y_9 = r_g, 0
        X_10, Y_10 = r_g * np.cos(theta_tau_s_new), r_g * np.sin(theta_tau_s_new)
        X_11, Y_11 = (r_g + h_s1) * np.cos(theta_tau_s_new), (r_g + h_s1) * np.sin(theta_tau_s_new)
        X_12, Y_12 = (r_g + h_s1 + h_s2) * np.cos(theta_tau_s_new2), (r_g + h_s1 + h_s2) * np.sin(theta_tau_s_new2)
        X_13, Y_13 = (r_g + h_s1 + h_s2 + h_s / 2) * np.cos(theta_tau_s_new2), (r_g + h_s1 + h_s2 + h_s / 2) * np.sin(
            theta_tau_s_new2
        )
        X_14, Y_14 = (r_g + h_s1 + h_s2 + h_s) * np.cos(theta_tau_s_new2), (r_g + h_s1 + h_s2 + h_s) * np.sin(
            theta_tau_s_new2
        )
        X_15, Y_15 = (r_g) * np.cos(theta_tau_s_new3), (r_g) * np.sin(theta_tau_s_new3)
        X_16, Y_16 = (r_g + h_s1) * np.cos(theta_tau_s_new3), (r_g + h_s1) * np.sin(theta_tau_s_new3)
        X_17, Y_17 = (r_g + h_s1 + h_s2) * np.cos(theta_tau_s_new4), (r_g + h_s1 + h_s2) * np.sin(theta_tau_s_new4)
        X_18, Y_18 = (r_g + h_s1 + h_s2 + h_s / 2) * np.cos(theta_tau_s_new4), (r_g + h_s1 + h_s2 + h_s / 2) * np.sin(
            theta_tau_s_new4
        )
        X_19, Y_19 = (r_g + h_s1 + h_s2 + h_s) * np.cos(theta_tau_s_new4), (r_g + h_s1 + h_s2 + h_s) * np.sin(
            theta_tau_s_new4
        )

        X_20, Y_20 = r_m, 0
        X_21, Y_21 = (r_g + h_s1 + h_s2 + h_s + h_ys), 0
        X_22, Y_22 = (r_g + h_s1 + h_s2 + h_s + h_ys) * np.cos(theta_p_r * 1), (
            r_g + h_s1 + h_s2 + h_s + h_ys
        ) * np.sin(theta_p_r * 1)
        ###
        ###
        ##
        femm.mi_addnode(X_9, Y_9)
        femm.mi_selectnode(X_9, Y_9)
        femm.mi_setgroup(2)

        femm.mi_addnode(X_10, Y_10)
        femm.mi_selectnode(X_10, Y_10)
        femm.mi_setgroup(2)

        femm.mi_addnode(X_11, Y_11)
        femm.mi_selectnode(X_11, Y_11)
        femm.mi_setgroup(2)

        femm.mi_addnode(X_12, Y_12)
        femm.mi_selectnode(X_12, Y_12)
        femm.mi_setgroup(2)

        femm.mi_addnode(X_13, Y_13)
        femm.mi_selectnode(X_13, Y_13)
        femm.mi_setgroup(2)

        femm.mi_addnode(X_14, Y_14)
        femm.mi_selectnode(X_14, Y_14)
        femm.mi_setgroup(2)

        femm.mi_addarc(X_9, Y_9, X_10, Y_10, np.rad2deg(theta_tau_s_new), 1)
        femm.mi_selectarcsegment(X_10, Y_10)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_addsegment(X_11, Y_11, X_10, Y_10)
        femm.mi_selectsegment(X_10, Y_10)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_addsegment(X_11, Y_11, X_12, Y_12)
        femm.mi_selectsegment(X_12, Y_12)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_addsegment(X_13, Y_13, X_12, Y_12)
        femm.mi_selectsegment(X_13, Y_13)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_addsegment(X_14, Y_14, X_13, Y_13)
        femm.mi_selectsegment(X_14, Y_14)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_selectgroup(2)
        femm.mi_mirror(x0, y0, r_g * np.cos(theta_tau_s * 0.5), r_g * np.sin(theta_tau_s * 0.5))
        femm.mi_clearselected()

        femm.mi_addarc(X_12, Y_12, X_17, Y_17, np.rad2deg(theta_b_s), 1)
        femm.mi_selectarcsegment(X_17, Y_17)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_addarc(X_13, Y_13, X_18, Y_18, np.rad2deg(theta_b_s), 1)
        femm.mi_selectarcsegment(X_18, Y_18)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_addarc(X_14, Y_14, X_19, Y_19, np.rad2deg(theta_b_s), 1)
        femm.mi_selectarcsegment(X_19, Y_19)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        femm.mi_selectgroup(2)
        femm.mi_copyrotate(x0, y0, np.rad2deg(theta_tau_s), 2)
        femm.mi_clearselected()

        femm.mi_addnode(X_20, Y_20)
        femm.mi_selectnode(X_20, Y_20)
        femm.mi_setgroup(2)

        femm.mi_addsegment(X_6, Y_6, X_20, Y_20)
        femm.mi_selectsegment(X_20, Y_20)
        femm.mi_setgroup(13)
        femm.mi_clearselected()
        femm.mi_selectsegment(X_20, Y_20)
        femm.mi_setsegmentprop("apbc2", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_addsegment(X_20, Y_20, X_9, Y_9)
        femm.mi_selectsegment(X_9, Y_9)
        femm.mi_setgroup(13)
        femm.mi_clearselected()
        femm.mi_selectsegment(X_9, Y_9)
        femm.mi_setsegmentprop("apbc3", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_addnode(X_21, Y_21)
        femm.mi_selectnode(X_21, Y_21)
        femm.mi_setgroup(2)

        femm.mi_addsegment(X_20, Y_20, X_21, Y_21)
        femm.mi_selectsegment(X_21, Y_21)
        femm.mi_setgroup(13)
        femm.mi_clearselected()

        femm.mi_selectsegment(X_21, Y_21)
        femm.mi_setsegmentprop("apbc4", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_selectgroup(13)
        femm.mi_copyrotate(x0, y0, theta_p_d * 1, 1)
        femm.mi_clearselected()

        femm.mi_addarc(X_21, Y_21, X_22, Y_22, np.rad2deg(theta_p_r * 1), 1)
        femm.mi_selectarcsegment(X_22, Y_22)
        femm.mi_setarcsegmentprop(10, "Dirichlet", 0, 50)
        femm.mi_clearselected()

        layer_1 = r_g + h_s1 + h_s2 + h_s * 0.25
        layer_2 = r_g + h_s1 + h_s2 + h_s * 0.75
        #
        femm.mi_addblocklabel(layer_1 * np.cos(theta_tau_s * 0.5), layer_1 * np.sin(theta_tau_s * 0.5))
        femm.mi_selectlabel(layer_1 * np.cos(theta_tau_s * 0.5), layer_1 * np.sin(theta_tau_s * 0.5))
        femm.mi_copyrotate(0, 0, np.rad2deg(theta_tau_s), 2)
        ##
        #
        femm.mi_addblocklabel(layer_2 * np.cos(theta_tau_s * 0.5), layer_2 * np.sin(theta_tau_s * 0.5))
        femm.mi_selectlabel(layer_2 * np.cos(theta_tau_s * 0.5), layer_2 * np.sin(theta_tau_s * 0.5))
        femm.mi_copyrotate(0, 0, np.rad2deg(theta_tau_s), 2)
        ###
        ##
        ##
        ##
        femm.mi_addblocklabel((r_m + g * 0.5) * np.cos(theta_p_r * 0.5), (r_m + g * 0.5) * np.sin(theta_p_r * 0.5))
        femm.mi_selectlabel((r_m + g * 0.5) * np.cos(theta_p_r * 0.5), (r_m + g * 0.5) * np.sin(theta_p_r * 0.5))
        femm.mi_setblockprop("Air")
        femm.mi_clearselected()

        femm.mi_addblocklabel(
            (r_g + h_s1 + h_s2 + h_s + h_ys * 0.5) * np.cos(theta_p_r * 0.5),
            (r_g + h_s1 + h_s2 + h_s + h_ys * 0.5) * np.sin(theta_p_r * 0.5),
        )
        femm.mi_selectlabel(
            (r_g + h_s1 + h_s2 + h_s + h_ys * 0.5) * np.cos(theta_p_r * 0.5),
            (r_g + h_s1 + h_s2 + h_s + h_ys * 0.5) * np.sin(theta_p_r * 0.5),
        )
        femm.mi_setblockprop("M-36 Steel")
        femm.mi_clearselected()

        Phases1 = ["B-", "A+", "C-"]
        Phases2 = ["A+", "C-", "B+"]

        angle_r = theta_tau_s * 0.5
        delta_theta = theta_tau_s
        for pitch in range(1, 4):
            femm.mi_selectlabel(
                layer_2 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
                layer_2 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
            )

            femm.mi_setblockprop("20 SWG", 1, 0, Phases1[pitch - 1], 0, 15, N_c)
            femm.mi_clearselected()
            femm.mi_selectlabel(
                layer_1 * np.cos(angle_r + (pitch - 1) * (delta_theta)),
                layer_1 * np.sin(angle_r + (pitch - 1) * (delta_theta)),
            )
            femm.mi_setblockprop("20 SWG", 1, 0, Phases2[pitch - 1], 0, 15, N_c)
            femm.mi_clearselected()
        ###
        ##
        ##
        femm.mi_saveas("MS_PMSG.fem")
        Time = 60 / (f * 2 * np.pi)

        Theta_elec = (theta_tau_s * Time) * 2 * np.pi * f

        r_outer = r_g + h_s1 + h_s2 + h_s + h_ys
        r_inner = r_g + h_s1 + h_s2 + h_s
        yoke_radius = r_m - h_m - h_yr

        try:
            femm.mi_createmesh()
            femm.smartmesh(0)
            femm.mi_analyze()
            (
                outputs["B_g"],
                outputs["B_rymax"],
                outputs["B_symax"],
                outputs["B_tmax"],
                outputs["Sigma_normal"],
                V_rotor,
                V_stator,
            ) = run_post_process(r_g, g, r_outer, h_yr, h_ys, r_m, yoke_radius, theta_p_r)
            V_Fesy = L_t * np.pi * ((r_g + h_s + h_ys) ** 2 - (r_g + h_s) ** 2)  # volume of iron in stator yoke
            outputs["M_Fes"] = V_stator * rho_Fe * p
            outputs["M_Fest"] = outputs["M_Fes"] - np.pi * (r_outer**2 - r_inner**2) * l_s * rho_Fe
            outputs["M_Fery"] = V_rotor * rho_Fe * p
            outputs["Iron"] = outputs["M_Fes"] + outputs["M_Fery"]
            outputs["M_Fesy"] = V_Fesy * rho_Fe  # Mass of stator yoke
            outputs["Iron"] = outputs["M_Fest"] + outputs["M_Fesy"] + outputs["M_Fery"]

            outputs["T_e"], outputs["Sigma_shear"] = B_r_B_t(
                Theta_elec, r_g, l_s, p, g, theta_p_r, I_s, theta_tau_s, layer_1, layer_2, N_c
            )
            if outputs["T_e"] >= 20e6:
                seed = random.randrange(0, 101, 2)
                femm.mi_saveas("IPM_new_" + str(seed) + ".fem")
        #
        except:

            #print(r_g, l_s, h_s, p, g, h_ys, h_yr, N_c, I_s, h_m, ratio)
            outputs = bad_inputs(outputs)


#
