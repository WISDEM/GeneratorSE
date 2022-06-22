# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:28:2 2021

@author: lsethura
"""

import femm
import numpy as np
import openmdao.api as om
from sympy import Point, Line, Segment, Polygon
from sympy.geometry.util import centroid
#import random
import os
import platform

def myopen():
    if platform.system().lower() == 'windows':
        femm.openfemm(1)
    else:
        femm.openfemm(winepath=os.environ["WINEPATH"], femmpath=os.environ["FEMMPATH"])
    femm.smartmesh(0)


def bad_inputs(outputs):
    #print("Bad inputs for geometry")
    outputs["B_g"] = 30.0
    outputs["B_rymax"] = 30.0
    outputs["B_smax"] = 30.0
    outputs["Sigma_normal"] = 1e9
    outputs["M_Fes"] = 1e9
    outputs["M_Fer"] = 1e9
    outputs["Iron"] = 1e9
    outputs["T_e"] = 1e9
    outputs["Sigma_shear"] = 1e9
    femm.mi_saveas("IPM_new_bad_geomtery.fem")
    return outputs


def run_post_process(D_a, g, r_outer, h_yr, h_ys, r_inner, theta_p_r):

    theta_p_d = np.rad2deg(theta_p_r)
    femm.mi_loadsolution()
    femm.mo_smooth("off")

    femm.mo_selectblock(
        (r_inner + h_ys * 0.5) * np.cos(theta_p_r * 2.5),
        (r_inner + h_ys * 0.5) * np.sin(theta_p_r * 2.5),
    )

    V_stator = femm.mo_blockintegral(10)
    femm.mo_clearblock()
    sy_area = []
    ry_area = []
    numelm2 = femm.mo_numelements()
    for r in range(1, numelm2):
        p3, p4, p5, x1, y1, a1, g1 = femm.mo_getelement(r)
        mag = np.sqrt(x1**2 + y1**2)
        if mag <= D_a / 2:
            a = femm.mo_getb(x1, y1)
            sy_area.append(np.sqrt(a[0] ** 2 + a[1] ** 2))
        elif mag >= (D_a / 2 + g):
            b = femm.mo_getb(x1, y1)
            ry_area.append(np.sqrt(b[0] ** 2 + b[1] ** 2))

    B_symax = max(sy_area)
    B_rymax = max(ry_area)

    femm.mo_addcontour((D_a / 2 + g * 0.5) * np.cos(0), (D_a / 2 + g * 0.5) * np.sin(0))
    femm.mo_addcontour((D_a / 2 + g * 0.5) * np.cos(theta_p_r * 5), (D_a / 2 + g * 0.5) * np.sin(theta_p_r * 5))
    femm.mo_bendcontour(theta_p_d * 5, 1)
    femm.mo_makeplot(1, 1000, "gap.csv", 1)
    femm.mo_makeplot(2, 100, "B_r.csv", 1)
    femm.mo_makeplot(3, 100, "B_t.csv", 1)
    femm.mo_clearcontour()
    # #
    femm.mo_selectblock(
        (r_outer - h_yr * 0.5) * np.cos(theta_p_r * 2.5),
        (r_outer - h_yr * 0.5) * np.sin(theta_p_r * 2.5),
    )
    V_rotor = femm.mo_blockintegral(10)

    B_g_peak = np.loadtxt("gap.csv")[:, 1].max()
    B_r_normal = np.loadtxt("B_r.csv")
    B_t_normal = np.loadtxt("B_t.csv")

    circ = B_r_normal[-1, 0]
    delta_L = np.diff(B_r_normal[:, 0])[0]

    force = np.sum((B_r_normal[:, 1]) ** 2 - (B_t_normal[:, 1]) ** 2) * delta_L
    sigma_n = abs(1 / (8 * np.pi * 1e-7) * force) / circ

    return B_g_peak, B_rymax, B_symax, sigma_n, V_rotor, V_stator


def B_r_B_t(Theta_elec, D_a, l_s, p1, g, theta_p_r, I_s, theta_tau_s, layer_1, layer_2, N_c, tau_p):

    theta_p_d = np.rad2deg(theta_p_r)

    myopen()
    femm.opendocument("IPM_new.fem")
    femm.mi_modifycircprop("A+", 1, I_s * np.sin(2 * np.pi / 3))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(0 * np.pi / 3))
    femm.mi_modifycircprop("C+", 1, I_s * np.sin(4 * np.pi / 3))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(2 * np.pi / 3))
    femm.mi_modifycircprop("B-", 1, -I_s * np.sin(0 * np.pi / 3))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(4 * np.pi / 3))

    femm.mi_saveas("IPM_new_I1.fem")
    femm.smartmesh(0)
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.mi_loadsolution()

    femm.mo_addcontour((D_a / 2 + g * 0.5) * np.cos(0), (D_a / 2 + g * 0.5) * np.sin(0))
    femm.mo_addcontour((D_a / 2 + g * 0.5) * np.cos(theta_p_r * 5), (D_a / 2 + g * 0.5) * np.sin(theta_p_r * 5))
    femm.mo_bendcontour(theta_p_d * 5, 1)
    femm.mo_makeplot(2, 100, "B_r_1.csv", 1)
    femm.mo_makeplot(3, 100, "B_t_1.csv", 1)
    femm.mo_clearcontour()
    femm.mo_close()
    myopen()
    femm.opendocument("IPM_new_I1.fem")

    Phases1 = ["A+", "B+", "B-", "C-", "C+", "A+"]
    Phases2 = ["B-", "B+", "C+", "C-", "A-", "A+"]

    angle_r = theta_tau_s * 0.5
    delta_theta = theta_tau_s
    for pitch in range(1, 7):
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
    femm.mi_modifycircprop("A+", 1, I_s * np.sin(Theta_elec + 2 * np.pi / 3))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(Theta_elec + 0 * np.pi / 3))
    femm.mi_modifycircprop("C+", 1, I_s * np.sin(Theta_elec + 4 * np.pi / 3))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(Theta_elec + 2 * np.pi / 3))
    femm.mi_modifycircprop("B-", 1, -I_s * np.sin(Theta_elec + 0 * np.pi / 3))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(Theta_elec + 4 * np.pi / 3))

    femm.mi_saveas("IPM_new_I2.fem")
    femm.smartmesh(0)
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_addcontour((D_a / 2 + g * 0.5) * np.cos(0), (D_a / 2 + g * 0.5) * np.sin(0))
    femm.mo_addcontour((D_a / 2 + g * 0.5) * np.cos(theta_p_r * 5), (D_a / 2 + g * 0.5) * np.sin(theta_p_r * 5))
    femm.mo_bendcontour(theta_p_d * 5, 1)
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
    torque = np.pi / 2 * sigma_t * D_a**2 * l_s
    #torque_ripple = (torque[0] - torque[1]) * 100 / torque.mean()
    #print(torque[0], torque[1], torque.mean())
    return torque.mean(), sigma_t.mean()


class FEMM_Geometry(om.ExplicitComponent):
    def setup(self):
        self.add_discrete_input("m", 3, desc="number of phases")
        #self.add_input("k_sfil", 0.65, desc="slot fill factor")

        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("p1", 0.0, desc="pole pairs ")
        #self.add_input("b", 0.0, desc="v-angle")
        #self.add_input("c", 0.0, desc="pole pairs ")
        self.add_input("h_s1", 0.010, desc="Slot Opening height")
        self.add_input("h_s2", 0.010, desc="Wedge Opening height")
        self.add_input("g", 0.0, units="m", desc="air gap length ")
        self.add_input("alpha_v", 0.0, units="deg", desc="v-angle")
        self.add_output("b_s", 0.0, units="m", desc="slot width")
        self.add_input("alpha_m", 0.5, desc="pole pair width")
        self.add_input("tau_s", 0.0, units="m", desc="Pole pitch")

        #self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("f", 0.0, units="Hz", desc="frequency")
        self.add_input("l_fe_ratio", 0, desc="bridge length")
        self.add_input("ratio", 0.0, desc="pole to bridge ratio")
        self.add_input("h_m", 0.0, units="m", desc="magnet height")
        self.add_output("l_m", 0.0, units="m", desc="magnet length")
        self.add_output("tau_p", 0.0, units="m", desc="pole pitch")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("h_ys", 0.0, units="m", desc="stator yoke height")
        self.add_input("h_t", 0.0, units="m", desc="stator tooth height")
        self.add_output("h_s", 0.0, units="m", desc="stator tooth height")
        self.add_input("r_g", 0.0, units="m", desc="air gap radius ")
        self.add_input("D_a", 0.0, units="m", desc="Stator outer diameter")
        self.add_input("I_s", 0.0, units="A", desc="Stator current ")
        self.add_input("N_c", 0.0, desc="Number of turns per coil in series")
        #self.add_input("J_s", 0.0, units="A/mm**2", desc="Conductor cross-section")
        self.add_input("d_mag", 0.0, units="m", desc=" magnet distance from inner radius")
        self.add_input("d_sep", 0.0, units="m", desc=" bridge separation width")
        self.add_input("m_sep", 0.0, units="m", desc=" bridge separation width")
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        self.add_output("Slot_aspect_ratio", 0.0, desc="Slot aspect ratio")
        self.add_output("B_g", 0.0, units="T", desc="Peak air gap flux density ")
        self.add_output("B_rymax", 0.0, units="T", desc="Peak Rotor yoke flux density")
        self.add_output("B_smax", 0.0, units="T", desc="Peak flux density in the stator")
        self.add_output("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_output("Sigma_shear", 0.0, units="Pa", desc="Shear stress")
        self.add_output("Sigma_normal", 0.0, units="Pa", desc="Normal stress")
        self.add_output("Iron", 0.0, units="kg", desc="magnet length ")
        self.add_output("M_Fes", 0.0, units="kg", desc="mstator iron mass ")
        self.add_output("M_Fer", 0.0, units="kg", desc="rotor iron mass ")
        self.add_output("M_Fest", 0.0, units="kg", desc="rotor iron mass ")
        self.add_output("r_outer_active", 0.0, units="m", desc="rotor outer diameter ")
        self.add_output("r_mag_center", 0.0, units="m", desc="rotor magnet radius ")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack variables
        m = int( discrete_inputs["m"] )
        l_s = float( inputs["l_s"] )
        p1 = float( inputs["p1"] )
        #b = float( inputs["b"] )
        h_s1 = float( inputs["h_s1"] )
        h_s2 = float( inputs["h_s2"] )
        g = float( inputs["g"] )
        alpha_v = float( inputs["alpha_v"] )
        alpha_m = float( inputs["alpha_m"] )
        tau_s = float( inputs["tau_s"] )
        f = float( inputs["f"] )
        l_fe_ratio = float( inputs["l_fe_ratio"] )
        ratio = float( inputs["ratio"] )
        h_m = float( inputs["h_m"] )
        h_yr = float( inputs["h_yr"] )
        h_ys = float( inputs["h_ys"] )
        h_t = float( inputs["h_t"] )
        r_g = float( inputs["r_g"] )
        D_a = float( inputs["D_a"] )
        I_s = float( inputs["I_s"] )
        N_c = float( inputs["N_c"] )
        d_mag = float( inputs["d_mag"] )
        d_sep = float( inputs["d_sep"] )
        m_sep = float( inputs["m_sep"] )
        rho_Fe = float( inputs["rho_Fe"] )

        #q = b / c
        #Slots = 2 * q * m * p1
        outputs["h_s"] = h_s = h_t - h_s1 - h_s2

        #Slots_pp = q * m

        theta_p_d = 180.0 / p1
        theta_p_r = np.deg2rad(theta_p_d)
        #print(theta_p_d, theta_p_r)
        r_a = 0.5*D_a
        b_so = 2 * g

        outputs["tau_p"] = tau_p = np.pi * r_g / p1

        bs_taus = 0.5
        outputs["b_s"] = b_s = bs_taus * tau_s
        theta_b_s = np.arctan(b_s / (r_a))
        theta_tau_s = theta_p_r * 5 / 6

        theta_tau_s_new = np.arctan((tau_s * 0.5 - b_so * 0.5) / (r_a))
        theta_tau_s_new2 = np.arctan((tau_s * 0.5 - b_s * 0.5) / (r_a))
        #theta_tau_s_new3 = np.arctan((tau_s * 0.5 + b_so * 0.5) / (r_a))
        theta_tau_s_new4 = np.arctan((tau_s * 0.5 + b_s * 0.5) / (r_a))

        #theta_b_so = b_so / (r_a)

        myopen()
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
        #b = [0, 1.27, 1.31, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39]
        #h = [
        #    0.000000,
        #    82000.000000,
        #    232000.000000,
        #    392000.000000,
        #    557000.000000,
        #    717000.000000,
        #    872000.000000,
        #    1037000.000000,
        #    1352000.000000,
        #    1512000.000000,
        #]

        femm.mi_modifymaterial("N48SH", 5, 0.7142857)

        #        femm.mi_clearbhpoints("N48SH")
        #        for i in range (0,10):
        #           femm.mi_addbhpoint('N48SH',b[i],h[i])

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

        theta_m = np.deg2rad(theta_p_d * alpha_m * 0.5)

        #ratio_1 = (1 - alpha_m) * 1.5 / ratio
        #theta_m_1 = theta_m * ratio_1
        #theta_m_2 = theta_p_r / 2 * ratio
        theta_rb = np.deg2rad(180 / p1) - theta_m * 2

        theta_m_r = theta_rb / 2

        xmag1 = 5 * np.cos(theta_p_r)
        ymag1 = 5 * np.sin(theta_p_r)

        #xmag2 = 5 * np.cos(0.5*theta_p_r)
        #ymag2 = 5 * np.sin(0.5*theta_p_r)

        #theta_L = 180 - 0.5 * alpha_v - np.rad2deg(theta_m)

        theta_m_r2 = l_fe_ratio * theta_m_r


        # Parametrization of the magnets

        def rotate(xo, yo, xp, yp, angle):
            ## Rotate a point clockwise by a given angle around a given origin.
            # angle *= -1.
            qx = xo + np.cos(angle) * (xp - xo) - np.sin(angle) * (yp - yo)
            qy = yo + np.sin(angle) * (xp - xo) + np.cos(angle) * (yp - yo)
            return qx, qy

        # We need 14 points to define the geometry, the first one is the center of the rotor at 0,0
        points = np.zeros((14,2))
        # Second point stays at y=0 and x=radius of the armature
        # points[1,:] = np.array([r_i, 0.])
        # r2 = r_i + air_gap
        # points[2,:] = np.array([r2, 0])
        # r3 = r_i + air_gap + struct_offset
        # points[3,:] = rotate(points[0,0], points[0,1], r3, 0, 0.5 * (alpha_p - alpha_pr))
        # r6 = r3 + d_mag
        # points[6,:] = rotate(points[0,0], points[0,1], r6, 0, 0.5 * alpha_p)

        # r7 = r6 + slot_height

        # outer_gen_diameter = r7+struct_offset
        # points[5,:] = np.array([outer_gen_diameter,0])

        # points[7,:] = rotate(points[0,0], points[0,1], r7, 0, 0.5 * alpha_p)
        # points[4,:] = rotate(points[0,0], points[0,1], r3 + slot_height, 0, 0.5 * (alpha_p - alpha_pr))

        # p6p3_angle = np.arctan((points[6,1]-points[3,1])/(points[6,0]-points[3,0]))
        # p6p3p4_angle = p6p3_angle - 0.5 * (alpha_p - alpha_pr)
        # p4r = rotate(points[3,0], points[3,1], points[4,0], points[4,1], -0.5 * (alpha_p - alpha_pr))
        # p11r =  (p4r[0] - points[3,0]) * np.cos(p6p3p4_angle) * np.array([np.cos(p6p3p4_angle), np.sin(p6p3p4_angle)]) + points[3,:]
        # points[11,:] = rotate(points[3,0], points[3,1], p11r[0], p11r[1], 0.5 * (alpha_p - alpha_pr))

        # mml = (points[6,1] - points[11,1]) / np.sin(p6p3_angle) # max magnet length
        # points[8,:] = points[4,:] + mml * np.array([np.cos(p6p3_angle),np.sin(p6p3_angle)])
        # ml = mml * magnet_l_pc
        # points[9,:] = points[4,:] + ml * np.array([np.cos(p6p3_angle),np.sin(p6p3_angle)])
        # points[10,:] = points[11,:] + ml * np.array([np.cos(p6p3_angle),np.sin(p6p3_angle)])




        x0, y0 = points[0,:]

        theta_1 = theta_p_r * 0.5 - d_sep / (r_g + d_mag)
        X_1, Y_1 = (r_g + d_mag) * np.cos(theta_1), (r_g + d_mag) * np.sin(theta_1)

        bridge_thickness = h_m * np.sin(np.pi / 2) / np.sin(np.deg2rad(alpha_v * 0.5))

        X_2 = (r_g + d_mag + bridge_thickness) * np.cos(theta_1)
        Y_2 = (r_g + d_mag + bridge_thickness) * np.sin(theta_1)
        xc = (r_g + d_mag + bridge_thickness) * np.cos(theta_p_r * 0.5)
        yc = (r_g + d_mag + bridge_thickness) * np.sin(theta_p_r * 0.5)
        m1 = np.tan(np.deg2rad(180 - (180 - alpha_v * 0.5 - np.rad2deg(theta_1))))

        X_3 = (m1 * X_1 - Y_1) / m1

        X_5 = (r_g + d_mag + bridge_thickness) * np.cos(theta_m_r - theta_m_r2)
        Y_5 = (r_g + d_mag + bridge_thickness) * np.sin(theta_m_r - theta_m_r2)

        line1 = Line(Point(X_1, Y_1, evaluate=False), Point(X_3, 0, evaluate=False))
        line3 = Line(Point(0, 0, evaluate=False), Point(X_5, Y_5, evaluate=False))

        r3 = line1.intersection(line3)

        #x4, y4 = 7 * np.cos(theta_m_2), 7 * np.sin(theta_m_2)
        rad = (r3[0].x ** 2 + r3[0].y ** 2) ** 0.5
        #mag_dir2 = np.arctan(float(m)) * 180 / np.pi + 180

        line2 = Segment(Point(X_1, Y_1, evaluate=False), Point(r3[0].x, r3[0].y, evaluate=False))

        m_sep = m_sep / line2.length
        xstart, ystart = (1 - m_sep) * X_1 + m_sep * r3[0].x, (1 - m_sep) * Y_1 + m_sep * r3[0].y

        angle = np.arctan(float(ystart) / float(xstart))

        line4 = Segment(Point(xstart, ystart, evaluate=False), Point(r3[0].x, r3[0].y, evaluate=False))
        Total_avail_len = line4.length
        m = -1 / (line4.slope)
        # mag_dir=np.arctan(float(m))*180/np.pi

        end_dist = (Total_avail_len - ((bridge_thickness**2 - h_m**2) ** 0.5)) * ratio / Total_avail_len
        xend, yend = (1 - end_dist) * xstart + end_dist * r3[0].x, (1 - end_dist) * ystart + end_dist * r3[0].y
        mag = Segment(Point(xstart, ystart, evaluate=False), Point(xend, yend, evaluate=False))
        outputs["l_m"] = mag.length

        femm.mi_addnode(X_1, Y_1)
        femm.mi_selectnode(X_1, Y_1)
        femm.mi_setgroup(1)
        femm.mi_addnode(X_2, Y_2)
        femm.mi_selectnode(X_2, Y_2)
        femm.mi_setgroup(1)
        femm.mi_addnode(r3[0].x, r3[0].y)
        femm.mi_selectnode(r3[0].x, r3[0].y)
        femm.mi_setgroup(1)
        femm.mi_addnode(xstart, ystart)
        femm.mi_selectnode(xstart, ystart)
        femm.mi_setgroup(1)
        femm.mi_addnode(xend, yend)
        femm.mi_selectnode(xend, yend)
        femm.mi_setgroup(1)

        X_7 = xstart + h_m / (1 + m**2) ** 0.5
        Y_7 = m * (X_7 - xstart) + ystart
        femm.mi_addnode(X_7, Y_7)
        femm.mi_selectnode(X_7, Y_7)
        femm.mi_setgroup(1)
        femm.mi_addsegment(X_1, Y_1, X_2, Y_2)
        femm.mi_selectsegment(X_2, Y_2)
        femm.mi_setgroup(1)

        femm.mi_addsegment(X_1, Y_1, xstart, ystart)

        femm.mi_selectsegment(xstart, ystart)
        femm.mi_setgroup(1)
        femm.mi_addsegment(xstart, ystart, xend, yend)
        femm.mi_selectsegment(xend, yend)
        femm.mi_setgroup(1)
        femm.mi_addsegment(xstart, ystart, X_7, Y_7)

        femm.mi_selectsegment(X_7, Y_7)
        femm.mi_setgroup(1)
        femm.mi_addsegment(X_2, Y_2, X_7, Y_7)
        g2 = Segment(Point(X_2, Y_2, evaluate=False), Point(X_7, Y_7, evaluate=False))
        l2 = g2.midpoint.evalf()
        femm.mi_selectsegment(l2.x, l2.y)
        femm.mi_setgroup(1)

        if ratio == 1:
            femm.mi_addsegment(r3[0].x, r3[0].y, xend, yend)
            g3 = Segment(Point(r3[0].x, r3[0].y, evaluate=False), Point(xend, yend, evaluate=False))
            l3 = g3.midpoint.evalf()
            femm.mi_selectsegment(l3.x, l3.y)
            femm.mi_setgroup(1)
            X_8 = xend + h_m / (1 + m**2) ** 0.5
            Y_8 = m * (X_8 - xend) + yend
            femm.mi_addnode(X_8, Y_8)
            femm.mi_selectnode(X_8, Y_8)
            femm.mi_setgroup(1)
            femm.mi_addsegment(X_8, Y_8, xend, yend)
            g4 = Segment(Point(xend, yend, evaluate=False), Point(X_8, Y_8, evaluate=False))
            l4 = g4.midpoint.evalf()
            mag_dir = np.arctan(float(g4.slope)) * 180 / np.pi
            femm.mi_selectsegment(l4.x, l4.y)
            femm.mi_setgroup(1)
            femm.mi_addsegment(r3[0].x, r3[0].y, X_8, Y_8)
            g5 = Segment(Point(r3[0].x, r3[0].y, evaluate=False), Point(X_8, Y_8, evaluate=False))
            l5 = g5.midpoint.evalf()
            femm.mi_selectsegment(l5.x, l5.y)
            femm.mi_setgroup(1)
            femm.mi_addsegment(X_8, Y_8, X_7, Y_7)
            g6 = Segment(Point(X_7, Y_7, evaluate=False), Point(X_8, Y_8, evaluate=False))
            l6 = g6.midpoint.evalf()
            femm.mi_selectsegment(l6.x, l6.y)

            femm.mi_setgroup(1)
            air_1 = Polygon(
                Point(xend, yend, evaluate=False),
                Point(r3[0].x, r3[0].y, evaluate=False),
                Point(X_8, Y_8, evaluate=False),
            )
            mag = Polygon(Point(xstart, ystart, evaluate=False), Point(X_8, Y_8, evaluate=False))
            l1 = mag.midpoint.evalf()

        else:

            angle = np.arctan(float(r3[0].y) / float(r3[0].x))
            X_6, Y_6 = (rad + bridge_thickness) * np.cos(angle), (rad + bridge_thickness) * np.sin(angle)
            X_8 = xend + h_m / (1 + m**2) ** 0.5
            Y_8 = m * (X_8 - xend) + yend
            femm.mi_addnode(X_8, Y_8)
            femm.mi_selectnode(X_8, Y_8)

            femm.mi_setgroup(1)
            femm.mi_addsegment(xend, yend, X_8, Y_8)
            femm.mi_selectsegment(X_8, Y_8)
            femm.mi_setgroup(1)
            g4 = Segment(Point(xend, yend, evaluate=False), Point(X_8, Y_8, evaluate=False))
            l4 = g4.midpoint.evalf()
            line4 = Segment(Point(xstart, ystart, evaluate=False), Point(X_7, Y_7, evaluate=False))
            m = line4.slope
            mag_dir = np.arctan(float(m)) * 180 / np.pi
            femm.mi_selectsegment(l4.x, l4.y)
            femm.mi_setgroup(1)
            femm.mi_addsegment(r3[0].x, r3[0].y, xend, yend)
            g5 = Segment(Point(xend, yend, evaluate=False), Point(r3[0].x, r3[0].y, evaluate=False))
            l5 = g5.midpoint.evalf()
            femm.mi_selectsegment(l5.x, l5.y)
            femm.mi_setgroup(1)
            femm.mi_addsegment(X_8, Y_8, X_7, Y_7)
            g6 = Segment(Point(X_8, Y_8, evaluate=False), Point(X_7, Y_7, evaluate=False))
            l6 = g6.midpoint.evalf()
            femm.mi_selectsegment(l6.x, l6.y)
            femm.mi_setgroup(1)
            femm.mi_addnode(X_6, Y_6)
            femm.mi_selectnode(X_6, Y_6)
            femm.mi_setgroup(1)
            femm.mi_addsegment(X_8, Y_8, X_6, Y_6)
            g7 = Segment(Point(X_8, Y_8, evaluate=False), Point(X_6, Y_6, evaluate=False))
            l7 = g7.midpoint.evalf()
            femm.mi_selectsegment(l7.x, l7.y)
            femm.mi_setgroup(1)

            femm.mi_addsegment(r3[0].x, r3[0].y, X_6, Y_6)
            g8 = Segment(Point(r3[0].x, r3[0].y, evaluate=False), Point(X_6, Y_6, evaluate=False))
            l8 = g8.midpoint.evalf()
            femm.mi_selectsegment(l8.x, l8.y)
            femm.mi_setgroup(1)
            air_1 = Polygon(
                Point(xend, yend, evaluate=False),
                Point(r3[0].x, r3[0].y, evaluate=False),
                Point(X_6, Y_6, evaluate=False),
                Point(X_8, Y_8, evaluate=False),
            )
            mag = Polygon(Point(xstart, ystart, evaluate=False), Point(X_8, Y_8, evaluate=False))
            l1 = mag.midpoint.evalf()

        air_2 = Polygon(
            Point(X_1, Y_1, evaluate=False),
            Point(X_2, Y_2, evaluate=False),
            Point(X_7, Y_7, evaluate=False),
            Point(xstart, ystart, evaluate=False),
        )
        femm.mi_addblocklabel(centroid(air_1).evalf().x, centroid(air_1).evalf().y)
        femm.mi_selectlabel(centroid(air_1).evalf().x, centroid(air_1).evalf().y)
        femm.mi_setblockprop("Air", 1, 1, 0, 0, 1, 0)
        femm.mi_clearselected()

        femm.mi_addblocklabel(l1.x, l1.y)
        femm.mi_selectlabel(l1.x, l1.y)
        femm.mi_setblockprop("N48SH", 1, 1, 0, mag_dir, 1, 0)
        femm.mi_clearselected()

        femm.mi_addblocklabel(centroid(air_2).evalf().x, centroid(air_2).evalf().y)
        femm.mi_selectlabel(centroid(air_2).evalf().x, centroid(air_2).evalf().y)
        femm.mi_setblockprop("Air", 1, 1, 0, 0, 1, 0)
        femm.mi_clearselected()
        outputs["r_mag_center"] = r_mag_label = (centroid(mag).evalf().x ** 2 + centroid(mag).evalf().y ** 2) ** 0.5

        femm.mi_selectgroup(1)
        femm.mi_mirror(x0, y0, xc, yc)
        femm.mi_clearselected()

        mag1 = np.sqrt(X_2**2 + Y_2**2)
        mag2 = np.sqrt(X_1**2 + Y_2**2)
        x9, y9 = femm.mi_selectnode(mag1 * np.cos(theta_p_r - angle), mag1 * np.sin(theta_p_r - angle))
        x10, y10 = femm.mi_selectnode(mag2 * np.cos(theta_p_r - angle), mag2 * np.sin(theta_p_r - angle))
        line6 = Segment(Point(x9, y9, evaluate=False), Point(x10, y10, evaluate=False))
        # mag_dir2=line6.slope*180/np.pi
        #        femm.mi_clearselected()
        femm.mi_selectgroup(1)
        femm.mi_mirror(x0, y0, xmag1, ymag1)

        femm.mi_clearselected()
        outputs["r_mag_center"] = r_mag_label = (centroid(mag).evalf().x ** 2 + centroid(mag).evalf().y ** 2) ** 0.5
        r_center = (xc**2 + yc**2) ** 0.5
        femm.mi_selectlabel(
            r_mag_label * np.cos(1.25*theta_p_r + 0.5*theta_m_r),
            r_mag_label * np.sin(1.25*theta_p_r + 0.5*theta_m_r),
        )
        femm.mi_setblockprop("N48SH", 1, 1, 0, mag_dir + theta_p_d + 180, 1, 0)
        femm.mi_clearselected()
        femm.mi_selectlabel(
            r_mag_label * np.cos(1.75*theta_p_r),
            r_mag_label * np.sin(1.75*theta_p_r),
        )
        femm.mi_deleteselected()
        femm.mi_selectlabel(
            r_mag_label * np.cos(1.25*theta_p_r),
            r_mag_label * np.sin(1.25*theta_p_r),
        )
        femm.mi_mirror(
            x0,
            y0,
            (r_g + d_mag + bridge_thickness) * np.cos(1.5*theta_p_r),
            (r_g + d_mag + bridge_thickness) * np.sin(1.5*theta_p_r),
        )
        femm.mi_clearselected()

        femm.mi_selectgroup(1)
        femm.mi_copyrotate(x0, y0, theta_p_d * 2, 1)
        femm.mi_setgroup(1)
        femm.mi_clearselected()

        outputs["r_outer_active"] = r_outer = r_center + h_yr
        femm.mi_seteditmode("group")
        femm.mi_selectrectangle(
            (r_g - g * 0.5) * np.cos(theta_p_r),
            (r_g - g * 0.5) * np.sin(theta_p_r),
            r_outer * np.cos(np.deg2rad(-0.5)),
            r_outer * np.sin(np.deg2rad(-0.5)),
        )

        femm.mi_copyrotate(x0, y0, theta_p_d * 4, 1)
        femm.mi_clearselected()
        seg = Segment(
            Point(r3[0].x, r3[0].y, evaluate=False),
            Point((rad + bridge_thickness) * np.cos(angle), (rad + bridge_thickness) * np.sin(angle), evaluate=False),
        )
        l11 = seg.midpoint.evalf()
        femm.mi_selectsegment(l11.x, l11.y)
        femm.mi_copyrotate(x0, y0, theta_p_d * 4, 1)
        femm.mi_clearselected()

        femm.mi_addnode(r_g * np.cos(0), r_g * np.sin(0))

        femm.mi_addnode(r_g * np.cos(theta_p_r * 5), r_g * np.sin(theta_p_r * 5))
        femm.mi_addarc(
            r_g * np.cos(0),
            r_g * np.sin(0),
            r_g * np.cos(theta_p_r * 5),
            r_g * np.sin(theta_p_r * 5),
            theta_p_d * 5,
            1,
        )

        femm.mi_selectgroup(1)

        femm.mi_addnode(r_outer * np.cos(0), r_outer * np.sin(0))
        femm.mi_addnode(r_outer * np.cos(theta_p_r * 5), r_outer * np.sin(theta_p_r * 5))
        femm.mi_addarc(
            r_outer * np.cos(0),
            r_outer * np.sin(0),
            r_outer * np.cos(theta_p_r * 5),
            r_outer * np.sin(theta_p_r * 5),
            theta_p_d * 5,
            1,
        )

        # Draw the stator slots and Stator coils

        X_9, Y_9 = r_a, 0
        X_10, Y_10 = r_a * np.cos(theta_tau_s_new), r_a * np.sin(theta_tau_s_new)
        X_11, Y_11 = (r_a - h_s1) * np.cos(theta_tau_s_new), (r_a - h_s1) * np.sin(theta_tau_s_new)
        X_12, Y_12 = (r_a - h_s1 - h_s2) * np.cos(theta_tau_s_new2), (r_a - h_s1 - h_s2) * np.sin(theta_tau_s_new2)
        X_13, Y_13 = (r_a - h_s1 - h_s2 - h_s / 2) * np.cos(theta_tau_s_new2), (r_a - h_s1 - h_s2 - h_s / 2) * np.sin(
            theta_tau_s_new2
        )
        X_14, Y_14 = (r_a - h_s1 - h_s2 - h_s) * np.cos(theta_tau_s_new2), (r_a - h_s1 - h_s2 - h_s) * np.sin(
            theta_tau_s_new2
        )
        #X_15, Y_15 = (r_a) * np.cos(theta_tau_s_new3), (r_a) * np.sin(theta_tau_s_new3)
        #X_16, Y_16 = (r_a - h_s1) * np.cos(theta_tau_s_new3), (r_a - h_s1) * np.sin(theta_tau_s_new3)
        X_17, Y_17 = (r_a - h_s1 - h_s2) * np.cos(theta_tau_s_new4), (r_a - h_s1 - h_s2) * np.sin(theta_tau_s_new4)
        X_18, Y_18 = (r_a - h_s1 - h_s2 - h_s / 2) * np.cos(theta_tau_s_new4), (r_a - h_s1 - h_s2 - h_s / 2) * np.sin(
            theta_tau_s_new4
        )
        X_19, Y_19 = (r_a - h_s1 - h_s2 - h_s) * np.cos(theta_tau_s_new4), (r_a - h_s1 - h_s2 - h_s) * np.sin(
            theta_tau_s_new4
        )

        X_21, Y_21 = (r_a - h_s1 - h_s2 - h_s - h_ys), 0
        X_22, Y_22 = (r_a - h_s1 - h_s2 - h_s - h_ys) * np.cos(theta_p_r * 5), (
            r_a - h_s1 - h_s2 - h_s - h_ys
        ) * np.sin(theta_p_r * 5)

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
        femm.mi_copyrotate(x0, y0, np.rad2deg(theta_tau_s), 5)
        femm.mi_clearselected()

        femm.mi_addnode(X_21, Y_21)
        femm.mi_selectnode(X_21, Y_21)
        femm.mi_setgroup(2)

        femm.mi_addnode(X_22, Y_22)
        femm.mi_selectnode(X_22, Y_22)
        femm.mi_setgroup(2)

        femm.mi_addarc(X_21, Y_21, X_22, Y_22, theta_p_d * 5, 1)
        femm.mi_selectarcsegment(X_22, Y_22)
        femm.mi_setarcsegmentprop(10, "Dirichlet", 0, 50)
        femm.mi_clearselected()

        femm.mi_addsegment(X_9, Y_9, X_21, Y_21)
        femm.mi_selectsegment(X_21, Y_21)
        femm.mi_setgroup(13)
        femm.mi_clearselected()
        femm.mi_selectsegment(X_21, Y_21)
        femm.mi_setsegmentprop("apbc1", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_addsegment(X_9, Y_9, r_g, 0)
        femm.mi_selectsegment(r_g, 0)
        femm.mi_setgroup(13)
        femm.mi_clearselected()
        femm.mi_selectsegment(r_g, 0)
        femm.mi_setsegmentprop("apbc2", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_addnode(r_outer, 0)
        femm.mi_selectnode(r_outer, 0)
        femm.mi_setgroup(2)

        femm.mi_addsegment(r_g, 0, r_outer, 0)
        femm.mi_selectsegment(r_outer, 0)
        femm.mi_setgroup(13)

        femm.mi_setsegmentprop("apbc3", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_selectgroup(13)
        femm.mi_copyrotate(x0, y0, theta_p_d * 5, 1)
        femm.mi_clearselected()

        femm.mi_selectarcsegment(r_outer * np.cos(theta_p_r * 2.5), r_outer * np.sin(theta_p_r * 2.5))
        femm.mi_setarcsegmentprop(10, "Dirichlet", 0, 50)
        femm.mi_clearselected()

        femm.mi_addblocklabel(
            (r_a - h_s1 - h_s2 - h_s - h_ys * 0.5) * np.cos(theta_p_r * 2.5),
            (r_a - h_s1 - h_s2 - h_s - h_ys * 0.5) * np.sin(theta_p_r * 2.5),
        )
        femm.mi_selectlabel(
            (r_a - h_s1 - h_s2 - h_s - h_ys * 0.5) * np.cos(theta_p_r * 2.5),
            (r_a - h_s1 - h_s2 - h_s - h_ys * 0.5) * np.sin(theta_p_r * 2.5),
        )
        femm.mi_setblockprop("M-36 Steel")
        femm.mi_clearselected()

        femm.mi_addblocklabel(
            (r_outer - 0.5 * h_yr) * np.cos(theta_p_r * 2.5), (r_outer - 0.5 * h_yr) * np.sin(theta_p_r * 2.5)
        )
        femm.mi_selectlabel(
            (r_outer - 0.5 * h_yr) * np.cos(theta_p_r * 2.5), (r_outer - 0.5 * h_yr) * np.sin(theta_p_r * 2.5)
        )
        femm.mi_setblockprop("M-36 Steel")
        femm.mi_clearselected()

        femm.mi_addblocklabel(
            (D_a / 2 + (r_g - D_a / 2) * 0.5) * np.cos(theta_p_r * 2.5),
            (D_a / 2 + (r_g - D_a / 2) * 0.5) * np.sin(theta_p_r * 2.5),
        )
        femm.mi_selectlabel(
            (D_a / 2 + (r_g - D_a / 2) * 0.5) * np.cos(theta_p_r * 2.5),
            (D_a / 2 + (r_g - D_a / 2) * 0.5) * np.sin(theta_p_r * 2.5),
        )
        femm.mi_setblockprop("Air")
        femm.mi_clearselected()

        ##        femm.mi_addblocklabel((yoke_radius*0.5)*np.cos(theta_p_r*2.5),(yoke_radius*0.5)*np.sin(theta_p_r*2.5))
        ##        femm.mi_selectlabel((yoke_radius*0.5)*np.cos(theta_p_r*2.5),(yoke_radius*0.5)*np.sin(theta_p_r*2.5))
        ##        femm.mi_setblockprop("Air")
        ##        femm.mi_clearselected()
        ##
        ##        femm.mi_addblocklabel((r_outer*1.5)*np.cos(theta_p_r*2.5),(r_outer*1.5)*np.sin(theta_p_r*2.5))
        ##        femm.mi_selectlabel((r_outer*1.5)*np.cos(theta_p_r*2.5),(r_outer*1.5)*np.sin(theta_p_r*2.5))
        ##        femm.mi_setblockprop("Air")
        ##        femm.mi_clearselected()

        layer_1 = r_a - h_s1 - h_s2 - h_s * 0.25
        layer_2 = r_a - h_s1 - h_s2 - h_s * 0.75

        femm.mi_addblocklabel(layer_1 * np.cos(theta_tau_s * 0.5), layer_1 * np.sin(theta_tau_s * 0.5))
        femm.mi_selectlabel(layer_1 * np.cos(theta_tau_s * 0.5), layer_1 * np.sin(theta_tau_s * 0.5))
        femm.mi_copyrotate(0, 0, np.rad2deg(theta_tau_s), 5)

        femm.mi_addblocklabel(layer_2 * np.cos(theta_tau_s * 0.5), layer_2 * np.sin(theta_tau_s * 0.5))
        femm.mi_selectlabel(layer_2 * np.cos(theta_tau_s * 0.5), layer_2 * np.sin(theta_tau_s * 0.5))
        femm.mi_copyrotate(0, 0, np.rad2deg(theta_tau_s), 5)

        Phases1 = ["A-", "A+", "B+", "B-", "C-", "C+"]
        Phases2 = ["A-", "B-", "B+", "C+", "C-", "A-"]

        angle_r = theta_tau_s * 0.5
        delta_theta = theta_tau_s
        for pitch in range(1, 7):
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

        femm.mi_saveas("IPM_new.fem")
        Time = 60 / (f * 360)
        Theta_elec = (theta_tau_s * Time * 180 / np.pi) * 2 * np.pi * f
        r_yoke_stator = r_a - h_t
        r_inner = r_a - h_t - h_ys
        try:
            femm.mi_createmesh()
            femm.smartmesh(0)
            femm.mi_analyze()
            (
                outputs["B_g"],
                outputs["B_rymax"],
                outputs["B_smax"],
                outputs["Sigma_normal"],
                V_rotor,
                V_stator,
            ) = run_post_process(D_a, g, r_outer, h_yr, h_ys, r_inner, theta_p_r)

            outputs["M_Fes"] = V_stator * 2 * rho_Fe * p1 / 10
            outputs["M_Fest"] = outputs["M_Fes"] - np.pi * (r_yoke_stator**2 - r_inner**2) * l_s * rho_Fe
            outputs["M_Fer"] = V_rotor * 2 * rho_Fe * p1 / 10
            outputs["Iron"] = outputs["M_Fes"] + outputs["M_Fer"]
            outputs["T_e"], outputs["Sigma_shear"] = B_r_B_t(
                Theta_elec, D_a, l_s, p1, g, theta_p_r, I_s, theta_tau_s, layer_1, layer_2, N_c, tau_p
            )
            #if outputs["T_e"] >= 20e6:
            #    seed = random.randrange(0, 101, 2)
            #    femm.mi_saveas("IPM_new_" + str(seed) + ".fem")

        except Exception as e:
            #print(
            #    D_a,
            #    l_s,
            #    h_t,
            #    p1,
            #    g,
            #    h_ys,
            #    h_yr,
            #    alpha_v,
            #    N_c,
            #    I_s,
            #    h_m,
            #    d_mag,
            #    d_sep,
            #    m_sep * line2.length,
            #    l_fe_ratio,
            #    ratio,
            #)
            outputs = bad_inputs(outputs)
            raise(e)
