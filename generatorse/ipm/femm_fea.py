# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:28:2 2021

@author: lsethura
"""

import femm
import numpy as np
import openmdao.api as om
from sympy import Point, Segment, Polygon
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
        # Inputs
        self.add_input("p1", 0.0, desc="Number of pole pairs")
        self.add_input("r_g", 0.0, units="m", desc="Air gap radius")
        self.add_input("D_a", 0.0, units="m", desc="Stator outer diameter")
        self.add_input("d_mag", 0.0, units="m", desc="Magnet distance from inner radius")
        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("f", 0.0, units="Hz", desc="Frequency")
        self.add_input("h_s1", 0.010, desc="Slot Opening height")
        self.add_input("h_s2", 0.010, desc="Wedge Opening height")
        self.add_input("h_yr", 0.0, units="m", desc="Rotor yoke height")
        self.add_input("h_ys", 0.0, units="m", desc="Stator yoke height")
        self.add_input("h_t", 0.0, units="m", desc="Stator tooth height")
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical steel density")
        self.add_input("I_s", 0.0, units="A", desc="Stator current")
        self.add_input("N_c", 0.0, desc="Number of turns per coil in series")
        self.add_input("tau_s", 0.0, units="m", desc="Slot pitch")
        self.add_input('magnet_l_pc', 1.0, desc = "Length of magnet divided by max magnet length")


        # Outputs
        self.add_output("b_s", 0.0, units="m", desc="Slot width")
        self.add_output("h_s", 0.0, units="m", desc="Stator tooth height")
        self.add_output("l_m", 0.0, units="m", desc="Magnet length")
        self.add_output("tau_p", 0.0, units="m", desc="Pole pitch")
        self.add_output("Slot_aspect_ratio", 0.0, desc="Slot aspect ratio")
        self.add_output("B_g", 0.0, units="T", desc="Peak air gap flux density ")
        self.add_output("B_rymax", 0.0, units="T", desc="Peak Rotor yoke flux density")
        self.add_output("B_smax", 0.0, units="T", desc="Peak flux density in the stator")
        self.add_output("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_output("Sigma_shear", 0.0, units="Pa", desc="Shear stress")
        self.add_output("Sigma_normal", 0.0, units="Pa", desc="Normal stress")
        self.add_output("Iron", 0.0, units="kg", desc="Magnet length")
        self.add_output("M_Fes", 0.0, units="kg", desc="Stator iron mass")
        self.add_output("M_Fer", 0.0, units="kg", desc="Rotor iron mass")
        self.add_output("M_Fest", 0.0, units="kg", desc="Stator teeth mass")
        self.add_output("r_outer_active", 0.0, units="m", desc="Rotor outer diameter")
        self.add_output("r_mag_center", 0.0, units="m", desc="Rotor magnet radius")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        # Inputs
        n2P = float(inputs['p1']) # 200 # number pole pairs
        r_g = float(inputs['r_g']) # 5.3 # air gap radius
        D_a = float(inputs['D_a']) # 5.295 * 2. # stator outer diameter
        slot_height = float(inputs['h_s1']) # 0.01
        d_mag = float(inputs['d_mag']) # 0.04 # magnet distance from inner radius
        magnet_l_pc = float(inputs['magnet_l_pc']) # length of magnet divided by max magnet length
        # magnet_h_pc float(= 0.8 # height of magnet divided by max magnet height
        h_yr = float(inputs['h_yr']) # 0.01 # Rotor yoke height
        h_ys = float(inputs['h_ys']) # 0.01 # Stator yoke height
        h_so = float(inputs['h_s1']) # 0.01 # Slot opening height
        h_wo = float(inputs['h_s2']) # 0.01 # Wedge opening height
        h_t =  float(inputs['h_t']) # 0.27 # Stator tooth height
        l_s = float(inputs['l_s']) # 2.918 # stack length
        N_c = float(inputs['N_c']) # 3 # Number of turns per coil in series
        f = float(inputs["f"])
        rho_Fe = float(inputs['rho_Fe'])
        I_s = float(inputs['I_s'])
        tau_s = float(inputs['tau_s'])

        # Preprocess inputs
        alpha_p = np.pi / n2P # pole sector
        alpha_pr = 0.9 * alpha_p # pole sector reduced to 90%
        r_so = D_a / 2. # Outer radius of the stator
        g = r_g - r_so # Air gap length
        r_si = r_so - (h_so + h_wo + h_t + h_ys) # Inner radius of the stator


        def rotate(xo, yo, xp, yp, angle):
            ## Rotate a point counterclockwise by a given angle around a given origin.
            # angle *= -1.
            qx = xo + np.cos(angle) * (xp - xo) - np.sin(angle) * (yp - yo)
            qy = yo + np.sin(angle) * (xp - xo) + np.cos(angle) * (yp - yo)
            return qx, qy

        # Get the sector geometry for five magnet
        alpha_s = alpha_p * 5.
        # Get the yoke sector for 6 armature coils
        alpha_y = alpha_s / 6.

        # Draw the inner stator
        stator = np.zeros((4,2))
        stator[0,0] = r_si
        stator[1,0] = r_so
        stator[2,:] = rotate(0., 0., stator[1,0], stator[1,1], alpha_s)
        stator[3,:] = rotate(0., 0., stator[0,0], stator[0,1], alpha_s)

        # Draw the first of six coil slots located next to six yoke teeth
        coil_slot1 = np.zeros((10,2))
        coil_slot1[0,:] = rotate(0., 0., r_si + h_ys, 0., alpha_y*0.25)
        coil_slot1[1,:] = rotate(0., 0., r_si + h_ys + h_t/2., 0., alpha_y*0.25)
        coil_slot1[2,:] = rotate(0., 0., r_si + h_ys + h_t, 0., alpha_y*0.25)
        coil_slot1[3,:] = rotate(0., 0., r_si + h_ys + h_t + h_wo, 0., alpha_y*0.45)
        coil_slot1[4,:] = rotate(0., 0., r_si + h_ys + h_t + h_wo + h_so, 0., alpha_y*0.45)
        coil_slot1[5,:] = rotate(0., 0., r_si + h_ys + h_t + h_wo + h_so, 0., alpha_y*0.55)
        coil_slot1[6,:] = rotate(0., 0., r_si + h_ys + h_t + h_wo, 0., alpha_y*0.55)
        coil_slot1[7,:] = rotate(0., 0., r_si + h_ys + h_t, 0., alpha_y*0.75)
        coil_slot1[8,:] = rotate(0., 0., r_si + h_ys + h_t/2., 0., alpha_y*0.75)
        coil_slot1[9,:] = rotate(0., 0., r_si + h_ys, 0., alpha_y*0.75)

        # Draw the first magnet using 8 points
        magnet1 = np.zeros((8,2))
        magnet1[0,:] = rotate(0., 0., r_g + h_yr, 0, 0.5 * (alpha_p - alpha_pr))
        magnet1[2,:] = rotate(0., 0., r_g + h_yr + d_mag, 0, 0.5 * alpha_p)
        r7 =  r_g + h_yr + d_mag + slot_height
        r_ro = r7+h_yr # Outer rotor radius
        magnet1[3,:] = rotate(0., 0., r7, 0, 0.5 * alpha_p)
        magnet1[1,:] = rotate(0., 0., r_g + h_yr + slot_height, 0, 0.5 * (alpha_p - alpha_pr))
        # We might need only one angle here
        p2p0_angle = np.arctan((magnet1[2,1]-magnet1[0,1])/(magnet1[2,0]-magnet1[0,0]))
        p2p0p1_angle = p2p0_angle - 0.5 * (alpha_p - alpha_pr)
        p4r = rotate(magnet1[0,0], magnet1[0,1], magnet1[1,0], magnet1[1,1], -0.5 * (alpha_p - alpha_pr))
        p11r =  (p4r[0] - magnet1[0,0]) * np.cos(p2p0p1_angle) * np.array([np.cos(p2p0p1_angle), np.sin(p2p0p1_angle)]) + magnet1[0,:]
        magnet1[7,:] = rotate(magnet1[0,0], magnet1[0,1], p11r[0], p11r[1], 0.5 * (alpha_p - alpha_pr))
        mml = (magnet1[2,1] - magnet1[7,1]) / np.sin(p2p0_angle) # max magnet1 length
        magnet1[4,:] = magnet1[1,:] + mml * np.array([np.cos(p2p0_angle),np.sin(p2p0_angle)])
        ml = mml * magnet_l_pc
        magnet1[5,:] = magnet1[1,:] + ml * np.array([np.cos(p2p0_angle),np.sin(p2p0_angle)])
        magnet1[6,:] = magnet1[7,:] + ml * np.array([np.cos(p2p0_angle),np.sin(p2p0_angle)])

        # Mirror the points for the second magnet
        magnet2 = np.zeros_like(magnet1)
        for i in range(len(magnet1[:,0])): 
            temp = np.zeros(2)
            temp[0], temp[1] = rotate(0., 0., magnet1[i,0], magnet1[i,1], -0.5 * alpha_p)
            temp[1] *= -1
            magnet2[i,:] = rotate(0., 0., temp[0], temp[1], 0.5 * alpha_p)

        # Draw the outer rotor
        rotor = np.zeros((4,2))
        rotor[0,0] = r_g
        rotor[1,0] = r_ro
        rotor[2,:] = rotate(0., 0., rotor[1,0], rotor[1,1], alpha_s)
        rotor[3,:] = rotate(0., 0., rotor[0,0], rotor[0,1], alpha_s)

        # Create femm document
        myopen()
        femm.newdocument(0)
        femm.mi_probdef(0, "meters", "planar", 1.0e-8, l_s, 30, 0)
        Current = 0
        femm.mi_addmaterial("Air")
        femm.mi_getmaterial("M-36 Steel")
        femm.mi_getmaterial("20 SWG")
        femm.mi_getmaterial("N48")
        femm.mi_modifymaterial("N48", 0, "N48SH")
        femm.mi_modifymaterial("N48SH", 5, 0.7142857)
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

        # Draw nodes in femm
        # Stator
        for i in range(len(stator[:,0])):
            femm.mi_addnode(stator[i,0],stator[i,1])
        # Rotor
        for i in range(len(rotor[:,0])):
            femm.mi_addnode(rotor[i,0],rotor[i,1])
        # Coil slot 1
        for i in range(len(coil_slot1[:,0])):
            femm.mi_addnode(coil_slot1[i,0],coil_slot1[i,1])
            femm.mi_selectnode(coil_slot1[i,0],coil_slot1[i,1])
            femm.mi_setgroup(1)
            femm.mi_clearselected()
        # Magnet 1
        for i in range(len(magnet1[:,0])):
            femm.mi_addnode(magnet1[i,0],magnet1[i,1])
            femm.mi_selectnode(magnet1[i,0],magnet1[i,1])
            femm.mi_setgroup(2)
            femm.mi_clearselected()
        # Magnet 2
        for i in range(len(magnet2[:,0])):
            femm.mi_addnode(magnet2[i,0],magnet2[i,1])
            femm.mi_selectnode(magnet2[i,0],magnet2[i,1])
            femm.mi_setgroup(2)
            femm.mi_clearselected()

        # Draw coils
        start_index = np.array([0,1,8,9,1,2,7,2,3,5,6], dtype=int)
        end_index = np.array([1,8,9,0,2,7,8,3,4,6,7], dtype=int)
        for i in range(len(start_index)):
            femm.mi_addsegment(coil_slot1[start_index[i],0],coil_slot1[start_index[i],1],coil_slot1[end_index[i],0],coil_slot1[end_index[i],1])
            femm.mi_selectsegment((coil_slot1[start_index[i],0]+coil_slot1[end_index[i],0])/2,(coil_slot1[start_index[i],1]+coil_slot1[end_index[i],1])/2)
            femm.mi_setgroup(1)
            femm.mi_clearselected()

        # Copy coils five times
        femm.mi_selectgroup(1)
        femm.mi_copyrotate(0, 0, np.rad2deg(alpha_y), 5)
        femm.mi_clearselected()

        # Draw stator
        femm.mi_addsegment(stator[0,0],stator[0,1],stator[1,0],stator[1,1])
        femm.mi_addsegment(stator[2,0],stator[2,1],stator[3,0],stator[3,1])
        femm.mi_addarc(stator[0,0],stator[0,1], stator[3,0],stator[3,1],np.rad2deg(alpha_s),1)

        # Get coordinates of the six slot openings
        slot_o = np.zeros((12,2))
        for i in range(6): 
            slot_o[i*2,:] = rotate(0., 0., coil_slot1[4,0], coil_slot1[4,1], alpha_y*(i))
            slot_o[i*2+1,:] = rotate(0., 0., coil_slot1[5,0], coil_slot1[5,1], alpha_y*(i))

        # Complete drawing of the six yoke teeth
        femm.mi_addarc(stator[1,0],stator[1,1],slot_o[0,0],slot_o[0,1],np.rad2deg(alpha_y*0.45),1)
        for i in range(5):
            femm.mi_addarc(slot_o[i*2+1,0],slot_o[i*2+1,1],slot_o[i*2+2,0],slot_o[i*2+2,1],np.rad2deg(alpha_y*0.45),1)
        femm.mi_addarc(slot_o[-1,0],+slot_o[-1,1],stator[2,0],stator[2,1],np.rad2deg(alpha_y*0.45),1)


        # Draw rotor
        start_index = np.array([0,2], dtype=int)
        end_index = np.array([1,3], dtype=int)
        for i in range(len(start_index)):
            femm.mi_addsegment(rotor[start_index[i],0],rotor[start_index[i],1],rotor[end_index[i],0],rotor[end_index[i],1])
        start_index = np.array([1,0], dtype=int)
        end_index = np.array([2,3], dtype=int)
        for i in range(len(start_index)):
            femm.mi_addarc(rotor[start_index[i],0],rotor[start_index[i],1],rotor[end_index[i],0],rotor[end_index[i],1],np.rad2deg(alpha_s),1)

        # Close sides of air gap to be able to define boundary conditions, see below
        femm.mi_addsegment(stator[1,0],stator[1,1],rotor[0,0],rotor[0,1])
        femm.mi_addsegment(stator[2,0],stator[2,1],rotor[3,0],rotor[3,1])

        # Draw first magnet
        start_index = np.array([0,1,7,1,5,6,5,4,2,2,3], dtype=int)
        end_index = np.array([1,7,0,5,6,7,4,3,6,3,4], dtype=int)
        for i in range(len(start_index)):
            femm.mi_addsegment(magnet1[start_index[i],0],magnet1[start_index[i],1],magnet1[end_index[i],0],magnet1[end_index[i],1])
            femm.mi_selectsegment((magnet1[start_index[i],0]+magnet1[end_index[i],0])/2,(magnet1[start_index[i],1]+magnet1[end_index[i],1])/2)
            femm.mi_setgroup(2)
            femm.mi_clearselected()
        # Add labels first magnet
        # Air
        air_1 = Polygon(
                        Point(magnet1[0,0], magnet1[0,1], evaluate=False),
                        Point(magnet1[1,0], magnet1[1,1], evaluate=False),
                        Point(magnet1[7,0], magnet1[7,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(air_1).evalf().x, centroid(air_1).evalf().y)
        femm.mi_selectlabel(centroid(air_1).evalf().x, centroid(air_1).evalf().y)
        femm.mi_setblockprop("Air", 1, 1, 0, 0, 1, 0)
        femm.mi_setgroup(2)
        femm.mi_clearselected()
        air_2 = Polygon(
                        Point(magnet1[2,0], magnet1[2,1], evaluate=False),
                        Point(magnet1[3,0], magnet1[3,1], evaluate=False),
                        Point(magnet1[4,0], magnet1[4,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(air_2).evalf().x, centroid(air_2).evalf().y)
        femm.mi_selectlabel(centroid(air_2).evalf().x, centroid(air_2).evalf().y)
        femm.mi_setblockprop("Air", 1, 1, 0, 0, 1, 0)
        femm.mi_setgroup(2)
        femm.mi_clearselected()
        # Magnet material
        mm = Polygon(
                        Point(magnet1[7,0], magnet1[7,1], evaluate=False),
                        Point(magnet1[1,0], magnet1[1,1], evaluate=False),
                        Point(magnet1[5,0], magnet1[5,1], evaluate=False),
                        Point(magnet1[6,0], magnet1[6,1], evaluate=False),
                    )
        r_mag_center = (centroid(mm).evalf().x ** 2 + centroid(mm).evalf().y ** 2) ** 0.5
        femm.mi_addblocklabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        femm.mi_selectlabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        mag_dir = np.rad2deg(p2p0_angle)-90.
        femm.mi_setblockprop("N48SH", 1, 1, 0, mag_dir, 1, 0)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        # Draw second magnet
        start_index = np.array([0,1,7,1,5,6,5,4,2,2,3], dtype=int)
        end_index = np.array([1,7,0,5,6,7,4,3,6,3,4], dtype=int)
        for i in range(len(start_index)):
            femm.mi_addsegment(magnet2[start_index[i],0],magnet2[start_index[i],1],magnet2[end_index[i],0],magnet2[end_index[i],1])
            femm.mi_selectsegment((magnet2[start_index[i],0]+magnet2[end_index[i],0])/2,(magnet2[start_index[i],1]+magnet2[end_index[i],1])/2)
            femm.mi_setgroup(2)
            femm.mi_clearselected()
        # Add labels second magnet
        # Air
        air_1 = Polygon(
                        Point(magnet2[0,0], magnet2[0,1], evaluate=False),
                        Point(magnet2[1,0], magnet2[1,1], evaluate=False),
                        Point(magnet2[7,0], magnet2[7,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(air_1).evalf().x, centroid(air_1).evalf().y)
        femm.mi_selectlabel(centroid(air_1).evalf().x, centroid(air_1).evalf().y)
        femm.mi_setblockprop("Air", 1, 1, 0, 0, 1, 0)
        femm.mi_setgroup(2)
        femm.mi_clearselected()
        air_2 = Polygon(
                        Point(magnet2[2,0], magnet2[2,1], evaluate=False),
                        Point(magnet2[3,0], magnet2[3,1], evaluate=False),
                        Point(magnet2[4,0], magnet2[4,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(air_2).evalf().x, centroid(air_2).evalf().y)
        femm.mi_selectlabel(centroid(air_2).evalf().x, centroid(air_2).evalf().y)
        femm.mi_setblockprop("Air", 1, 1, 0, 0, 1, 0)
        femm.mi_setgroup(2)
        femm.mi_clearselected()
        # Magnet material
        mm = Polygon(
                        Point(magnet2[7,0], magnet2[7,1], evaluate=False),
                        Point(magnet2[1,0], magnet2[1,1], evaluate=False),
                        Point(magnet2[5,0], magnet2[5,1], evaluate=False),
                        Point(magnet2[6,0], magnet2[6,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        femm.mi_selectlabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        mag_dir = 90. - np.rad2deg(p2p0_angle)
        femm.mi_setblockprop("N48SH", 1, 1, 0, mag_dir, 1, 0)
        femm.mi_setgroup(2)
        femm.mi_clearselected()

        # Copy magnet-pair four times
        femm.mi_selectgroup(2)
        femm.mi_copyrotate(0, 0, np.rad2deg(alpha_p), 4)
        femm.mi_clearselected()

        # Label rotor yoke
        rotor_yoke = Polygon(
                        Point(rotor[0,0], rotor[0,1], evaluate=False),
                        Point(rotor[1,0], rotor[1,1], evaluate=False),
                        Point(magnet1[0,0], magnet1[0,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(rotor_yoke).evalf().x, centroid(rotor_yoke).evalf().y)
        femm.mi_selectlabel(centroid(rotor_yoke).evalf().x, centroid(rotor_yoke).evalf().y)
        femm.mi_setblockprop("M-36 Steel")
        femm.mi_clearselected()
        # Label stator yoke
        stator_yoke = Polygon(
                        Point(stator[0,0], stator[0,1], evaluate=False),
                        Point(stator[1,0], stator[1,1], evaluate=False),
                        Point(coil_slot1[1,0], coil_slot1[1,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(stator_yoke).evalf().x, centroid(stator_yoke).evalf().y)
        femm.mi_selectlabel(centroid(stator_yoke).evalf().x, centroid(stator_yoke).evalf().y)
        femm.mi_setblockprop("M-36 Steel")
        femm.mi_clearselected()
        # Label air gap
        air_gap = Polygon(
                        Point(stator[1,0], stator[1,1], evaluate=False),
                        Point(rotor[0,0], rotor[0,1], evaluate=False),
                        Point(coil_slot1[4,0], coil_slot1[4,1], evaluate=False),
                    )
        femm.mi_addblocklabel(centroid(air_gap).evalf().x, centroid(air_gap).evalf().y)
        femm.mi_selectlabel(centroid(air_gap).evalf().x, centroid(air_gap).evalf().y)
        femm.mi_setblockprop("Air")
        femm.mi_clearselected()
        # Label 12 coils
        labels_coils = ["A-","A-","A+","B-","B+","B+","B-","C+","C-","C-","C+","A-"]
        coil1l = Polygon(
                        Point(coil_slot1[0,0], coil_slot1[0,1], evaluate=False),
                        Point(coil_slot1[1,0], coil_slot1[1,1], evaluate=False),
                        Point(coil_slot1[8,0], coil_slot1[8,1], evaluate=False),
                        Point(coil_slot1[9,0], coil_slot1[9,1], evaluate=False),
                    )
        coil1r = Polygon(
                        Point(coil_slot1[1,0], coil_slot1[1,1], evaluate=False),
                        Point(coil_slot1[2,0], coil_slot1[2,1], evaluate=False),
                        Point(coil_slot1[7,0], coil_slot1[7,1], evaluate=False),
                        Point(coil_slot1[8,0], coil_slot1[8,1], evaluate=False),
                    )
        xy_labels_coils = np.zeros((12,2))
        xy_labels_coils[0,:] = centroid(coil1l).evalf().x, centroid(coil1l).evalf().y
        xy_labels_coils[1,:] = centroid(coil1r).evalf().x, centroid(coil1r).evalf().y
        for i in range(2,len(xy_labels_coils[:,0]),2):
            xy_labels_coils[i,:] = rotate(0.,0., xy_labels_coils[i-2,0], xy_labels_coils[i-2,1], alpha_y)
            xy_labels_coils[i+1,:] = rotate(0.,0., xy_labels_coils[i-1,0], xy_labels_coils[i-1,1], alpha_y)
        for i in range(len(xy_labels_coils[:,0])):
            femm.mi_addblocklabel(xy_labels_coils[i,0],xy_labels_coils[i,1])
            femm.mi_selectlabel(xy_labels_coils[i,0],xy_labels_coils[i,1])
            femm.mi_setblockprop("20 SWG", 1, 0, labels_coils[i], 0, 15, N_c)
            femm.mi_clearselected()

        # Add boundary conditions
        # Inner radius stator
        femm.mi_selectarcsegment(stator[0,0], stator[0,1]+1.e-3)
        femm.mi_setarcsegmentprop(1, "Dirichlet", 0, 50)
        femm.mi_clearselected()
        # Outer radius rotor
        femm.mi_selectarcsegment(rotor[1,0], rotor[1,1]+1.e-3)
        femm.mi_setarcsegmentprop(1, "Dirichlet", 0, 50)
        femm.mi_clearselected()
        # Sides sector
        femm.mi_selectsegment((stator[0,0]+stator[1,0])/2., (stator[0,1]+stator[1,1])/2.)
        femm.mi_setsegmentprop("apbc1", 0, 1, 0, 13)
        femm.mi_clearselected()
        femm.mi_selectsegment((stator[2,0]+stator[3,0])/2., (stator[2,1]+stator[3,1])/2.)
        femm.mi_setsegmentprop("apbc1", 0, 1, 0, 13)
        femm.mi_clearselected()
        femm.mi_selectsegment((rotor[0,0]+stator[1,0])/2., (rotor[0,1]+stator[1,1])/2.)
        femm.mi_setsegmentprop("apbc2", 0, 1, 0, 13)
        femm.mi_clearselected()
        femm.mi_selectsegment((stator[2,0]+rotor[3,0])/2., (stator[2,1]+rotor[3,1])/2.)
        femm.mi_setsegmentprop("apbc2", 0, 1, 0, 13)
        femm.mi_clearselected()
        femm.mi_selectsegment((rotor[0,0]+rotor[1,0])/2., (rotor[0,1]+rotor[1,1])/2.)
        femm.mi_setsegmentprop("apbc3", 0, 1, 0, 13)
        femm.mi_clearselected()
        femm.mi_selectsegment((rotor[2,0]+rotor[3,0])/2., (rotor[2,1]+rotor[3,1])/2.)
        femm.mi_setsegmentprop("apbc3", 0, 1, 0, 13)
        femm.mi_clearselected()

        femm.mi_saveas("IPM_new.fem")

        # Compute outputs
        Time = 60 / (f * 360)
        Theta_elec = (alpha_y * Time * 180 / np.pi) * 2 * np.pi * f
        r_yoke_stator = r_so - h_t
        r_inner = r_so - h_t - h_ys
        outputs["r_mag_center"] = r_mag_center
        outputs["tau_p"] = tau_p = np.pi * r_g / n2P
        mag = Segment(Point(magnet1[1,0], magnet1[1,1], evaluate=False), Point(magnet1[5,0], magnet1[5,1], evaluate=False))
        outputs["l_m"] = mag.length
        bs_taus = 0.5
        outputs["b_s"] = bs_taus * tau_s
        outputs["h_s"] = h_s = h_t - h_so - h_wo
        outputs["r_outer_active"] = r_ro
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
            ) = run_post_process(D_a, g, r_ro, h_yr, h_ys, r_inner, alpha_pr)

            outputs["M_Fes"] = V_stator * 2 * rho_Fe * n2P / 10
            outputs["M_Fest"] = outputs["M_Fes"] - np.pi * (r_yoke_stator**2 - r_inner**2) * l_s * rho_Fe
            outputs["M_Fer"] = V_rotor * 2 * rho_Fe * n2P / 10
            outputs["Iron"] = outputs["M_Fes"] + outputs["M_Fer"]
            layer_1 = r_si + h_ys + h_t * 0.75
            layer_2 = r_si + h_ys + h_t * 0.25
            outputs["T_e"], outputs["Sigma_shear"] = B_r_B_t(
                Theta_elec, D_a, l_s, n2P, g, alpha_pr, I_s, alpha_y, layer_1, layer_2, N_c, tau_p
            )

        except Exception as e:
            outputs = bad_inputs(outputs)
            raise(e)
