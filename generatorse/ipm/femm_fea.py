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
from generatorse.common.femm_util import myopen, cleanup_femm_files

mu0 = 4 * np.pi * 1e-7

def bad_inputs(outputs):
    #print("Bad inputs for geometry")
    outputs["B_g"] = 30.0
    outputs["B_rymax"] = 30.0
    outputs["B_smax"] = 30.0
    outputs["Sigma_normal"] = 1e9
    outputs["M_Fes"] = 1e9
    outputs["M_Fer"] = 1e9
    outputs["mass_iron"] = 1e9
    outputs["T_e"] = 1e9
    outputs["Sigma_shear"] = 1e9
    femm.mi_saveas("IPM_new_bad_geomtery.fem")
    return outputs


def run_post_process(D_a, g, r_outer, h_yr, h_ys, r_inner, theta_p_r):

    theta_p_d = np.rad2deg(theta_p_r)
    r_a = 0.5 * D_a
    femm.mi_loadsolution()
    #femm.mo_smooth("off")

    femm.mo_selectblock(
        (r_inner + h_ys * 0.5) * np.cos(theta_p_r * 2.5),
        (r_inner + h_ys * 0.5) * np.sin(theta_p_r * 2.5),
    )
    V_stator = femm.mo_blockintegral(10)
    femm.mo_clearblock()

    # Approximate peak field
    def get_B_max(r1, r2):
        B_max = 0.0
        for k in np.linspace(r1, r2, 10):
            femm.mo_addcontour(k * np.cos(0), k * np.sin(0))
            femm.mo_addcontour(k * np.cos(5*theta_p_r), k * np.sin(5*theta_p_r))
            femm.mo_bendcontour(theta_p_d, 0.25)
            femm.mo_makeplot(1, 1000, "B_mag.csv", 1)
            femm.mo_clearcontour()
            B_mag = np.loadtxt("B_mag.csv")
            B_max = np.maximum(B_max, B_mag.max())
        return B_max

    B_rymax = get_B_max(r_inner, r_a)
    B_symax = get_B_max(r_a+g, r_outer)

    '''
    sy_area = []
    ry_area = []
    numelm2 = femm.mo_numelements()
    for r in range(1, numelm2):
        p3, p4, p5, x1, y1, a1, g1 = femm.mo_getelement(r)
        mag = np.sqrt(x1**2 + y1**2)
        if mag <= r_a:
            a = femm.mo_getb(x1, y1)
            sy_area.append(np.sqrt(a[0] ** 2 + a[1] ** 2))
        elif mag >= (r_a + g):
            b = femm.mo_getb(x1, y1)
            ry_area.append(np.sqrt(b[0] ** 2 + b[1] ** 2))

    B_symax2 = max(sy_area)
    B_rymax2 = max(ry_area)
    print(B_symax, B_symax2)
    print(B_rymax, B_rymax2)
    '''

    femm.mo_addcontour((r_a + g * 0.5) * np.cos(0), (r_a + g * 0.5) * np.sin(0))
    femm.mo_addcontour((r_a + g * 0.5) * np.cos(theta_p_r * 5), (r_a + g * 0.5) * np.sin(theta_p_r * 5))
    femm.mo_bendcontour(theta_p_d * 5, 1)
    femm.mo_makeplot(1, 1000, "gap.csv", 1)
    femm.mo_makeplot(2, 1000, "B_r.csv", 1)
    femm.mo_makeplot(3, 1000, "B_t.csv", 1)
    femm.mo_clearcontour()
    femm.mo_selectblock(
        (r_outer - h_yr * 0.5) * np.cos(theta_p_r * 2.5),
        (r_outer - h_yr * 0.5) * np.sin(theta_p_r * 2.5),
    )
    V_rotor = femm.mo_blockintegral(10)

    B_g_peak = np.loadtxt("gap.csv")[:, 1].max()
    B_r_normal = np.loadtxt("B_r.csv")
    B_t_normal = np.loadtxt("B_t.csv")

    circ = B_r_normal[-1, 0]
    force = np.trapz(B_r_normal[:, 1] ** 2 - B_t_normal[:, 1] ** 2, B_r_normal[:, 0])
    sigma_n = abs(force / (2*mu0)) / circ

    return B_g_peak, B_rymax, B_symax, sigma_n, V_rotor, V_stator


def B_r_B_t(Theta_elec, D_a, l_s, p1, g, theta_p_r, I_s, theta_tau_s, layer_1, layer_2, N_c, tau_p):

    theta_p_d = np.rad2deg(theta_p_r)
    r_a = 0.5 * D_a

    #myopen()
    #femm.opendocument("IPM_new.fem")
    femm.mi_modifycircprop("A+", 1, I_s * np.sin(2 * np.pi / 3))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(0 * np.pi / 3))
    femm.mi_modifycircprop("C+", 1, I_s * np.sin(4 * np.pi / 3))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(2 * np.pi / 3))
    femm.mi_modifycircprop("B-", 1, -I_s * np.sin(0 * np.pi / 3))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(4 * np.pi / 3))

    femm.mi_saveas("IPM_new_I1.fem")
    femm.mi_analyze()
    femm.mi_loadsolution()

    femm.mo_addcontour((r_a + g * 0.5) * np.cos(0), (r_a + g * 0.5) * np.sin(0))
    femm.mo_addcontour((r_a + g * 0.5) * np.cos(theta_p_r * 5), (r_a + g * 0.5) * np.sin(theta_p_r * 5))
    femm.mo_bendcontour(theta_p_d * 5, 1)
    #sigma_t, sigma_t2 = femm.mo_lineintegral(3)
    torque_sec, _ = femm.mo_lineintegral(4)
    torque = torque_sec*2*np.pi/theta_p_r/5
    sigma_t = torque / (2*np.pi * (r_a + g * 0.5)**2 * l_s)
    #femm.mo_makeplot(2, 1000, "B_r_1.csv", 1)
    #femm.mo_makeplot(3, 1000, "B_t_1.csv", 1)
    femm.mo_clearcontour()
    #femm.mo_selectblock((r_a + g * 0.5) * np.cos(theta_p_r * 2.5), (r_a + g * 0.5) * np.sin(theta_p_r * 2.5))
    #temp = femm.mo_blockintegral(22)
    femm.mo_close()
    #B_r_1 = np.loadtxt("B_r_1.csv")
    #B_t_1 = np.loadtxt("B_t_1.csv")

    '''
    #myopen()
    #femm.opendocument("IPM_new_I1.fem")

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
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_addcontour((r_a + g * 0.5) * np.cos(0), (r_a + g * 0.5) * np.sin(0))
    femm.mo_addcontour((r_a + g * 0.5) * np.cos(theta_p_r * 5), (r_a + g * 0.5) * np.sin(theta_p_r * 5))
    femm.mo_bendcontour(theta_p_d * 5, 1)
    femm.mo_makeplot(2, 1000, "B_r_2.csv", 1)
    femm.mo_makeplot(3, 1000, "B_t_2.csv", 1)
    femm.mo_clearcontour()
    femm.mo_close()
    B_r_2 = np.loadtxt("B_r_2.csv")
    B_t_2 = np.loadtxt("B_t_2.csv")
    '''
    #B_r_2 = B_r_1
    #B_t_2 = B_t_1

    #circ = B_r_1[-1, 0]
    #force = np.array(
    #    [np.trapz(B_r_1[:, 1] * B_t_1[:, 1], B_r_1[:, 0]), np.trapz(B_r_2[:, 1] * B_t_2[:, 1], B_r_2[:, 0])]
    #)
    #sigma_t = abs(force / mu0) / circ
    #torque = 0.5 * np.pi * sigma_t * D_a**2 * l_s

    #torque_ripple = (torque[0] - torque[1]) * 100 / torque.mean()
    #print(torque[0], torque[1], torque.mean())
    #return torque.mean(), sigma_t.mean()
    return np.abs(torque), np.abs(sigma_t)


class FEMM_Geometry(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("debug_prints", default=False)

    def setup(self):
        # Inputs
        self.add_input("pp", 0.0, desc="Number of pole pairs")
        self.add_input("g", 0.0, units="m", desc="Air gap length")
        self.add_input("D_a", 0.0, units="m", desc="Stator outer diameter")
        self.add_input("d_mag", 0.0, units="m", desc="Magnet distance from inner radius")
        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("h_m", 0.0, units="m", desc="Magnet height")
        self.add_input("h_s1", 0.010, desc="Slot Opening height")
        self.add_input("h_s2", 0.010, desc="Wedge Opening height")
        self.add_input("h_yr", 0.0, units="m", desc="Rotor yoke height")
        self.add_input("h_ys", 0.0, units="m", desc="Stator yoke height")
        self.add_input("h_t", 0.0, units="m", desc="Stator tooth height")
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical steel density")
        self.add_input("N_c", 0.0, desc="Number of turns per coil in series")
        self.add_input('magnet_l_pc', 1.0, desc = "Length of magnet divided by max magnet length")
        self.add_input("N_nom", 0.0, units="rpm", desc="Rated speed of the generator")
        self.add_input("b_t", 0.0, units="m", desc="tooth width ")
        self.add_input("J_s", 0.0, units="A/(mm*mm)", desc="Stator winding current density")
        self.add_input("b", 0.0, desc="Slot pole combination")
        self.add_input("c", 0.0, desc="Slot pole combination")
        self.add_input("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        self.add_input("resistivity_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        self.add_discrete_input("m", 3, desc=" no of phases")

        # Outputs
        self.add_output("f", 0.0, units="Hz", desc="Generator output frequency")
        self.add_output("r_g", 0.0, units="m", desc="Air gap radius")
        self.add_output("tau_s", 0.0, units="m", desc="Slot pitch")
        self.add_output("b_s", 0.0, units="m", desc="Width of the armature coil slot in the stator")
        self.add_output("h_s", 0.0, units="m", desc="Stator tooth height")
        self.add_output("l_m", 0.0, units="m", desc="Magnet length")
        self.add_output("tau_p", 0.0, units="m", desc="Pole pitch")
        self.add_output("alpha_v", 0.0, units="deg", desc="V-angle between the magnets")
        # self.add_output("Slot_aspect_ratio", 0.0, desc="Slot aspect ratio")
        self.add_output("B_g", 0.0, units="T", desc="Peak air gap flux density ")
        self.add_output("B_rymax", 0.0, units="T", desc="Peak rotor yoke flux density")
        self.add_output("B_smax", 0.0, units="T", desc="Peak flux density in the stator")
        self.add_output("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_output("Sigma_shear", 0.0, units="Pa", desc="Shear stress")
        self.add_output("Sigma_normal", 0.0, units="Pa", desc="Normal stress")
        self.add_output("mass_iron", 0.0, units="kg", desc="Magnet length")
        self.add_output("M_Fes", 0.0, units="kg", desc="Stator iron mass")
        self.add_output("M_Fer", 0.0, units="kg", desc="Rotor iron mass")
        self.add_output("M_Fest", 0.0, units="kg", desc="Stator teeth mass")
        self.add_output("r_outer_active", 0.0, units="m", desc="Rotor outer diameter")
        self.add_output("r_mag_center", 0.0, units="m", desc="Rotor magnet radius")
        self.add_output("I_s", 0.0, units="A", desc="Generator input phase current")
        self.add_output("k_w", 0.0, desc="winding factor ")
        self.add_output("N_s", 0.0, desc="Number of turns per coil")
        self.add_output("A_Cuscalc", 0.0, units="m**2", desc="Conductor cross-section")
        self.add_output("R_s", 0.0, units="ohm", desc="Stator resistance")
        self.add_output("K_rad", desc="Aspect ratio")
        self.add_output("P_Cu", 0, units="W", desc="Copper losses")
        self.add_output("S", 0.0, desc="Stator slots")
        self.add_output("mass_copper", 0.0, units="kg", desc="Copper Mass")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Inputs
        pp  = float(inputs['pp']) # number pole pairs
        g = float(inputs['g']) # air gap length
        D_a = float(inputs['D_a']) # stator outer diameter
        h_m = float(inputs['h_m']) # height
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
        rho_Fe = float(inputs['rho_Fe'])
        N_nom = float( inputs["N_nom"] )
        b_t = float( inputs["b_t"] )
        J_s = float( inputs["J_s"] )
        b = float( inputs["b"] )
        c = float( inputs["c"] )
        rho_Copper = float( inputs["rho_Copper"] )
        resistivity_Cu = float( inputs["resistivity_Cu"] )
        m = int( discrete_inputs["m"] )

        if self.options['debug_prints']:
            print('Inputs')
            print('pp: ', pp)
            print('g: ', g)
            print('D_a: ', D_a)
            print('h_m: ', h_m)
            print('d_mag: ', d_mag)
            print('magnet_l_pc: ', magnet_l_pc)
            print('h_yr: ', h_yr)
            print('h_ys: ', h_ys)
            print('h_so: ', h_so)
            print('h_wo: ', h_wo)
            print('h_t: ', h_t)
            print('l_s: ', l_s)
            print('N_c: ', N_c)
            print('N_nom: ', N_nom)
            print('------')

        # Preprocess inputs
        alpha_p = np.pi / pp  # pole sector
        alpha_pr = 0.9 * alpha_p # pole sector reduced to 90%
        r_so = 0.5 * D_a # Outer radius of the stator
        r_g = g + r_so # Air gap length
        r_si = r_so - (h_so + h_wo + h_t + h_ys) # Inner radius of the stator
        # angular frequency in radians
        f = 2 * pp * N_nom / 120  # outout frequency

        # Computations
        outputs["K_rad"] = l_s / (2 * r_g)  # Aspect ratio
        Slot_pole = b / c
        outputs["S"] = Slot_pole * 2 * pp * m
        outputs["k_w"] = 0.933
        outputs["N_s"] = N_s = N_c*(pp*b/c)*2  # Stator turns per phase
        # outputs["N_s"] = N_s = S * 2.0 / 3 * N_c  # Stator turns per phase

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
        h_ri2m = g # distance between inner rotor radius (air gap) and magnets
        magnet1 = np.zeros((8,2))
        magnet1[0,:] = rotate(0., 0., r_g + h_ri2m, 0, 0.5 * (alpha_p - alpha_pr))
        magnet1[2,:] = rotate(0., 0., r_g + h_ri2m + d_mag, 0, 0.5 * alpha_p)
        r7 =  r_g + h_ri2m + d_mag + h_m
        r_ro = r7+h_yr # Outer rotor radius
        magnet1[3,:] = rotate(0., 0., r7, 0, 0.5 * alpha_p)
        magnet1[1,:] = rotate(0., 0., r_g + h_ri2m + h_m, 0, 0.5 * (alpha_p - alpha_pr))
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
        # import matplotlib.pyplot as plt
        # for i in range(len(magnet1[:,0])):
        #     plt.plot(magnet1[i,0],magnet1[i,1], '*', label=str(i))
        # plt.legend()
        # plt.show()
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
        magnet1_centroid_xy = np.array([centroid(mm).evalf().x, centroid(mm).evalf().y])
        r_mag_center = (centroid(mm).evalf().x ** 2 + centroid(mm).evalf().y ** 2) ** 0.5
        femm.mi_addblocklabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        femm.mi_selectlabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        magnet1_dir = np.rad2deg(p2p0_angle)-90.
        femm.mi_setblockprop("N48SH", 1, 1, 0, magnet1_dir, 1, 0)
        femm.mi_setgroup(3)
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
        magnet2_centroid_xy = np.array([centroid(mm).evalf().x, centroid(mm).evalf().y])
        femm.mi_addblocklabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        femm.mi_selectlabel(centroid(mm).evalf().x, centroid(mm).evalf().y)
        magnet2_dir = 90. - np.rad2deg(p2p0_angle)
        femm.mi_setblockprop("N48SH", 1, 1, 0, magnet2_dir, 1, 0)
        femm.mi_setgroup(3)
        femm.mi_clearselected()

        # Copy magnet-pair four times
        femm.mi_selectgroup(2)
        femm.mi_copyrotate(0, 0, np.rad2deg(alpha_p), 4)
        femm.mi_clearselected()
        # Handle alternating magnet orientation
        # Mirror the ones pointing out
        femm.mi_selectgroup(3)
        femm.mi_copyrotate(0, 0, np.rad2deg(alpha_p)*2., 2)
        femm.mi_clearselected()
        # Assign the first pair pointing in
        magnet3_centroid_xy = rotate(0., 0., magnet1_centroid_xy[0], magnet1_centroid_xy[1], alpha_p)
        magnet4_centroid_xy = rotate(0., 0., magnet2_centroid_xy[0], magnet2_centroid_xy[1], alpha_p)
        femm.mi_addblocklabel(magnet3_centroid_xy[0], magnet3_centroid_xy[1])
        femm.mi_selectlabel(magnet3_centroid_xy[0], magnet3_centroid_xy[1])
        magnet3_dir = magnet1_dir + np.rad2deg(alpha_p) + 180.
        femm.mi_setblockprop("N48SH", 1, 1, 0, magnet3_dir, 1, 0)
        femm.mi_setgroup(4)
        femm.mi_clearselected()
        femm.mi_addblocklabel(magnet4_centroid_xy[0], magnet4_centroid_xy[1])
        femm.mi_selectlabel(magnet4_centroid_xy[0], magnet4_centroid_xy[1])
        magnet4_dir = magnet2_dir + np.rad2deg(alpha_p) + 180.
        femm.mi_setblockprop("N48SH", 1, 1, 0, magnet4_dir, 1, 0)
        femm.mi_setgroup(4)
        femm.mi_clearselected()
        # Mirror the ones pointing in
        femm.mi_selectgroup(4)
        femm.mi_copyrotate(0, 0, np.rad2deg(alpha_p)*2., 1)
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
        outputs["r_mag_center"] = r_mag_center
        outputs["tau_p"] = tau_p = np.pi * r_g / pp
        outputs["alpha_v"] = np.rad2deg(p2p0_angle) * 2.
        mag = Segment(Point(magnet1[1,0], magnet1[1,1], evaluate=False), Point(magnet1[5,0], magnet1[5,1], evaluate=False))
        outputs["l_m"] = mag.length
        outputs["tau_s"] = tau_s = alpha_pr * D_a / 2.
        b_s = coil_slot1[-1,1] - coil_slot1[0,1]
        outputs["b_s"] = b_s
        outputs["h_s"] = h_s = h_t - h_so - h_wo
        outputs["r_outer_active"] = r_ro
        outputs["r_g"] = r_g
        outputs["f"] = f

        # Calculating stator resistance
        l_Cus = 2 * (l_s + np.pi / 4 * (tau_s + b_t))  # length of a turn
        L_Cus = N_s * l_Cus
        A_slot = 0.5*h_s*b_s
        outputs["A_Cuscalc"] = A_Cus = A_slot * 0.65 / N_c # factor of 0.5 for 2 layers, 0.65 is fill density
        outputs["I_s"] = I_s = 1e6 * J_s * A_Cus # 1e6 to convert m^2 to mm^2
        outputs["R_s"] = R_s = resistivity_Cu* (1 + 20 * 0.00393)* L_Cus / A_Cus

        # Calculating Electromagnetically active mass
        V_Cus = 2 * m * L_Cus * A_Cus  # copper volume, factor of 2 for 2 layers
        outputs["mass_copper"] = V_Cus * rho_Copper
        # Calculating Losses
        K_R = 1.0  # Skin effect correction coefficient
        outputs["P_Cu"] = 0.5 * m * I_s ** 2 * R_s * K_R

        if self.options['debug_prints']:
            print('Outputs')
            print('r_ro:', r_ro)
        try:
            femm.mi_analyze()
            (
                outputs["B_g"],
                outputs["B_rymax"],
                outputs["B_smax"],
                outputs["Sigma_normal"],
                V_rotor,
                V_stator,
            ) = run_post_process(D_a, g, r_ro, h_yr, h_ys, r_si, alpha_pr)

            outputs["M_Fes"] = V_stator * rho_Fe * 2 * np.pi / (5 * alpha_pr)
            outputs["M_Fest"] = outputs["M_Fes"] - np.pi * ((r_si+h_ys)**2 - r_si**2) * l_s * rho_Fe
            outputs["M_Fer"] = V_rotor * rho_Fe * 2 * np.pi / (5 * alpha_pr)
            outputs["mass_iron"] = outputs["M_Fes"] + outputs["M_Fer"]
            layer_1 = r_si + h_ys + h_t * 0.75
            layer_2 = r_si + h_ys + h_t * 0.25
            outputs["T_e"], outputs["Sigma_shear"] = B_r_B_t(
                Theta_elec, D_a, l_s, pp , g, alpha_pr, I_s, alpha_y, layer_1, layer_2, N_c, tau_p
            )

        except Exception as e:
            outputs = bad_inputs(outputs)
            #raise(e)

        femm.closefemm()
