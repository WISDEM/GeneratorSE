import numpy as np
import matplotlib.pyplot as plt
import femm
import os, platform
from sympy import Point, Polygon
from sympy.geometry.util import centroid

def myopen():
    if platform.system().lower() == 'windows':
        femm.openfemm(1)
    else:
        femm.openfemm(winepath=os.environ["WINEPATH"], femmpath=os.environ["FEMMPATH"])
    femm.smartmesh(0)


flag_matplotlib = False
# Inputs
n2P = 200 # number pole pairs
r_g = 5.3 # air gap radius
D_a = 5.295 * 2. # stator outer diameter
slot_height = 0.01
d_mag = 0.04 # magnet distance from inner radius
magnet_l_pc = 0.7 # length of magnet divided by max magnet length
# magnet_h_pc = 0.8 # height of magnet divided by max magnet height
h_yr = 0.01 # Rotor yoke height
h_ys = 0.01 # Stator yoke height
h_so = 0.01 # Slot opening height
h_wo = 0.01 # Wedge opening height
h_t = 0.27 # Stator tooth height
l_s = 2.918 # stack length
N_c = 3 # Number of turns per coil in series


# Preprocess inputs
nP = n2P / 2. # number pole pairs
alpha_p = np.pi / nP # pole sector
alpha_pr = 0.9 * alpha_p # pole sector reduced to 90%
r_so = D_a / 2. # Outer radius of the stator
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

# femm.mi_addsegment(coil_slot1[1,0],coil_slot1[1,1],coil_slot1[8,0],coil_slot1[8,1])
# femm.mi_addsegment(coil_slot1[8,0],coil_slot1[8,1],coil_slot1[9,0],coil_slot1[9,1])
# femm.mi_addsegment(coil_slot1[9,0],coil_slot1[9,1],coil_slot1[0,0],coil_slot1[0,1])
# femm.mi_addblocklabel((coil_slot1[0,0]+coil_slot1[1,0])/2, (coil_slot1[0,1]+coil_slot1[9,1])/2)
# femm.mi_selectlabel((coil_slot1[0,0]+coil_slot1[1,0])/2, (coil_slot1[0,1]+coil_slot1[9,1])/2)
# femm.mi_setblockprop("20 SWG", 1, 0, "A+", 0, 15, N_c)
# femm.mi_clearselected()

# femm.mi_addsegment(coil_slot1[1,0],coil_slot1[1,1],coil_slot1[2,0],coil_slot1[2,1])
# femm.mi_addsegment(coil_slot1[2,0],coil_slot1[2,1],coil_slot1[7,0],coil_slot1[7,1])
# femm.mi_addsegment(coil_slot1[7,0],coil_slot1[7,1],coil_slot1[8,0],coil_slot1[8,1])
# femm.mi_addblocklabel((coil_slot1[1,0]+coil_slot1[2,0])/2, (coil_slot1[1,1]+coil_slot1[7,1])/2)
# femm.mi_selectlabel((coil_slot1[1,0]+coil_slot1[2,0])/2, (coil_slot1[1,1]+coil_slot1[7,1])/2)
# femm.mi_setblockprop("20 SWG", 1, 0, "B-", 0, 15, N_c)
# femm.mi_clearselected()

femm.mi_saveas("IPM_param.fem")




if flag_matplotlib:
    # matplotlib_plot
    fig, ax = plt.subplots(1,1)
    for i in range(len(stator[:,0])):
        ax.plot(stator[i,0], stator[i,1], 's', label='stator ' + str(i))
    for i in range(len(rotor[:,0])):
        ax.plot(rotor[i,0], rotor[i,1], 'o', label='rotor ' + str(i))
    for i in range(len(coil_slot1[:,0])):
        ax.plot(coil_slot1[i,0], coil_slot1[i,1], 'x', label='coil slot ' + str(i))
    for i in range(len(magnet1[:,0])):
        ax.plot(magnet1[i,0], magnet1[i,1], '*', label='magnet ' + str(i))
    ax.axis('equal')
    ax.legend()
    # ax.set_xlim([4.5,5.5])
    plt.show()
