import struct
import numpy as np
import matplotlib.pyplot as plt

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

# Draw the first magnet using 12 points
magnet1 = np.zeros((12,2))
magnet1[1,:] = np.array([r_so, 0.])
r2 = r_g
magnet1[2,:] = np.array([r2, 0])
r3 = r_g + h_yr
magnet1[3,:] = rotate(magnet1[0,0], magnet1[0,1], r3, 0, 0.5 * (alpha_p - alpha_pr))
r6 = r3 + d_mag
magnet1[6,:] = rotate(magnet1[0,0], magnet1[0,1], r6, 0, 0.5 * alpha_p)
r7 = r6 + slot_height
r_ro = r7+h_yr # Outer rotor radius
magnet1[5,:] = np.array([r_ro,0])
magnet1[7,:] = rotate(magnet1[0,0], magnet1[0,1], r7, 0, 0.5 * alpha_p)
magnet1[4,:] = rotate(magnet1[0,0], magnet1[0,1], r3 + slot_height, 0, 0.5 * (alpha_p - alpha_pr))
p6p3_angle = np.arctan((magnet1[6,1]-magnet1[3,1])/(magnet1[6,0]-magnet1[3,0]))
p6p3p4_angle = p6p3_angle - 0.5 * (alpha_p - alpha_pr)
p4r = rotate(magnet1[3,0], magnet1[3,1], magnet1[4,0], magnet1[4,1], -0.5 * (alpha_p - alpha_pr))
p11r =  (p4r[0] - magnet1[3,0]) * np.cos(p6p3p4_angle) * np.array([np.cos(p6p3p4_angle), np.sin(p6p3p4_angle)]) + magnet1[3,:]
magnet1[11,:] = rotate(magnet1[3,0], magnet1[3,1], p11r[0], p11r[1], 0.5 * (alpha_p - alpha_pr))
mml = (magnet1[6,1] - magnet1[11,1]) / np.sin(p6p3_angle) # max magnet1 length
magnet1[8,:] = magnet1[4,:] + mml * np.array([np.cos(p6p3_angle),np.sin(p6p3_angle)])
ml = mml * magnet_l_pc
magnet1[9,:] = magnet1[4,:] + ml * np.array([np.cos(p6p3_angle),np.sin(p6p3_angle)])
magnet1[10,:] = magnet1[11,:] + ml * np.array([np.cos(p6p3_angle),np.sin(p6p3_angle)])

# Draw the outer rotor
rotor = np.zeros((4,2))
rotor[0,0] = r_g
rotor[1,0] = r_ro
rotor[2,:] = rotate(0., 0., rotor[1,0], rotor[1,1], alpha_s)
rotor[3,:] = rotate(0., 0., rotor[0,0], rotor[0,1], alpha_s)



# Mirror the points for the second magnet
magnet2 = np.zeros_like(magnet1)
for i in range(len(magnet1[:,0])): 
    temp = np.zeros(2)
    temp[0], temp[1] = rotate(magnet1[0,0], magnet1[0,1], magnet1[i,0], magnet1[i,1], -0.5 * alpha_p)
    temp[1] *= -1
    magnet2[i,:] = rotate(magnet1[0,0], magnet1[0,1], temp[0], temp[1], 0.5 * alpha_p)

# Mirror the points for the third magnet
magnet3 = np.zeros_like(magnet1)
for i in range(len(magnet1[:,0])): 
    magnet3[i,:] = rotate(magnet1[0,0], magnet1[0,1], magnet1[i,0], magnet1[i,1], alpha_p)
# Mirror the points for the fourth magnet
magnet4 = np.zeros_like(magnet1)
for i in range(len(magnet1[:,0])): 
    magnet4[i,:] = rotate(magnet1[0,0], magnet1[0,1], magnet2[i,0], magnet2[i,1], alpha_p)

fig, ax = plt.subplots(1,1)
for i in range(len(stator[:,0])):
    ax.plot(stator[i,0], stator[i,1], 's', label='stator ' + str(i))
for i in range(len(rotor[:,0])):
    ax.plot(rotor[i,0], rotor[i,1], 'o', label='rotor ' + str(i))
for i in range(len(coil_slot1[:,0])):
    ax.plot(coil_slot1[i,0], coil_slot1[i,1], 'x', label='coil slot ' + str(i))

for i in range(len(magnet1[:,0])):
    ax.plot(magnet1[i,0], magnet1[i,1], '*', label='magnet ' + str(i))
    
# for i in range(4,len(magnet[:,0])):
#     if i!=5:
#         ax.plot(magnet2[i,0], magnet2[i,1], 'o')

# for i in range(4,len(magnet[:,0])):
#     if i!=5:
#         ax.plot(magnet3[i,0], magnet3[i,1], 'o')

# for i in range(4,len(magnet[:,0])):
#     if i!=5:
#         ax.plot(magnet4[i,0], magnet4[i,1], 'o')

ax.plot(magnet1[[0,1,2,5],0], magnet1[[0,1,2,5],1], 'k-')
ax.plot(magnet1[[3,4,7],0], magnet1[[3,4,7],1], 'k-')
ax.plot(magnet1[[3,6,7],0], magnet1[[3,6,7],1], 'k-')
ax.plot(magnet1[[6,8],0], magnet1[[6,8],1], 'k-')
ax.plot(magnet1[[4,9,10,11],0], magnet1[[4,9,10,11],1], 'k-')
ax.plot(magnet1[[4,11],0], magnet1[[4,11],1], 'k-')

ax.plot(magnet2[[3,4,7],0], magnet2[[3,4,7],1], 'k-')
ax.plot(magnet2[[3,6,7],0], magnet2[[3,6,7],1], 'k-')
ax.plot(magnet2[[6,8],0], magnet2[[6,8],1], 'k-')
ax.plot(magnet2[[4,9,10,11],0], magnet2[[4,9,10,11],1], 'k-')
ax.plot(magnet2[[4,11],0], magnet2[[4,11],1], 'k-')

ax.plot(magnet3[[3,4,7],0], magnet3[[3,4,7],1], 'k-')
ax.plot(magnet3[[3,6,7],0], magnet3[[3,6,7],1], 'k-')
ax.plot(magnet3[[6,8],0], magnet3[[6,8],1], 'k-')
ax.plot(magnet3[[4,9,10,11],0], magnet3[[4,9,10,11],1], 'k-')
ax.plot(magnet3[[4,11],0], magnet3[[4,11],1], 'k-')

ax.plot(magnet4[[3,4,7],0], magnet4[[3,4,7],1], 'k-')
ax.plot(magnet4[[3,6,7],0], magnet4[[3,6,7],1], 'k-')
ax.plot(magnet4[[6,8],0], magnet4[[6,8],1], 'k-')
ax.plot(magnet4[[4,9,10,11],0], magnet4[[4,9,10,11],1], 'k-')
ax.plot(magnet4[[4,11],0], magnet4[[4,11],1], 'k-')
ax.axis('equal')
ax.legend()
# ax.set_xlim([4.5,5.5])
plt.show()
