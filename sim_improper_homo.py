import numpy as np
import matplotlib.pyplot as plt

from utils import ref2mat
# global fig params
cross_size = 110
dot_size = 100
marker_size = 100
tick_label_size = 14
c_orange = [x/255 for x in (255,143,25)]
c_green = [x/255 for x in (23, 251, 98)]
c_blue = [x/255 for x in (104, 64, 252)]

# simulation of homogeneous proper rotation
N = 4
L = np.array([[2,0,-1,-1],[0,1,-1,0],[-1,-1,3,-1],[-1,0,-1,2]])
# L = np.array([[3,-1,-1,-1],[-1,3,-1,-1],[-1,-1,3,-1],[-1,-1,-1,3]])

# 2D Case
D = 2
LL = np.kron(L,np.eye(D))
eigvals, eigvecs = np.linalg.eig(LL)
idx = np.where(eigvals > 1e-6)
U1 = eigvecs[:,idx[0]]

angles = np.linspace(-1,1,180)
rst = np.zeros((len(angles),(N-1)*D))
for i in range(len(angles)):
    H = ref2mat(np.pi*angles[i],D)
    HH = np.kron(np.eye(N),H)
    PD = 0.5*(U1.T@(HH+HH.T)@U1)

    rst[i,:] = np.linalg.eigvals(PD)

fig, (ax1) = plt.subplots(figsize=(8, 3))
for i in range((N-1)*D):
    c = plt.cm.hsv(i/((N-1)*D))
    ax1.scatter(angles,rst[:,i],s=3, color=c)
# plt.tight_layout()
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
# set x range
ax1.set_xlim([-1,1])
plt.savefig('../figures/eig_GPD_improp2d.svg', format='svg')
plt.show()

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# show a few examples of eigenvalues
eig_org = np.linalg.eigvals(-np.kron(L, np.eye(D)))
eig_pi4 = np.linalg.eigvals(-np.kron(L,ref2mat(np.pi/6,D)))
eig_pi2 = np.linalg.eigvals(-np.kron(L,ref2mat(np.pi/2,D)))
eig_pi34 = np.linalg.eigvals(-np.kron(L,ref2mat(3*np.pi/4,D)))

fig, (ax1) = plt.subplots(figsize = (4,5))
ax1.scatter(eig_pi2.real, eig_pi2.imag, color=c_green, marker = 's', s=marker_size)
ax1.scatter(eig_pi4.real, eig_pi4.imag, color=c_orange, s=dot_size)
ax1.scatter(eig_pi34.real, eig_pi34.imag, color=c_blue, marker = '^', s=marker_size)
ax1.scatter(eig_org.real, eig_org.imag, marker = 'x', color='black',s= cross_size)
ax1.grid(True)
ax1.axis('equal')
ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
plt.tight_layout()
# plt.savefig('../figures/eig_Lap_improp2d.svg', format='svg')
# plt.show()




