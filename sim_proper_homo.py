import numpy as np
import matplotlib.pyplot as plt

from utils import ang2mat

# global fig params
fig_size = (4,5)
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
# sweep angles and plot the eigenvalues
D = 2
LL = np.kron(L,np.eye(D))
eigvals, eigvecs = np.linalg.eig(LL)
idx = np.where(eigvals > 1e-6)
U1 = eigvecs[:,idx[0]]

angles = np.linspace(-1,1,100)
rst = np.zeros((len(angles),(N-1)*D))
for i in range(len(angles)):
    H = ang2mat(np.pi*angles[i],D)
    HH = np.kron(np.eye(N),H)
    PD = 0.5*(U1.T@(HH+HH.T)@U1)

    rst[i,:] = np.linalg.eigvals(PD)

fig, (ax2) = plt.subplots(figsize=(8, 3))
# use color of spectral colormap
for i in range((N-1)*D):
    c = plt.cm.hsv(i/((N-1)*D))
    ax2.scatter(angles,rst[:,i],s=3, color=c)
# plt.tight_layout()
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=tick_label_size)
# set x range
ax2.set_xlim([-1,1])
plt.savefig('../figures/eig_GPD_prop2d.svg', format='svg')
plt.show()

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# show a few examples of eigenvalues
eig_org = np.linalg.eigvals(-np.kron(L, np.eye(D)))
eig_pi4 = np.linalg.eigvals(-np.kron(L,ang2mat(np.pi/6,D)))
eig_pi2 = np.linalg.eigvals(-np.kron(L,ang2mat(np.pi/2,D)))
eig_pi32 = np.linalg.eigvals(-np.kron(L,ang2mat(3*np.pi/4,D)))

fig, (ax1) = plt.subplots(figsize=fig_size)
ax1.scatter(eig_pi4.real, eig_pi4.imag, color=c_orange, s=dot_size)
ax1.scatter(eig_pi2.real, eig_pi2.imag, color=c_green, marker = 's', s=marker_size)
ax1.scatter(eig_pi32.real, eig_pi32.imag, color=c_blue, marker = '^', s=marker_size)
ax1.scatter(eig_org.real, eig_org.imag, marker = 'x', color='black',s= cross_size)
ax1.grid(True)
ax1.axis('equal')
ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
plt.tight_layout()
plt.savefig('../figures/eig_Lap_prop2d.svg', format='svg')
plt.show()







