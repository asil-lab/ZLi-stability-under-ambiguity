import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils import ang2mat

fig_size = (3,4)
cross_size = 110
dot_size = 100
marker_size = 100
tick_label_size = 14
c_orange = [x/255 for x in (255,216,150)]
c_green = [x/255 for x in (202, 234, 158)]
c_blue = [x/255 for x in (187, 213, 232)]

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


angles3 = np.linspace(-1,1,300)
angles4 = np.linspace(-1,1,300)
rst = np.zeros((len(angles3),len(angles3)))
H1 = ang2mat(0,D)
H2 = ang2mat(0,D)
for i in range(len(angles3)):
    for j in range(len(angles4)):
        H3 = ang2mat(np.pi*angles3[i],D)
        H4 = ang2mat(np.pi*angles4[j],D)
        HH = scipy.linalg.block_diag(H1,H2,H3,H4)
        PD = 0.5*(U1.T@(HH+HH.T)@U1)
        rst[i,j] = np.min(np.linalg.eigvals(PD))

# plots
zero_indices = np.where(np.abs(rst) < 1e-2)
plt.figure(figsize=(6, 4))
plt.imshow(rst,extent=[-1,1,-1,1],cmap='cool')
plt.colorbar()
# highlight zero values
plt.scatter(angles3[zero_indices[0]], angles4[zero_indices[1]], color='white', marker='.',s=1)
plt.tight_layout()
# plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
plt.savefig('../figures/eig_GPD_hetero.svg', format='svg')
plt.show()

# # 3D Case
# D = 3
# LL = np.kron(L,np.eye(D))
# eigvals, eigvecs = np.linalg.eig(LL)
# idx = np.where(eigvals > 1e-6)
# U1 = eigvecs[:,idx[0]]
#
# angles = np.linspace(-np.pi,np.pi,180)
# rst = np.zeros((len(angles),(N-1)*D))
# for i in range(len(angles)):
#     H = ang2mat(angles[i],D)
#     HH = np.kron(np.eye(N),H)
#     PD = 0.5*(U1.T@(HH+HH.T)@U1)
#
#     rst[i,:] = np.linalg.eigvals(PD)
#
# angles3 = np.linspace(-1,1,300)
# angles4 = np.linspace(-1,1,300)
# rst = np.zeros((len(angles3),len(angles3)))
# H1 = ang2mat(0,D)
# H2 = ang2mat(0,D)
# for i in range(len(angles3)):
#     for j in range(len(angles4)):
#         H3 = ang2mat(np.pi*angles3[i],D)
#         H4 = ang2mat(np.pi*angles4[j],D)
#         HH = scipy.linalg.block_diag(H1,H2,H3,H4)
#         PD = 0.5*(U1.T@(HH+HH.T)@U1)
#         rst[i,j] = np.min(np.linalg.eigvals(PD))
#
# # plots
# zero_indices = np.where(np.abs(rst) < 1e-2)
# plt.figure(figsize=(8, 5))
# plt.imshow(rst,extent=[-1,1,-1,1],cmap='cool')
# plt.colorbar()
# # highlight zero values
# plt.scatter(angles3[zero_indices[0]], angles4[zero_indices[1]], color='white', marker='.',s=1)
# plt.tight_layout()
# # plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
# # plt.savefig('../figures/eig_GPD_improp2d.svg', format='svg')
# plt.show()








