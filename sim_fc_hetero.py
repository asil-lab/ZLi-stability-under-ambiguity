import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from utils import ref2mat
from utils import ang2mat

def afc(ITR, dt, HH, LL, N, D, z,z_target):
    error = np.zeros(ITR)
    for i in range(ITR):
        z = z - dt * HH @ LL @ z
        z[0:6] = np.array([0,0,1,0,1,1]) # leader
        error[i] = np.linalg.norm(z - z_target)
        error[i] = np.linalg.norm(z - np.array([0,0,1,0,1,1,0,1]))
    return error

if __name__ == '__main__':

    D = 2
    N = 4
    L = np.array([[1,-1,1,-1],[-1,1,-1,1],[1,-1,1,-1],[-1,1,-1,1]])
    LL = np.kron(L, np.eye(D))

    ITR = 500
    dt = 0.1
    t = np.arange(0, ITR * dt, dt)
    z_init = np.random.randn(N * D)
    # z_init = np.array([0,0,2,0,1,1,1,2])
    ######### get target config #########
    z = z_init
    for i in range(ITR):
        z = z - dt * LL @ z
    z_target = z
    ###############################

    # unambiguous case
    HH = np.eye(N*D)
    error = afc(ITR, dt, HH, LL, N, D, z_init,z_target)

    Errors = np.zeros((4,ITR))
    # ok case
    H1 = ang2mat(0,D)
    H2 = ang2mat(np.pi/6,D)
    H3 = ang2mat(np.pi/4,D)
    H4 = ang2mat(np.pi/3,D)
    HH = block_diag(H1,H2,H3,H4)
    ##########
    eigvals, eigvecs = np.linalg.eig(LL)
    idx = np.where(eigvals > 1e-6)
    U1 = eigvecs[:, idx[0]]
    PD = 0.5 * (U1.T @ (HH + HH.T) @ U1)
    eigs = np.linalg.eigvals(PD)
    ##########
    eigss = np.linalg.eigvals(HH@LL)
    Errors[0,:]= afc(ITR, dt, HH, LL, N, D, z_init,z_target)

    # slightly out of range
    H1 = ang2mat(0,D)
    H2 = ang2mat(0,D)
    H3 = ang2mat(np.pi/6,D)
    H4 = ang2mat(3*np.pi/2,D)
    HH = block_diag(H1,H2,H3,H4)
    Errors[1,:] = afc(ITR, dt, HH, LL, N, D, z_init,z_target)
    Errors[1, :] = np.ones(ITR)

    # ambiguity matrix
    H1 = ang2mat(0, D)
    H2 = ang2mat(np.pi/6, D)
    H3 = ang2mat(np.pi / 4, D)
    H4 = ang2mat(2*np.pi / 3, D)
    HH = block_diag(H1, H2, H3, H4)
    Errors[2,:] = afc(ITR, dt, HH, LL, N, D, z_init,z_target)
    Errors[2, :] = np.ones(ITR)
    # ambiguity matrix
    H1 = ang2mat(0, D)
    H2 = ang2mat(0, D)
    H3 = ang2mat(0, D)
    H4 = ref2mat(0, D)
    HH = block_diag(H1, H2, H3, H4)
    Errors[3,:] = afc(ITR, dt, HH, LL, N, D, z_init,z_target)
    Errors[3, :] = np.ones(ITR)


    fig, (ax1) = plt.subplots(figsize=(4, 6))
    ax1.plot(t, error, linestyle='--',color='black')
    for i in range(4):
        c = plt.cm.hsv(i / 4)
        ax1.plot(t, Errors[i,:], color=c)
    # plt.tight_layout()
    # ax1.grid(True)
    # ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
    # set x range
    ax1.set_ylim([1e-2, 1e3])
    plt.yscale('log')
    # plt.savefig('../figures/error_avgcon_hetero_log.svg', format='svg')
    plt.show()

    # fig, (ax2) = plt.subplots(figsize=(3, 2))
    # ax2.plot(t, error, linestyle='--', color='black')
    # for i in range(4):
    #     c = plt.cm.hsv(i / 4)
    #     ax2.plot(t, Errors[i, :], color=c)
    # plt.tight_layout()
    # ax2.set_ylim([0, 4])
    # ax2.set_xlim([0, 30])
    # plt.savefig('../figures/error_avgcon_homo.svg', format='svg')
    # plt.show()



