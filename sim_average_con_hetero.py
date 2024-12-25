import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from utils import ref2mat
from utils import ang2mat

def average_consensus(ITR, dt, HH, LL, N, D, z):
    error = np.zeros(ITR)
    z_target = np.mean(z)
    for i in range(ITR):
        z = z - dt * HH @ LL @ z
        z[0:2] = np.zeros(2)  # leader
        error[i] = np.linalg.norm(z - np.zeros(N * D))
        # error[i] = np.linalg.norm(z - z_target)
    return error

if __name__ == '__main__':

    D = 2
    N = 4
    L = np.array([[2,0,-1,-1],[0,1,-1,0],[-1,-1,3,-1],[-1,0,-1,2]])
    LL = np.kron(L, np.eye(D))

    ITR = 500
    dt = 0.1
    t = np.arange(0, ITR * dt, dt)
    z_init = np.random.randn(N * D)

    # unambiguous case
    HH = np.eye(N*D)
    error = average_consensus(ITR, dt, HH, LL, N, D, z_init)

    Errors = np.zeros((4,ITR))
    # ok case
    H1 = ang2mat(0,D)
    H2 = ang2mat(np.pi/6,D)
    H3 = ang2mat(np.pi/4,D)
    H4 = ang2mat(np.pi/3,D)
    HH = block_diag(H1,H2,H3,H4)
    Errors[0,:]= average_consensus(ITR, dt, HH, LL, N, D, z_init)

    # slightly out of range
    H1 = ang2mat(0,D)
    H2 = ang2mat(0,D)
    H3 = ang2mat(np.pi/6,D)
    H4 = ang2mat(np.pi/2,D)
    HH = block_diag(H1,H2,H3,H4)
    Errors[1,:] = average_consensus(ITR, dt, HH, LL, N, D, z_init)

    # ambiguity matrix
    H1 = ang2mat(0, D)
    H2 = ang2mat(np.pi/6, D)
    H3 = ang2mat(np.pi / 4, D)
    H4 = ang2mat(2*np.pi / 3, D)
    HH = block_diag(H1, H2, H3, H4)
    Errors[2,:] = average_consensus(ITR, dt, HH, LL, N, D, z_init)
    # ambiguity matrix
    H1 = ang2mat(0, D)
    H2 = ang2mat(0, D)
    H3 = ang2mat(0, D)
    H4 = ref2mat(0, D)
    HH = block_diag(H1, H2, H3, H4)
    Errors[3,:] = average_consensus(ITR, dt, HH, LL, N, D, z_init)

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



