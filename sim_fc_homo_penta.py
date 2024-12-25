import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import scipy.io as sio

from utils import ref2mat
from utils import ang2mat


def afc(ITR, dt, HH, LL, z,z_target):
    error = np.zeros(ITR)
    for i in range(ITR):
        z = z - 10*dt * HH @ LL @ z
        z[0:6] = z_target[0:6] # leader
        error[i] = np.linalg.norm(z - z_target)
        # error[i] = np.linalg.norm(z - np.array([0,0,1,0,1,1,0,1]))
    return error

if __name__ == '__main__':

    D = 2
    N = 7

    L = sio.loadmat('stress_penta.mat')['L']
    LL = np.kron(L, np.eye(D))

    ITR = 500
    dt = 0.1
    t = np.arange(0, ITR * dt, dt)
    z_init = np.random.randn(N * D)
    z_target = np.reshape(np.array([[2,1,1,0,0,-1,-1],
                      [0,1,-1,1,-1,1,-1]]),14,order='F')

    # unambiguous case
    HH = np.eye(N * D)
    error = afc(ITR, dt, HH, LL, z_init, z_target)

    Errors = np.zeros((4, ITR))
    # ambiguity matrix
    HH = np.kron(np.eye(N), ang2mat(np.pi / 4, D))
    Errors[0, :] = afc(ITR, dt, HH, LL,  z_init,z_target)

    # ambiguity matrix
    HH = np.kron(np.eye(N), ang2mat(np.pi / 3, D))
    Errors[1, :] = afc(ITR, dt, HH, LL,  z_init,z_target)

    # ambiguity matrix
    HH = np.kron(np.eye(N), ang2mat(np.pi / 2, D))
    Errors[2, :] = afc(ITR, dt, HH, LL,  z_init,z_target)

    # ambiguity matrix
    HH = np.kron(np.eye(N), ref2mat(0, D))
    Errors[3, :] = afc(ITR, dt, HH, LL,  z_init,z_target)

    fig, (ax1) = plt.subplots(figsize=(4, 6))
    ax1.plot(t, error, linestyle='--', color='black')
    for i in range(4):
        c = plt.cm.hsv(i / 4)
        ax1.plot(t, Errors[i, :], color=c)
    # plt.tight_layout()
    # ax1.grid(True)
    # ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
    # set x range
    ax1.set_ylim([1e-6, 1e3])
    plt.yscale('log')
    plt.savefig('../figures/error_fc_homo_log.svg', format='svg')
    plt.show()
