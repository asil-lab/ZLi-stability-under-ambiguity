import numpy as np

def ang2mat(angle, dimension=2):
    if dimension == 2:
        # 2D rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
    elif dimension == 3:
        # 3D rotation matrix around the z-axis
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Dimension must be 2 or 3")

    return rotation_matrix

def ref2mat(angle, dimension=2):
    if dimension == 2:
        # 2D rotation matrix
        ref_matrix = np.array([
            [np.cos(angle), np.sin(angle)],
            [np.sin(angle), -np.cos(angle)]
        ])
    elif dimension == 3:
        # 3D rotation matrix around the z-axis
        ref_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, -1]
        ])
    else:
        raise ValueError("Dimension must be 2 or 3")

    return ref_matrix