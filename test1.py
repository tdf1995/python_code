import numpy as np
ndim, dt = 4, 1
motion_mat = np.eye(ndim, 2 * ndim)

# for i in range(ndim):
#     motion_mat[i, ndim + i] = dt
print(motion_mat)