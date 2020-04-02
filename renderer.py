import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

def vectorfield_2D(data):
    x = np.arange(0, len(data), 1)
    y = np.arange(0, len(data[0]), 1)

    xx, yy = np.meshgrid(x, y, indexing='ij')

    u = data[:, :, 0]
    v = data[:, :, 1]

    u = [uvec / np.linalg.norm(uvec) for uvec in u]
    v = [vvec / np.linalg.norm(vvec) for vvec in v]

    ax.quiver(xx, yy, u, v, color='black')

def trajectory_2D(data):
    data = np.array(data)
    ax.plot(data[:, 0], data[:, 1])

def show():
    plt.show()