import numpy as np
from matplotlib import pyplot as plt

from pykoala import cubing

x_edges = np.arange(-10, 11)
y_edges = np.arange(-10, 11)

xx, yy = np.meshgrid(x_edges, y_edges)
scale = 3
n_oversampling = 500
weights = np.zeros_like(xx, dtype=float)

for ith, (x, y) in enumerate(zip(xx.flatten(), yy.flatten())):
    oversampling_x, oversampling_y = np.meshgrid(
        np.arange(x - 0.5, x + 0.5, scale / n_oversampling),
        np.arange(y - 0.5, y + 0.5, scale / n_oversampling))

    rr2 = (oversampling_x**2 + oversampling_y**2) / scale**2
    rr2 = rr2.clip(0, 1)
    parab = 2 / np.pi * (1 - rr2)
    weight = np.sum(parab * (scale / 10)**2) / (np.pi * scale**2)
    weights[np.unravel_index(ith, xx.shape)] = weight

print(weights.sum())

kernel = cubing.ParabolicKernel(scale=1.0, truncation_radius=5.0)

kernel = cubing.GaussianKernel(scale=1.0, truncation_radius=5.0)
plt.figure(figsize=(4, 4))
plt.pcolormesh(xx, yy, kernel.kernel_2D(x, y), cmap="nipy_spectral")
plt.colorbar()

plt.figure(figsize=(4, 4))
plt.pcolormesh(xx, yy, parab, cmap="nipy_spectral")
plt.colorbar()


