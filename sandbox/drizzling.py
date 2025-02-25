import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from pykoala import cubing
from pykoala.ancillary import pixel_in_circle

from shapely import geometry

# kernel = cubing.DrizzlingKernel(scale= 10 << u.dimensionless_unscaled,
#                                 pixel_scale_arcsec=1 << u.arcsec)


# x_edges = np.arange(-10, 10)
# y_edges = np.arange(-10, 10)

# xx, yy = np.meshgrid(x_edges, y_edges)

# weights = kernel.kernel_2D(x_edges, y_edges)

# # plt.figure()
# # plt.pcolormesh(xx, yy, weights)
# # plt.colorbar()

# cube = np.zeros((100, 50, 50)) * u.erg / u.s
# cube_var = np.zeros((100, 50, 50)) * (u.erg / u.s)**2
# cube_weight = np.zeros((100, 50, 50))

# fib_spectra = np.ones(100) * u.erg / u.s
# fib_variance = np.ones(100) * (u.erg / u.s)**2

# cubing.interpolate_fibre(
#     fib_spectra, fib_variance, cube, cube_var, cube_weight, 25, 25,
#     kernel=kernel)

# print(cube.sum())

# plt.figure()
# plt.imshow(np.log10(cube[0].value), interpolation="none")


geometry.Polygon()

circle_rad = 1.0
circle_pos = (1.9, 1.5)
pixel_size = 2
pixel_pos = (-1, -1)

pixel_area, area_frac = pixel_in_circle(
    pixel_pos, pixel_size, circle_pos, circle_rad)

print(pixel_area, area_frac)

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
patch = plt.Circle(circle_pos, radius=circle_rad)
ax.add_patch(patch)
ax.add_patch(plt.Rectangle(pixel_pos, pixel_size, pixel_size,
                           facecolor="none", edgecolor='r'))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
