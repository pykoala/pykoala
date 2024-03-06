from gaussian_rss import koala_gauss_rss
from matplotlib import pyplot as plt
from koala.register import registration
from koala.cubing import build_cube

### Create a set of KOALA RSS

rss_offsets = [[0., 0.], [3., -5.], [1., 3.]]
list_of_rss = []

for offset in rss_offsets:
    rss = koala_gauss_rss(gauss_x_offset=offset[0], gauss_y_offset=offset[1],
                                       n_wave=1)
    rss.info['exptime'] = 1.
    rss.info['pos_angle'] = 0.
    list_of_rss.append(rss)

fig = registration.register_centroid(list_of_rss, plot=True, centroider='gauss')

for f in fig:
    f.savefig("test_registration_centroids.png")

cube = build_cube(rss_set=list_of_rss,
                  reference_coords=(0, 0),
                    reference_pa=0, cube_size_arcsec=(30, 30),
                    pixel_size_arcsec=.5,
                    )

white = cube.get_white_image()

plt.figure()
plt.imshow(white)
plt.show()
####################################################

list_of_rss = []

for offset in rss_offsets:
    rss = koala_gauss_rss(gauss_x_offset=offset[0], gauss_y_offset=offset[1],
                                       n_wave=1)
    rss.info['exptime'] = 1.
    rss.info['pos_angle'] = 0.
    list_of_rss.append(rss)

fig = registration.register_crosscorr(list_of_rss, plot=True)

for f in fig:
    f.savefig("test_registration_crosscorr.png")

cube = build_cube(rss_set=list_of_rss,
                  reference_coords=(0, 0),
                    reference_pa=0, cube_size_arcsec=(30, 30),
                    pixel_size_arcsec=.5,
                    )

white = cube.get_white_image()

plt.figure()
plt.imshow(white)
plt.show()