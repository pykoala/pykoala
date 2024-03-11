from pykoala.rss import RSS

import numpy as np

def gaussian_rss(fibre_x_offset, fibre_y_offset, n_wave=1,
                 gauss_x_offset=0, gauss_y_offset=0, gauss_intensity=1.0,
                 gauss_x_var=1.0, gauss_y_var=1.0):
    intensity = np.zeros((fibre_x_offset.size, int(n_wave)))
    wavelength = np.arange(int(n_wave))
    intensity += (
        np.exp(
            - 0.5 * (fibre_x_offset - gauss_x_offset)**2 / gauss_x_var
            - 0.5 * (fibre_y_offset - gauss_y_offset)**2 / gauss_y_var)
    )[:, np.newaxis]
    intensity *= gauss_intensity / intensity.sum()

    return RSS(intensity=intensity, variance=intensity**2, wavelength=wavelength,
               info=dict(fib_ra=fibre_x_offset,
                         fib_dec=fibre_y_offset,
                         cen_ra=np.array(0.0), cen_dec=np.array(0.0)))

def koala_gauss_rss(gauss_x_offset=0.0, gauss_y_offset=0.0, gauss_intensity=1.0, gauss_x_var=1.0, gauss_y_var=1.0, n_wave=1):
    fibre_x_offset, fibre_y_offset = np.loadtxt(
    'koala_fibre_offset', unpack=True)

    rss = gaussian_rss(fibre_x_offset=fibre_x_offset, fibre_y_offset=fibre_y_offset,
                 n_wave=n_wave,
                 gauss_x_offset=gauss_x_offset, gauss_y_offset=gauss_y_offset,
                 gauss_intensity=gauss_intensity, gauss_x_var=gauss_x_var, gauss_y_var=gauss_y_var)
    return rss

if __name__ == "__main__":

    rss = koala_gauss_rss()
    
    from matplotlib import pyplot as plt
    plt.figure()
    plt.scatter(rss.info['fib_ra'],
                rss.info['fib_dec'], c=rss.intensity[:, 0])
    plt.show()
