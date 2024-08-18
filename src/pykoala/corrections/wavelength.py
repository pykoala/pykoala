import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
from scipy.ndimage import median_filter, gaussian_filter, percentile_filter

from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.ancillary import flux_conserving_interpolation, vac_to_air
# from pykoala.corrections.sky import ContinuumEstimator


class WavelengthOffset(object):
    """Wavelength offset class.

    This class stores a 2D wavelength offset.

    Attributes
    ----------
    offset_data : wavelength offset, in pixels
    offset_error : standard deviation of `offset_data`
    """
    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        self.offset_data = offset_data
        self.offset_error = offset_error

    def tofits(self, output_path=None):
        if output_path is None:
            if self.path is None:
                raise NameError("Provide output path")
            else:
                output_path = self.path
        primary = fits.PrimaryHDU()
        data = fits.ImageHDU(data=self.offset_data, name='OFFSET')
        error = fits.ImageHDU(data=self.offset_error, name='OFFSET_ERR')
        hdul = fits.HDUList([primary, data, error])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        vprint(f"Wavelength offset saved at {output_path}")

    @classmethod
    def from_fits(cls, path):
        """Load the offset data from a fits file.

        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.
        """
        if not os.path.isfile(path):
            raise NameError(f"offset file {path} does not exist.")
        vprint(f"Loading wavelength offset from {path}")
        with fits.open(path) as hdul:
            offset_data = hdul[1].data
            offset_error = hdul[2].data
        return cls(offset_data=offset_data, offset_error=offset_error,
                   path=path)


class WavelengthCorrection(CorrectionBase):
    """Wavelength correction class.

    This class accounts for the relative wavelength offset between fibres.

    Attributes
    ----------
    name : str
        Correction name, to be recorded in the log.
    offset : WavelengthOffset
        2D wavelength offset (n_fibres x n_wavelengths)
    verbose: bool
        False by default.
    """
    name = "WavelengthCorrection"
    offset = None
    verbose = False

    def __init__(self, offset_path=None, offset=None, **correction_kwargs):
        super().__init__(**correction_kwargs)

        self.path = offset_path
        self.offset = offset

    @classmethod
    def from_fits(cls, path):
        return cls(offset=WavelengthOffset.from_fits(path=path),
                   offset_path=path)

    def apply(self, rss):
        """Apply a 2D wavelength offset model to a RSS.

        Parameters
        ----------
        rss : RSS
            Original Row-Stacked-Spectra object to be corrected.

        Returns
        -------
        RSS
            Corrected RSS object.
        """

        assert isinstance(rss, RSS)

        rss_out = copy.deepcopy(rss)
        x = np.arange(rss.wavelength.size)
        for i in range(rss.intensity.shape[0]):
            rss_out.intensity[i] = flux_conserving_interpolation(
                x, x - self.offset.offset_data[i], rss.intensity[i])

        self.record_correction(rss_out, status='applied')
        return rss_out


class SolarCrossCorrOffset(WavelengthCorrection):

    name = "SolarCrossCorrelationOffset"

    def __init__(self, sun_wavelength, sun_intensity, **kwargs):
        super().__init__(offset=WavelengthOffset(), **kwargs)
        self.sun_wavelength = sun_wavelength
        self.sun_intensity = sun_intensity

    @classmethod
    def from_fits(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..',
                     'input_data', 'spectrophotometric_stars',
                     'sun_mod_001.fits')
        with fits.open(path) as hdul:
            sun_wavelength = hdul[1].data['WAVELENGTH']
            sun_wavelength = vac_to_air(sun_wavelength)
            sun_intensity = hdul[1].data['FLUX']
        return cls(sun_wavelength=sun_wavelength,
                   sun_intensity=sun_intensity)

    @classmethod
    def from_text_file(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..',
                     'input_data', 'spectrophotometric_stars',
                     'sun_mod_001.fits')
        sun_wavelength, sun_intensity = np.loadtxt(path, unpack=True,
                                                   usecols=(0, 1))
        sun_wavelength = vac_to_air(sun_wavelength)
        return cls(sun_wavelength=sun_wavelength,
                   sun_intensity=sun_intensity)


    def get_solar_features(self, solar_wavelength, solar_spectra,
                            window_size_aa=20):
        
        delta_pixel = int(window_size_aa
                          / (solar_wavelength[-1] - solar_wavelength[0])
                          * solar_wavelength.size)
        if delta_pixel % 2 == 0:
            delta_pixel += 1
        # solar_continuum = ContinuumEstimator.medfilt_continuum(solar_spectra,
        #                                                        delta_pixel)
        solar_continuum = median_filter(solar_spectra, delta_pixel)

        # Detect absorption features
        median_continuum_ratio = np.nanmedian(solar_spectra / solar_continuum)
        weights = np.abs(solar_spectra / solar_continuum -  median_continuum_ratio)
        return weights

    def compute_grid_of_models(self, pix_shift_array, pix_std_array, pix_array,
                              sun_intensity, weights):
        print("Computing grid of models") #TODO
        models_grid = np.zeros((pix_shift_array.size, pix_std_array.size,
                           sun_intensity.size))
        weights_grid = np.zeros((pix_shift_array.size, pix_std_array.size,
                           sun_intensity.size))
        for i, velshift in enumerate(pix_shift_array):
                for j, gauss_std in enumerate(pix_std_array):
                    new_pixel_array = pix_array + velshift
                    
                    interp_sun_intensity = flux_conserving_interpolation(
                        new_pixel_array, pix_array, sun_intensity)
                    interp_sun_intensity = gaussian_filter(
                        interp_sun_intensity, gauss_std)
                    models_grid[i, j] = interp_sun_intensity

                    interp_sun_weight = flux_conserving_interpolation(
                        new_pixel_array, pix_array, weights)
                    interp_sun_weight = gaussian_filter(
                        interp_sun_weight, gauss_std, truncate=2.0)
                    interp_sun_weight /= np.nansum(interp_sun_weight)
                    weights_grid[i, j] = interp_sun_weight
        return models_grid, weights_grid

    def compute_shift_from_twilight(self, data_container, logspace=True,
                                    sun_window_size_aa=20, keep_features_frac=0.1,
                                    response_window_size_aa=200,
                                    wave_range=None,
                                    pix_shift_array=np.arange(-5, 5, 0.1),
                                    pix_std_array=np.arange(0.1, 3, 0.1),
                                    inspect_fibres=None):
        if logspace:
            new_wavelength = np.geomspace(data_container.wavelength[0],
                                          data_container.wavelength[-1],
                                          data_container.wavelength.size)
            rss_intensity = np.array([flux_conserving_interpolation(
                new_wavelength, data_container.wavelength, fibre
                ) for fibre in data_container.rss_intensity])
        else:
            new_wavelength = data_container.wavelength
            rss_intensity = data_container.rss_intensity
        
        # Interpolate the solar spectrum to the new grid of wavelengths
        sun_intensity = flux_conserving_interpolation(
        new_wavelength, self.sun_wavelength, self.sun_intensity)

        # Make an array of weights to focus on the absorption lines
        if wave_range is None:
            weights = self.get_solar_features(new_wavelength, sun_intensity,
                                            window_size_aa=sun_window_size_aa)
            weights[weights < np.nanpercentile(weights, 1 - keep_features_frac)] = 0
            weights[:100] = 0
            weights[-100:] = 0
        else:
            weights = np.array(
                (new_wavelength >= wave_range[0]) & (new_wavelength <= wave_range[-1]),
                dtype=float)
        
        valid_pixels = weights > 0
        print("Number of valid pixels: ", np.count_nonzero(valid_pixels))

        # Estimate the response curve for each individual fibre
        delta_pixel = int(response_window_size_aa
                        / (new_wavelength[-1] - new_wavelength[0])
                        * new_wavelength.size)
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        response_spectrograph = rss_intensity / sun_intensity[np.newaxis]
        smoothed_r_spectrograph = median_filter(
            response_spectrograph, delta_pixel, axes=1)
        spectrograph_upper_env = percentile_filter(
            smoothed_r_spectrograph, 95, delta_pixel, axes=1)
        # Avoid regions dominated by telluric absorption
        fibre_weights =  1 / (1  + (
                spectrograph_upper_env / smoothed_r_spectrograph
                - np.nanmedian(spectrograph_upper_env / smoothed_r_spectrograph)
                )**2)

        normalized_rss_intensity = rss_intensity / smoothed_r_spectrograph
        # Generate and fit the model
        pix_array = np.arange(new_wavelength.size)

        models_grid, weights_grid = self.compute_grid_of_models(
            pix_shift_array, pix_std_array, pix_array, sun_intensity, weights)

        # loop over one variable to avoir memory errors
        all_chi2 = np.zeros((pix_shift_array.size,
                             pix_std_array.size,
                             rss_intensity.shape[0]))
        print("Fitting models to data")
        for i in range(pix_shift_array.size):
            all_chi2[i] = np.nansum(
                (models_grid[i, :, np.newaxis]
                 - normalized_rss_intensity[np.newaxis, :, :])**2
                * weights_grid[i, :, np.newaxis]
                * fibre_weights[np.newaxis, :, :],
                axis=-1) / np.nansum(
                    weights_grid[i, :, np.newaxis]
                    * fibre_weights[np.newaxis, :, :],
                    axis=-1)
        
        best_fit_idx = np.argmin(all_chi2.reshape((-1, all_chi2.shape[-1])),
                                 axis=0)
        best_vel_idx, best_sigma_idx = np.unravel_index(
                best_fit_idx, all_chi2.shape[:-1])
        best_sigma, best_shift = (pix_std_array[best_sigma_idx],
                                    pix_shift_array[best_vel_idx])

        if inspect_fibres is not None:
            print("Inspecting input fibres")
            self.inspect_fibres(inspect_fibres, pix_shift_array, pix_std_array,
                                best_vel_idx, best_sigma_idx,
                                all_chi2, models_grid, weights_grid,
                                normalized_rss_intensity, new_wavelength)

        self.offset.offset_data = -best_shift
        self.offset.offset_error = np.full_like(best_shift, fill_value=np.nan)

        return best_shift, best_sigma
    
    def inspect_fibres(self, fibres, pix_shift_array, pix_std_array,
                       best_vel_idx, best_sigma_idx, chi2,
                       models_grid, weights_grid,
                       normalized_rss_intensity, wavelength):
        
        best_sigma, best_shift = (pix_std_array[best_sigma_idx],
                                  pix_shift_array[best_vel_idx])
        for fibre in fibres:
            fig, ax = plt.subplots()
            ax.set_title(f"Fibre: {fibre}")
            mappable = ax.pcolormesh(
                pix_std_array, pix_shift_array, chi2[:, :, fibre],
                cmap='gnuplot', norm=LogNorm())
            plt.colorbar(mappable, ax=ax,
                         label=r"$\sum_\lambda w(I - \hat{I}(s, \sigma))^2$")
            ax.plot(best_sigma[fibre], best_shift[fibre], '+w',
                    label=r'Best fit: $\Delta\lambda$='
                    + f'{best_shift[fibre]:.2}, ' + r'$\sigma$=' + f'{best_sigma[fibre]:.2f}')
            ax.set_xlabel(r"$\sigma$ (pix)")
            ax.set_ylabel(r"$\Delta \lambda$ (pix)")
            ax.legend()

            sun_intensity = models_grid[best_vel_idx[fibre],
                                        best_sigma_idx[fibre]]
            weight = weights_grid[best_vel_idx[fibre],
                                        best_sigma_idx[fibre]]
            fig, axs = plt.subplots(nrows=2, figsize=(12, 8),
                                    constrained_layout=True)
            ax = axs[0]
            ax.set_title(f"Fibre: {fibre}")
            ax.plot(wavelength, sun_intensity, label='Model')
            ax.plot(wavelength, normalized_rss_intensity[fibre],
                    label='Fibre', lw=2)
            twax = ax.twinx()
            twax.plot(wavelength, weight, c='fuchsia',
                    zorder=-1, alpha=0.5, label='Weight')
            ax.legend()
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength")
            twax.set_ylabel("Relative weight")

            ax = axs[1]
            max_idx = np.argmax(weight)
            max_weight_range = range(np.max((max_idx - 80, 0)),
                  np.min((max_idx + 80, wavelength.size - 1)))
            ax.plot(wavelength[max_weight_range],
                    sun_intensity[max_weight_range], label='Model')
            ax.plot(wavelength[max_weight_range],
                    normalized_rss_intensity[fibre][max_weight_range],
                    label='Fibre', lw=2)
            
            ax.set_xlim(wavelength[max_weight_range][0],
                        wavelength[max_weight_range][-1])
            twax = ax.twinx()
            twax.plot(wavelength[max_weight_range], weight[max_weight_range],
                      c='fuchsia',
                    zorder=-1, alpha=0.5, label='Weight')
            twax.axhline(0)
            ax.annotate(r'Best fit: $\Delta\lambda$='
                    + f'{best_shift[fibre]:.3}, ' + r'$\sigma$=' + f'{best_sigma[fibre]:.3f}',
                    xy=(0.05, 0.95), xycoords='axes fraction', va='top')
            twax.legend()
            ax.legend()
            

            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength")
            twax.set_ylabel("Relative weight")
            plt.show()
        plt.close()


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
