import os
import requests
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture, aperture_photometry, ApertureStats

from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.cubing import Cube
from pykoala.query import PSQuery
from pykoala.ancillary import update_wcs_coords

def crosscorrelate_im_apertures(ref_aperture_flux, ref_aperture_flux_err,
                                ref_coord, image, wcs,
                                ra_offset_range=[-10, 10],
                                dec_offset_range=[-10, 10],
                                offset_step=0.5, aperture_diameter=1.25,
                                plot=True):
    """Cross-correlate an image with an input set of apertures.
    
    Description
    -----------
    This method performs a spatial cross-correlation between a list of input aperture fluxes
    and a reference image. For example, the aperture fluxes can simply correspond to the
    flux measured within a fibre or a set of aperture measured in a datacube.
    First, a grid of aperture position offsets is generated, and every iteration
    will compute the difference between the two sets. 

    The figure of merit used can be defined as:
        :math: `w=e^{-(A+B)/2}`
        where 
        :math: `A=\langle f_{Ap} - \hat{f_{Ap}} \rangle`
        :math: `B=|1 - \frac{\langle f_{Ap} \rangle}{\langle \hat{f_{Ap}} \rangle}|`
        where :math: `f_{Ap}, \hat{f_{Ap}` correspond to the aperture flux in the
        reference (DC) and ancillary data

    Parameters
    ----------
    - ref_aperture_flux: np.ndarray
        A set of aperture fluxes measured in the target image.
    - ref_aperture_flux_err: np.ndarray
        The associated errors of the aperture fluxes.
    - ref_coords: astropy.coordinates.SkyCoord
        The celestial coordinates of each aperture.
    - image: np.ndarray
        Pixel data associated to the reference image.
    - wcs: astropy.wcs.WCS
        WCS associated to the reference image.
    - ra_offset_range: list or tuple, default=[-10, 10],
        The range of offsets in RA to be explored in arcseconds.
        Defaul is +-10 arcsec.
    - dec_offset_range: list or tuple, default=[-10, 10],
        The range of offsets in DEC to be explored in arcseconds.
        Defaul is +-10 arcsec.
    - offset_step: float, default=0.5
        Offset step size in arcseconds. Default is 0.5.
    - aperture_diameter: float, default=1.25
        Aperture diameter size in arcseconds. Default is 1.25.

    Returns
    -------
    - results
    """
    print("Cross-correlating image to list of apertures")
    print("Input number of apertures: ", len(ref_aperture_flux))
    # Renormalize the reference aperture
    ref_ap_norm = np.nanmean(ref_aperture_flux)
    ref_flux = ref_aperture_flux / ref_ap_norm
    ref_flux_err = ref_aperture_flux_err / ref_ap_norm
    good_aper = np.isfinite(ref_flux)

    # Make a grid of offsets
    ra_offset = np.arange(*ra_offset_range, offset_step)
    dec_offset = np.arange(*dec_offset_range, offset_step)

    # Initialise the results variable
    sampling = np.full((4, dec_offset.size, ra_offset.size), fill_value=np.nan)
    sampling[2:4] = np.meshgrid(dec_offset, ra_offset, indexing='ij')

    for i, (ra_offset_arcsec, dec_offset_arcsec) in enumerate(
        zip(sampling[2].flatten(), sampling[3].flatten())):
        # Create a set of apertures
        new_coords = SkyCoord(
            ref_coord.ra + ra_offset_arcsec * u.arcsec,
            ref_coord.dec + dec_offset_arcsec * u.arcsec)
        new_apertures = SkyCircularAperture(
            new_coords, r=aperture_diameter / 2 * u.arcsec)
        table = ApertureStats(image, new_apertures, wcs=wcs, sum_method='exact')
        flux_in_ap = table.mean * np.sqrt(table.center_aper_area.value)
        flux_in_ap_err = table.sum_err
        flux_in_ap_norm = np.nanmean(flux_in_ap)
        flux_in_ap /= flux_in_ap_norm
        flux_in_ap_err /= flux_in_ap_norm
        # Ensure that none of the apertures contain NaN's/inf
        mask = np.isfinite(flux_in_ap) & good_aper
        n_valid = np.count_nonzero(mask)
        flux_diff = np.nansum((flux_in_ap[mask] - ref_flux[mask])**2) / n_valid
        flax_ratio = ref_ap_norm / flux_in_ap_norm
        idx = np.unravel_index(i, sampling[0].shape)
        sampling[:2, idx[0], idx[1]] = (flux_diff, flax_ratio)

    print("Computing the offset solution")
    weights = np.exp(- (sampling[0] + np.abs(1 - sampling[1])) / 2)
    weights /= np.nansum(weights)
    # Minimum
    min_pos = np.nanargmax(weights)
    min_pos = np.unravel_index(min_pos, sampling[0].shape)

    ra_min = sampling[3][min_pos]
    dec_min = sampling[2][min_pos]
    ra_mean = np.nansum(sampling[3] * weights)
    dec_mean = np.nansum(sampling[2] * weights)
    
    min_coords = SkyCoord(
            ref_coord.ra + ra_min * u.arcsec,
            ref_coord.dec + dec_min * u.arcsec)
    apertures = SkyCircularAperture(
        min_coords, r=aperture_diameter / 2 * u.arcsec)
    table = ApertureStats(image, apertures, wcs=wcs, sum_method='exact')
    min_flux_in_ap = table.mean * np.sqrt(table.center_aper_area.value)

    mean_coords = SkyCoord(
            ref_coord.ra + ra_mean * u.arcsec,
            ref_coord.dec + dec_mean * u.arcsec)
    apertures = SkyCircularAperture(
        mean_coords, r=aperture_diameter / 2 * u.arcsec)
    table = ApertureStats(image, apertures, wcs=wcs, sum_method='exact')
    mean_flux_in_ap = table.mean * np.sqrt(table.center_aper_area.value)

    results = {
        'offset_min': (ra_min, dec_min), 'offset_mean': (ra_mean, dec_mean),
        'ra_offset': ra_offset, 'dec_offset': dec_offset,
        'sampling': sampling, 'weights': weights}
    if plot:
        results['fig'] = make_crosscorr_plot(results)
    return results
    
def make_crosscorr_plot(results):
    """Make a plot showing the aperture flux cross-correlatation results.
    
    Parameters
    ----------
    - results: dict
        Dictionary containing the results returned by `crosscorrelate_im_apertures`.
    """
    ra_mean, dec_mean = results['offset_mean']
    ra_min, dec_min = results['offset_min']

    fig, axs = plt.subplots(ncols=3, nrows=1, constrained_layout=True,
                            figsize=(12, 4),
                            sharex=False, sharey=False)
    # Plot the flux offset
    ax = axs[0]
    mappable = ax.pcolormesh(
        results['ra_offset'], results['dec_offset'],
        results['sampling'][0]**0.5, cmap='Spectral')
    ax.plot(ra_mean, dec_mean, 'k+', ms=10, label='Weighted')
    ax.plot(ra_min, dec_min, 'k^', label='Minimum')
    ax.set_xlabel("RA offset  (arcsec)")
    ax.set_ylabel("DEC offset  (arcsec)")
    plt.colorbar(mappable, ax=ax, label='A',
                 orientation='horizontal', pad=0.2)

    ax = axs[1]
    p95 = np.clip(np.nanpercentile(np.abs(1 - results['sampling'][1]), 95),
                  a_min=1, a_max=5)
    mappable = ax.pcolormesh(results['ra_offset'], results['dec_offset'],
                             np.abs(1 - results['sampling'][1]),
                             cmap='Spectral', vmin=-p95, vmax=p95)
    ax.plot(ra_mean, dec_mean, 'k+', ms=10, label='Weighted')
    ax.plot(ra_min, dec_min, 'k^', label='Minimum')
    ax.set_xlabel("RA offset (arcsec)")
    plt.colorbar(mappable, ax=ax, label='B',
                 orientation='horizontal', pad=0.2)

    ax = axs[2]
    mappable = ax.pcolormesh(
        results['ra_offset'], results['dec_offset'], results['weights'],
        cmap='Spectral')
    ax.plot(ra_mean, dec_mean, 'k+', ms=10, label='Weighted')
    ax.plot(ra_min, dec_min, 'k^', label='Minimum')
    ax.legend()
    ax.set_xlabel("RA offset (arcsec)")
    plt.colorbar(mappable, ax=ax, label='W',
                 orientation='horizontal', pad=0.2)
    
    vax = ax.inset_axes((1.03, 0, .7, 1))
    vax.plot(results['weights'].sum(axis=1), results['dec_offset'], 'k')
    vax.axhline(dec_min, color='b', label='Max. weight')
    vax.axhline(dec_mean, color='r', label='Avg.')
    vax.legend()

    hax = ax.inset_axes((0, 1.03, 1, .7))
    hax.plot(results['ra_offset'], results['weights'].sum(axis=0), 'k')
    hax.axvline(ra_min, color='b')
    hax.axvline(ra_mean, color='r')

    plt.close(fig)
    return fig

class AncillaryDataCorrection(CorrectionBase):
    """Correction using ancillary data.
    
    Description
    -----------

    Attributes
    ----------
    - data_containers: `pykoala.data_container.DataContainer`
        Set of DC that will be corrected with the ancillary information.
    - images: dict
        A dictionary containing a collection of reference images. Each entry of
        the dictionary must use as key the name of some photometric filter that
        PST could recognise. Each filter entry must contain a dictionary with
        the key `intensity` where the pixel values are stored, a key `wcs` storing
        the WCS of the image, and a key `pixel_size` storing the size of the
        pixel in arcseconds.
    - dc_photometry: dict
        A dictionary containing the synthetic photometry extracted from the 
        data containers. See `get_dc_aperture_fluxes` for a complete description
        of its contents.
    Methods
    -------
    """
    name = 'AncillaryData'
    verbose = True

    def __init__(self, data_containers, **kwargs):
        self.data_containers = data_containers
        self.verbose = kwargs.get('verbose', True)

        self.images = kwargs.get("images", dict())
        self.dc_photometry = kwargs.get("dc_photometry", dict())

    def get_effective_sky_footprint(self):
        """Computes the effective footprint the contains all DataContainers.
        
        Description
        -----------
        Computes the effective footprint the contains all DataContainers.

        Return
        ------
        - centre: tuple
            Centre coordinates (ra_mean, dec_mean)
        - width: tuple
            Window width (ra_max - ra_min, dec_max - dec_min)
        """
        data_containers_footprint = []
        for dc in self.data_containers:
            if isinstance(dc, Cube):
                footprint = dc.wcs.celestial.calc_footprint()
            elif isinstance(dc, RSS):
                min_ra, max_ra = dc.info['fib_ra'].min(), dc.info['fib_ra'].max()
                min_dec, max_dec = dc.info['fib_dec'].min(), dc.info['fib_dec'].max()
                footprint = np.array(
                    [[max_ra, max_dec],
                     [max_ra, min_dec],
                     [min_ra, max_dec],
                     [min_ra, min_dec]])
            else:
                self.corr_print(f"Unrecognized data container of type: {dc.__class__}")
                continue
            data_containers_footprint.append(footprint)

        self.corr_print("Object footprint: ", data_containers_footprint)
        # Select a rectangle containing all footprints
        max_ra, max_dec = np.nanmax(data_containers_footprint, axis=(0, 1))
        min_ra, min_dec = np.nanmin(data_containers_footprint, axis=(0, 1))
        ra_cen, dec_cen = (max_ra + min_ra) / 2, (max_dec + min_dec) / 2
        ra_width, dec_width = max_ra - min_ra, max_dec - min_dec
        self.corr_print("Combined footprint Fov: ", ra_width * 60, dec_width * 60)
        return (ra_cen, dec_cen), (ra_width, dec_width)
    
    def query_image(self, survey='PS', filters='r', im_extra_size_arcsec=30,
                    im_output_dir='.'):
        """Perform a query of external images that overlap with the DataContainers.
        
        Description
        -----------
        This method performs a query to the database of some photometric survey (e.g. PS)
        and retrieves a set of images that overlap with the IFS data.

        Parameters
        ----------
        - survey: (str, default="PS")
            Name of the external survey/database to perform the query. At present only PS
            queries are available.
        - filters: (str, default='r)
            String containing the filters to be included in the query (e.g. "ugriz").
        - im_extra_size_arcsec: (float, default=30)
            Additional extension of the images in arcseconds with respect to the net FoV
            of the DataContainers.
        - im_output_dir: (str, default='.')
            Path to a directory where the queried images will be stored. Default is current
            working directory.
        Returns
        -------
        """
        self.corr_print("Querying image to external database")
        # TODO: Include more options
        if survey != 'PS':
            raise NotImplementedError("Currently only PS queries available")

        # Compute the effective footprint of all DC and use that as input for
        # the query
        im_pos, im_fov = self.get_effective_sky_footprint()
        # Convert the size to pixels
        im_size_pix = int(
            (np.max(im_fov) * 3600 + im_extra_size_arcsec
             ) / PSQuery.pixelsize_arcsec)
        
        self.corr_print("Image center sky position (RA, DEC): ", im_pos)
        self.corr_print("Image size (pixels): ", im_size_pix)
        # Perform the query
        tab = PSQuery.getimage(*im_pos, size=im_size_pix, filters=filters)
        if len(tab) == 0:
            return
        # Overide variable
        self.images = {}
        for row in tab:
            print(f"Retrieving cutout: ra={row['ra']}, dec={row['dec']}, filter={row['filter']}")
            sign = np.sign(row['dec'])
            if sign == -1:
                sign_str = 'n'
            else:
                sign_str = ''
            filename = f"ps_query_{row['ra']:.4f}_{sign_str}{np.abs(row['dec']):.4f}_{row['filter']}.fits"
            filename.replace("-", "n")
            output = os.path.join(im_output_dir, filename)
            filename = PSQuery.download_image(row['url'], fname=output)
            if filename is not None:
                intensity, wcs = PSQuery.read_ps_fits(filename)
                self.images[f"PANSTARRS_PS1.{row['filter']}"] = {
                    "intensity": intensity, "wcs":wcs,
                    "pix_size": PSQuery.pixelsize_arcsec}

    def get_dc_aperture_fluxes(self, filter_names, dc_intensity_units=None,
                               aperture_diameter=1.25, sample_every=2):
        """Compute aperture fluxes from the DataContainers
        
        Description
        -----------
        This method computes a set of aperture fluxes from an input data container.
        If the input DC is a Cube, a grid of equally-spaced apertures will be
        computed. If the input DC is a RSS, the fibre positions will be used as
        reference apertures.

        Parameters
        ----------
        - filter_names: list
            A list of filter names to initialise a list of
            `pst.observables.Filter` objects.
        - dc_intensity_units: `astropy.units.Quantity`, default=1e-16 erg/s/AA/cm2
            Intensity units of the DC.
        - aperture_diameter: float
            Diameter size of the circular apertures. In the case of an RSS, this
            will match the size of the fibres.
        - sample_every: int, default=2
            Spatial aperture sampling in units of the aperture radius. If
            `sample_every=2`, the aperture will be defined every two aperture
            diameters in the image.

        Returns
        -------
        - dc_photometry: dict
            A dictionary containing the results of the computation
            - synth_photo: The synthetic photometry of the DC, it can be a list
            of fluxes (if the input DC is a RSS) or an image (if DC is a Cube).
            For more details, see the method `get_synthetic_photometry`.
            - synth_photo_err: Associated error fo `synth_photo`.
            - wcs: WCS of the DC. If the DC is a RSS it will be None.
            - coordinates: Celestial position of the apertures stored as a list
            of `astropy.coordinates.Skycoord`.
            - aperture_mask: A mask for invalid aperture fluxes.
            - figs: A list of QC figures showing the .
        """
        try:
            from pst.observables import Filter
        except:
            raise ImportError("PST package not found")
        if dc_intensity_units is None:
            dc_intensity_units= 1e-16 * u.erg / u.s / u.cm**2 / u.angstrom

        self.dc_photometry = {}
        for photo_filter in filter_names:
            self.dc_photometry[photo_filter] = {
                'synth_photo': [], 'synth_photo_err': [],
                'wcs': [],
                'coordinates': [], 'aperture_flux': [],
                'flux_unit': dc_intensity_units,
                'aperture_flux_err': [],
                'aperture_mask': [],
                'figs': []}
            photometric_filter = Filter(filter_name=photo_filter)
            # Compute the synthetic photometry on each DC
            for dc in self.data_containers:
                synth_photo, synth_photo_err = self.get_synthetic_photometry(
                    photometric_filter, dc, dc_intensity_units)
                self.dc_photometry[photo_filter]['synth_photo'].append(synth_photo)
                self.dc_photometry[photo_filter]['synth_photo_err'].append(synth_photo_err)
                if isinstance(dc, Cube):
                    self.corr_print(
                        "Computing aperture fluxes using Cube synthetic photometry")
                    # Create a grid of apertures equally spaced
                    pix_size_arcsec = np.max(dc.wcs.celestial.wcs.cdelt) * 3600
                    delta_pix = aperture_diameter / pix_size_arcsec * sample_every
                    self.corr_print(
                        f"Creating a grid of circular aperture (rad={aperture_diameter / 2 / pix_size_arcsec:.2f} px) every {delta_pix:.1f} pixels")
                    rows = np.arange(0, synth_photo.shape[0], delta_pix)
                    columns = np.arange(0, synth_photo.shape[1], delta_pix)
                    yy, xx = np.meshgrid(rows, columns)
                    coordinates = dc.wcs.celestial.pixel_to_world(xx.flatten(), yy.flatten())
                    apertures = SkyCircularAperture(
                        coordinates, r=aperture_diameter / 2 * u.arcsec)
                    self.corr_print(f"Total number of apertures: {len(apertures)}")
                    reference_table = ApertureStats(data=synth_photo, error=synth_photo_err,
                    aperture=apertures, wcs=dc.wcs.celestial, sum_method='exact')
                    # Compute the total flux in the aperture using the mean value
                    flux_in_ap = reference_table.mean * np.sqrt(
                        reference_table.center_aper_area.value)
                    # Compute standard error from the std
                    flux_in_ap_err = reference_table.sum_err
                    self.dc_photometry[photo_filter]['wcs'].append(dc.wcs.celestial.deepcopy())
                elif isinstance(dc, RSS):
                    self.corr_print("Using RSS synthetic photometry as apertures")
                    coordinates = SkyCoord(dc.info['fib_ra'], dc.info['fib_dec'])
                    flux_in_ap, flux_in_ap_err = synth_photo, synth_photo_err
                    self.dc_photometry[photo_filter]['wcs'].append(None)

                # Make a QC plot of the apertures
                fig = self.make_plot_apertures(
                    dc, synth_photo, synth_photo_err, coordinates, flux_in_ap,
                    flux_in_ap_err)
                # Store the results
                self.dc_photometry[photo_filter]['figs'].append(fig)
                self.dc_photometry[photo_filter]['coordinates'].append(coordinates)
                self.dc_photometry[photo_filter]['aperture_flux'].append(flux_in_ap)
                self.dc_photometry[photo_filter]['aperture_flux_err'].append(flux_in_ap_err)
                self.dc_photometry[photo_filter]['aperture_mask'].append(
                    np.isfinite(flux_in_ap) & np.isfinite(flux_in_ap_err))
        return self.dc_photometry
    
    def get_synthetic_photometry(self, filter, dc, dc_intensity_units):
        """Compute synthetic photometry from a DataContainer.
        
        Description
        -----------
        This method extracts synthetic photometry using the spectral information
        of a DataContainer.

        Parameters
        ----------
        - filter: `pst.observables.Filter`
            A Filter for computing the synthetic photometry from the spectra.
        - dc: `DataContainer`
            The input `DataContainer`.
        - dc_intensity_units: `astropy.units.Quantity`
            The units of the intensity of the DC.

        Returns
        -------
        - synth_phot: `np.ndarray`
            Array containing the flux estimates expressed in Jy.
        - synth_phot_err:
            Array containing the error associated to the flux estimate.
        """
        # interpolate the filter response curve to the DC wavelength grid.
        filter.interpolate(dc.wavelength)
        if isinstance(dc, Cube):
            spx_intensity = dc.intensity.reshape(
                    dc.intensity.shape[0], dc.intensity.shape[1] * dc.intensity.shape[2])
            spx_var = dc.variance.reshape(spx_intensity.shape)
            synth_photo = np.full(spx_intensity.shape[1], fill_value=np.nan)
            synth_photo_err = np.full(spx_intensity.shape[1], fill_value=np.nan)
            for ith, (intensity, var) in enumerate(zip(spx_intensity.T, spx_var.T)):
                mask = np.isfinite(intensity) & np.isfinite(var)
                if not mask.any():
                    continue
                f_nu, f_nu_err = filter.get_fnu(intensity * dc_intensity_units,
                                                var**0.5 * dc_intensity_units)
                synth_photo[ith] = f_nu.to('Jy').value
                synth_photo_err[ith] = f_nu_err.to('Jy').value
            synth_photo = synth_photo.reshape(dc.intensity.shape[1:])
            synth_photo_err = synth_photo_err.reshape(dc.intensity.shape[1:])
        elif isinstance(dc, RSS):
            synth_photo = np.full(dc.intensity.shape[1], fill_value=np.nan)
            synth_photo_err = np.full(dc.intensity.shape[1], fill_value=np.nan)
            for ith, (intensity, var) in enumerate(zip(dc.intensity.T, dc.variance.T)):
                mask = np.isfinite(intensity) & np.isfinite(var)
                if not mask.any():
                    continue
                f_nu, f_nu_err = filter.get_fnu(intensity * dc_intensity_units,
                                                            var**0.5 * dc_intensity_units)
                synth_photo[ith] = f_nu.value
                synth_photo_err[ith] = f_nu_err.value

        return synth_photo, synth_photo_err

    def make_plot_apertures(self, dc, synth_phot, synth_phot_err, ap_coords,
                            ap_flux, ap_flux_err):
        """Plot the synthetic aperture fluxes measured from a DC.
        
        Description
        -----------
        This method creates a plot showing the synth
        """
        if isinstance(dc, Cube):
            fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True,
                                    subplot_kw={'projection': dc.wcs.celestial},
                                    constrained_layout=True)
            ax = axs[0, 0]
            mappable = ax.imshow(-2.5 * np.log10(synth_phot / 3631), vmin=16, vmax=23,
                                 interpolation='none')
            plt.colorbar(mappable, ax=ax, label='SB (mag/pix)')
            ax = axs[0, 1]
            mappable = ax.imshow(synth_phot / synth_phot_err, vmin=0, vmax=10, cmap='jet',
                                 interpolation='none')
            plt.colorbar(mappable, ax=ax, label='Flux SNR')
            ax = axs[1, 0]            
            mappable = ax.scatter(ap_coords.ra, ap_coords.dec,
            marker='o', transform=ax.get_transform('world'),
            c= -2.5 * np.log10(ap_flux / 3631), vmin=16, vmax=23)
            plt.colorbar(mappable, ax=ax, label='SB (mag/aperture)')
            ax = axs[1, 1]            
            mappable = ax.scatter(ap_coords.ra, ap_coords.dec,
            marker='o', transform=ax.get_transform('world'),
            c= ap_flux /ap_flux_err, vmin=0, vmax=10)
            plt.colorbar(mappable, ax=ax, label='Aper flux SNR')
            for ax in axs.flatten():
                ax.coords.grid(True, color='orange', ls='solid')
                ax.coords[0].set_format_unit('deg')
            plt.close(fig)
        elif isinstance(dc, RSS):
            pass

        return fig
    
    def make_plot_astrometry_offset(self, ref_image, ref_wcs, image, results):
        """Plot the DC and ancillary data including the astrometry correction."""      
        synt_sb = -2.5 * np.log10(ref_image / 3631)
        im_sb = -2.5 * np.log10(image['intensity'] / 3631 / image['pix_size']**2)

        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        ax = fig.add_subplot(121, title='Original', projection=image['wcs'])

        contourf_params = dict(cmap='Spectral', levels=[18, 19, 20, 21, 22, 23],
                               vmin=19, vmax=23, extend='both')
        contour_params = dict(levels=[18, 19, 20, 21, 22, 23], colors='k')

        ax.coords.grid(True, color='orange', ls='solid')
        ax.coords[0].set_format_unit('deg')
        mappable = ax.contourf(im_sb, **contourf_params)
        plt.colorbar(mappable, ax=ax,
                     label=r"$\rm \log_{10}(F_\nu / 3631 Jy / arcsec^2)$")
        ax.contour(synt_sb,
                   transform=ax.get_transform(ref_wcs), **contour_params)
        # Compute the correctec WCS
        correct_wcs = update_wcs_coords(wcs=ref_wcs,
                                    ra_dec_offset=(
                                        results['offset_min'][0] / 3600,
                                        results['offset_min'][1] / 3600))

        ax = fig.add_subplot(122, title='Corrected', projection=image['wcs'])
        ax.coords.grid(True, color='orange', ls='solid')
        ax.coords[0].set_format_unit('deg')
        mappable = ax.contourf(im_sb, **contourf_params)
        plt.colorbar(mappable, ax=ax,
                     label=r"$\rm \log_{10}(F_\nu / 3631 Jy / arcsec^2)$")

        ax.contour(synt_sb,
                   transform=ax.get_transform(correct_wcs), **contour_params)
        plt.close(fig)
        return fig

    def get_astrometry_offset(self):
        """Compute astrometric offsets using apertures.
        
        Description
        -----------
        Compute the astrometric offsets that match the synthetic
        aperture photometry to the external image.
        """
        self.corr_print("Computing astrometric offsets")
        astrometry_results = []
        for ith, dc in enumerate(self.data_containers):
            dc_results = {}
            for filter in self.images.keys():
                mask = self.dc_photometry[filter]['aperture_mask'][ith]
                results = crosscorrelate_im_apertures(
                    self.dc_photometry[filter]['aperture_flux'][ith][mask],
                    self.dc_photometry[filter]['aperture_flux_err'][ith][mask],
                    self.dc_photometry[filter]['coordinates'][ith][mask],
                    self.images[filter]['intensity'],
                    self.images[filter]['wcs'])
                fig = self.make_plot_astrometry_offset(
                    self.dc_photometry[filter]['synth_photo'][ith],
                    self.dc_photometry[filter]['wcs'][ith],
                    self.images[filter], results)
                results['offset_fig'] = fig
                dc_results[filter] = results
            astrometry_results.append(dc_results)
        return astrometry_results

    def apply(self, dc, ra_dec_offset):
        self.corr_print(
            "Applying astrometry offset correction to DC (RA, DEC): ",
            ra_dec_offset)
        dc.update_coordinates(offset=np.array(ra_dec_offset) / 3600)
        self.log_correction(dc, status='applied',
                            ra_offset_arcsec=str(ra_dec_offset[0]),
                            dec_offset_arcsec=str(ra_dec_offset[1]),
                            )
        return dc