"""
This module contains tools for measuring synthetic photometry from DataContainers
as well as tools for retrieveing and manipulating external imaging data.
""" 
import os
import requests
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy import constants
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture, ApertureStats


from pykoala import vprint
from pykoala.data_container import Cube, RSS
from pykoala.ancillary import update_wcs_coords

class QueryMixin:

    def download_image(url, fname):
        """Download an image and write it to a binary file."""
        vprint(f"Downloading: {url}")
        try:
            r = requests.get(url)
        except Exception as e:
            vprint(f"ERROR: Download unsuccessful (error: {e})")
            return None
        vprint(f"Saving file at: {fname}")
        with open(fname,"wb") as f:
            f.write(r.content)
        return fname

    def filename_from_pos(ra, dec, filter, survey):
        sign = np.sign(dec)
        if sign == -1:
            sign_str = 'n'
        else:
            sign_str = ''
        filename = f"{survey}_query_{ra:.4f}_{sign_str}{np.abs(dec):.4f}_{filter}.fits"
        return filename

class LegacySurveyQuery(QueryMixin):
    """Query to the LegacySurvey imaging data."""
    fitscutout = "https://www.legacysurvey.org/viewer/fits-cutout"
    default_layer = "ls-dr10"
    pixelsize_arcsec = 0.27

    @classmethod
    def getimage(cls, ra, dec, size=240, filters="r", images={}, output_dir="."):
        for f in filters:
            url = cls.fitscutout
            url += f"?ra={ra:.5f}&dec={dec:.5f}&layer={cls.default_layer}"
            url += f"&pixscale={cls.pixelsize_arcsec}&bands={f}&size={size}"
            filename = cls.filename_from_pos(ra, dec, f, "ls")
            output = os.path.join(output_dir, filename)
            print(output)
            filename = cls.download_image(url, fname=output)


            if filename is not None:
                intensity, wcs = cls.read_image(filename)
                images[f"LS.{f}"] = {"intensity": intensity, "wcs":wcs,
                                     "pix_size": cls.pixelsize_arcsec}

    def read_image(filename):
        vprint("Opening Legacy Survey fits file")
        with fits.open(filename) as hdul:
            wcs = WCS(hdul[0].header)
            intensity = hdul[0].data * 3631e-9  # Jy
        return intensity, wcs

class PSQuery(QueryMixin):
    """Query to the PANSTARRS (DR2) Survey."""
    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    pixelsize_arcsec = 0.25

    def getimages(tra, tdec, size=240, filters="grizy", format="fits", imagetypes="stack"):    
        """

        Description
        -----------
        Code selected from:
        https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-python

        Query ps1filenames.py service for multiple positions to get a list of images
        This adds a url column to the table to retrieve the cutout.
        
        tra, tdec = list of positions in degrees
        size = image size in pixels (0.25 arcsec/pixel)
        filters = string with filters to include
        format = data format (options are "fits", "jpg", or "png")
        imagetypes = list of any of the acceptable image types.  Default is stack;
            other common choices include warp (single-epoch images), stack.wt (weight image),
            stack.mask, stack.exp (exposure time), stack.num (number of exposures),
            warp.wt, and warp.mask.  This parameter can be a list of strings or a
            comma-separated string.
    
        Returns an astropy table with the results
        """
        
        if format not in ("jpg","png","fits"):
            raise ValueError("format must be one of jpg, png, fits")
        # if imagetypes is a list, convert to a comma-separated string
        if not isinstance(imagetypes,str):
            imagetypes = ",".join(imagetypes)
        # put the positions in an in-memory file object
        cbuf = StringIO()
        cbuf.write('\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra,tdec)]))
        cbuf.seek(0)
        # use requests.post to pass in positions as a file
        r = requests.post(PSQuery.ps1filename, data=dict(filters=filters, type=imagetypes),
            files=dict(file=cbuf))
        r.raise_for_status()
        tab = Table.read(r.text, format="ascii")
    
        urlbase = "{}?size={}&format={}".format(PSQuery.fitscut,size,format)
        tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase,ra,dec,filename)
                for (filename,ra,dec) in zip(tab["filename"],tab["ra"],tab["dec"])]
        return tab

    @classmethod
    def getimage(cls, ra, dec, size=240, filters="grizy", format="fits", imagetypes="stack",
                 images={}, output_dir="."):
        
        """Query ps1filenames.py service for multiple positions to get a list of images
        This adds a url column to the table to retrieve the cutout.
        
        tra, tdec = list of positions in degrees
        size = image size in pixels (0.25 arcsec/pixel)
        filters = string with filters to include
        format = data format (options are "fits", "jpg", or "png")
        imagetypes = list of any of the acceptable image types.  Default is stack;
            other common choices include warp (single-epoch images), stack.wt (weight image),
            stack.mask, stack.exp (exposure time), stack.num (number of exposures),
            warp.wt, and warp.mask.  This parameter can be a list of strings or a
            comma-separated string.
    
        Returns an astropy table with the results
        """
        
        if format not in ("jpg","png","fits"):
            raise ValueError("format must be one of jpg, png, fits")
        # if imagetypes is a list, convert to a comma-separated string
        if not isinstance(imagetypes,str):
            imagetypes = ",".join(imagetypes)
        # put the positions in an in-memory file object
        cbuf = StringIO()
        cbuf.write('\n'.join(["{} {}".format(ra, dec)]))
        cbuf.seek(0)
        # use requests.post to pass in positions as a file
        r = requests.post(cls.ps1filename, data=dict(filters=filters, type=imagetypes),
            files=dict(file=cbuf))
        r.raise_for_status()
        tab = Table.read(r.text, format="ascii")
    
        urlbase = "{}?size={}&format={}".format(cls.fitscut,size,format)
        tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase,ra,dec,filename)
                for (filename,ra,dec) in zip(tab["filename"],tab["ra"],tab["dec"])]
        # Retrieve each filter image
        for row in tab:
            print(row)
            vprint(f"Retrieving cutout: ra={row['ra']}, dec={row['dec']}, filter={row['filter']}")
            sign = np.sign(row['dec'])
            if sign == -1:
                sign_str = 'n'
            else:
                sign_str = ''
            filename = f"ps_query_{row['ra']:.4f}_{sign_str}{np.abs(row['dec']):.4f}_{row['filter']}.fits"
            filename.replace("-", "n")
            output = os.path.join(output_dir, filename)
            filename = cls.download_image(row['url'], fname=output)
            if filename is not None:
                intensity, wcs = cls.read_ps_fits(filename)
                images[f"PS1.{row['filter']}"] = {
                        "intensity": intensity, "wcs":wcs,
                        "pix_size": cls.pixelsize_arcsec}

        return images

    def read_ps_fits(fname):
        """Load a PANSTARRS image."""
        vprint("Opening PANSTARRS fits file")
        with fits.open(fname) as hdul:
            wcs = WCS(hdul[0].header)
            zp = hdul[0].header['FPA.ZP']
            exptime = hdul[0].header['exptime']
            ps_pix_area = 0.25**2  # arcsec^2/pix
            intensity_zp = 10**(0.4 * zp)
            intensity = 3631 * hdul[0].data / intensity_zp / exptime
            #sb = -2.5 * np.log10(hdul[0].data / ps_pix_area) + zp + 2.5*np.log10(exptime)
            #intensity = 10**(-0.4 * sb) * 3631  # jy
        return intensity, wcs


def get_effective_sky_footprint(data_containers):
    """Compute the effective footprint containing all input DataContainers.
    
    Return
    ------
    - centre: tuple
        Centre coordinates (ra_mean, dec_mean)
    - width: tuple
        Window width (ra_max - ra_min, dec_max - dec_min)
    """
    data_containers_footprint = []
    for dc in data_containers:
        data_containers_footprint.append(dc.get_footprint())
    # Select a rectangle containing all footprints
    max_ra, max_dec = np.nanmax(data_containers_footprint, axis=(0, 1))
    min_ra, min_dec = np.nanmin(data_containers_footprint, axis=(0, 1))
    ra_cen, dec_cen = (max_ra + min_ra) / 2, (max_dec + min_dec) / 2
    ra_width, dec_width = max_ra - min_ra, max_dec - min_dec
    vprint("Combined footprint Fov: {}, {}".format(ra_width * 60,
                                                    dec_width * 60))
    return (ra_cen, dec_cen), (ra_width, dec_width)

def query_image(data_containers, query=PSQuery, filters='r', im_extra_size_arcsec=30,
                im_output_dir='.'):
    """Perform a query of external images that overlap with the DataContainers.
    
    This method performs a query to the database of some photometric survey (e.g. PS)
    and retrieves a set of images that overlap with the IFS data.

    Parameters
    ----------
    survey : str, default="PS"
        Name of the external survey/database to perform the query. At present only PS
        queries are available.
    filters : str, default='r'
        String containing the filters to be included in the query (e.g. "ugriz").
    im_extra_size_arcsec : float, default=30
        Additional extension of the images in arcseconds with respect to the net FoV
        of the DataContainers.
    im_output_dir : str, default='.'
        Path to a directory where the queried images will be stored. Default is current
        working directory.

    Returns
    -------
    images : dict
        Dictionary containing the image data (``intensity``), wcs (``wcs``)
        and pixel size (``pix_size``) of each queried filter.
    """
    vprint("Querying image to external database")

    # Compute the effective footprint of all DC and use that as input for
    # the query
    im_pos, im_fov = get_effective_sky_footprint(data_containers)
    # Convert the size to pixels
    im_size_pix = int(
        (np.max(im_fov) * 3600 + im_extra_size_arcsec
            ) / query.pixelsize_arcsec)
    vprint(f"Image center sky position (RA, DEC): {im_pos}")
    vprint(f"Image size (pixels): {im_size_pix}")
    # Perform the query
    images = query.getimage(*im_pos, size=im_size_pix, filters=filters,
                            output_dir=im_output_dir)
    return images

def get_dc_aperture_flux(data_container, filter_name, aperture_diameter=1.25,
                         sample_every=2):
    """Compute aperture fluxes from the DataContainers

    This method computes a set of aperture fluxes from an input data container.
    If the input DC is a Cube, a grid of equally-spaced apertures will be
    computed. If the input DC is a RSS, the fibre positions will be used as
    reference apertures.

    Parameters
    ----------
    filter_names: list
        A list of filter names to initialise a list of
        :class:`pst.observables.Filter` objects.
    dc_intensity_units: `astropy.units.Quantity`, default=1e-16 erg/s/AA/cm2
        Intensity units of the DC.
    aperture_diameter: float
        Diameter size of the circular apertures. In the case of an RSS, this
        will match the size of the fibres.
    sample_every: int, default=2
        Spatial aperture sampling in units of the aperture radius. If
        `sample_every=2`, the aperture will be defined every two aperture
        diameters in the image.

    Returns
    -------
    dc_photometry: dict
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

    result = {}
    
    photometric_filter = Filter(filter_name=filter_name)

    synth_photo, synth_photo_err = get_synthetic_photometry(
        photometric_filter, data_container)
    result['synth_photo'] = synth_photo
    result['synth_photo_err'] = synth_photo_err
    if isinstance(data_container, Cube):
        vprint("Computing aperture fluxes using Cube synthetic"
                + "photometry")
        # Create a grid of apertures equally spaced
        pix_size_arcsec = np.max(data_container.wcs.celestial.wcs.cdelt) * 3600
        delta_pix = aperture_diameter / pix_size_arcsec * sample_every
        vprint("Creating a grid of circular aperture "
                + f"(rad={aperture_diameter / 2 / pix_size_arcsec:.2f}"
                + f" px) every {delta_pix:.1f} pixels")
        rows = np.arange(0, synth_photo.shape[0], delta_pix)
        columns = np.arange(0, synth_photo.shape[1], delta_pix)
        yy, xx = np.meshgrid(rows, columns)
        coordinates = data_container.wcs.celestial.pixel_to_world(
            xx.flatten(), yy.flatten())
        apertures = SkyCircularAperture(
            coordinates, r=aperture_diameter / 2 * u.arcsec)
        vprint(f"Total number of apertures: {len(apertures)}")
        reference_table = ApertureStats(
            data=synth_photo, error=synth_photo_err,
        aperture=apertures, wcs=data_container.wcs.celestial, sum_method='exact')
        # Compute the total flux in the aperture using the mean value
        flux_in_ap = reference_table.mean * np.sqrt(
            reference_table.center_aper_area.value)
        # Compute standard error from the std
        flux_in_ap_err = reference_table.sum_err
        result['wcs'] = data_container.wcs.celestial.deepcopy()
    elif isinstance(data_container, RSS):
        vprint("Using RSS synthetic photometry as apertures")
        coordinates = SkyCoord(data_container.info['fib_ra'],
                               data_container.info['fib_dec'])
        flux_in_ap, flux_in_ap_err = synth_photo, synth_photo_err
        result['wcs'] = None

    # Make a QC plot of the apertures
    fig = make_plot_apertures(
        data_container, synth_photo, synth_photo_err, coordinates, flux_in_ap,
        flux_in_ap_err)
    # Store the results
    result['fig'] = fig
    result['coordinates'] = coordinates
    result['aperture_flux'] = flux_in_ap
    result['aperture_flux_err'] = flux_in_ap_err
    result['aperture_mask'] =  np.isfinite(flux_in_ap
                                           ) & np.isfinite(flux_in_ap_err)
    return result

def get_dc_aperture_fluxes(data_containers, filter_names,
                           aperture_diameter=1.25, sample_every=2):
        """Compute aperture fluxes from the DataContainers
        
        This method computes a set of aperture fluxes from an input data container.
        If the input DC is a Cube, a grid of equally-spaced apertures will be
        computed. If the input DC is a RSS, the fibre positions will be used as
        reference apertures.

        Parameters
        ----------
        filter_names: list
            A list of filter names to initialise a list of
            :class:`pst.observables.Filter` objects.
        dc_intensity_units: `astropy.units.Quantity`, default=1e-16 erg/s/AA/cm2
            Intensity units of the DC.
        aperture_diameter: float
            Diameter size of the circular apertures. In the case of an RSS, this
            will match the size of the fibres.
        sample_every: int, default=2
            Spatial aperture sampling in units of the aperture radius. If
            `sample_every=2`, the aperture will be defined every two aperture
            diameters in the image.

        Returns
        -------
        dc_photometry: dict
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

        dc_photometry = {}
        for photo_filter in filter_names:
            dc_photometry[photo_filter] = {
                'synth_photo': [], 'synth_photo_err': [],
                'wcs': [],
                'coordinates': [], 'aperture_flux': [],
                'aperture_flux_err': [],
                'aperture_mask': [],
                'figs': []}
            photometric_filter = Filter(filter_name=photo_filter)
            # Compute the synthetic photometry on each DC
            for dc in data_containers:
                synth_photo, synth_photo_err = get_synthetic_photometry(
                    photometric_filter, dc)
                dc_photometry[photo_filter]['synth_photo'].append(synth_photo)
                dc_photometry[photo_filter]['synth_photo_err'].append(synth_photo_err)
                if isinstance(dc, Cube):
                    vprint("Computing aperture fluxes using Cube synthetic"
                           + "photometry")
                    # Create a grid of apertures equally spaced
                    pix_size_arcsec = np.max(dc.wcs.celestial.wcs.cdelt) * 3600
                    delta_pix = aperture_diameter / pix_size_arcsec * sample_every
                    vprint("Creating a grid of circular aperture "
                           + f"(rad={aperture_diameter / 2 / pix_size_arcsec:.2f}"
                           + f" px) every {delta_pix:.1f} pixels")
                    rows = np.arange(0, synth_photo.shape[0], delta_pix)
                    columns = np.arange(0, synth_photo.shape[1], delta_pix)
                    yy, xx = np.meshgrid(rows, columns)
                    coordinates = dc.wcs.celestial.pixel_to_world(xx.flatten(), yy.flatten())
                    apertures = SkyCircularAperture(
                        coordinates, r=aperture_diameter / 2 * u.arcsec)
                    vprint(f"Total number of apertures: {len(apertures)}")
                    reference_table = ApertureStats(
                        data=synth_photo, error=synth_photo_err,
                    aperture=apertures, wcs=dc.wcs.celestial, sum_method='exact')
                    # Compute the total flux in the aperture using the mean value
                    flux_in_ap = reference_table.mean * np.sqrt(
                        reference_table.center_aper_area.value)
                    # Compute standard error from the std
                    flux_in_ap_err = reference_table.sum_err
                    dc_photometry[photo_filter]['wcs'].append(dc.wcs.celestial.deepcopy())
                elif isinstance(dc, RSS):
                    vprint("Using RSS synthetic photometry as apertures")
                    coordinates = SkyCoord(dc.info['fib_ra'], dc.info['fib_dec'])
                    flux_in_ap, flux_in_ap_err = synth_photo, synth_photo_err
                    dc_photometry[photo_filter]['wcs'].append(None)

                # Make a QC plot of the apertures
                fig = make_plot_apertures(
                    dc, synth_photo, synth_photo_err, coordinates, flux_in_ap,
                    flux_in_ap_err)
                # Store the results
                dc_photometry[photo_filter]['figs'].append(fig)
                dc_photometry[photo_filter]['coordinates'].append(coordinates)
                dc_photometry[photo_filter]['aperture_flux'].append(flux_in_ap)
                dc_photometry[photo_filter]['aperture_flux_err'].append(flux_in_ap_err)
                dc_photometry[photo_filter]['aperture_mask'].append(
                    np.isfinite(flux_in_ap) & np.isfinite(flux_in_ap_err))
        return dc_photometry

def make_plot_apertures(dc, synth_phot, synth_phot_err, ap_coords,
                            ap_flux, ap_flux_err):
        """Plot the synthetic aperture fluxes measured from a DC."""
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

def get_synthetic_photometry(filter, dc):
    """Compute synthetic photometry from a DataContainer.
    
    This method extracts synthetic photometry using the spectral information
    of a DataContainer.

    Parameters
    ----------
    filter: :class:`pst.observables.Filter`
        A Filter for computing the synthetic photometry from the spectra.
    dc: :class:`DataContainer`
        The input DataContainer.
    dc_intensity_units: :class:`astropy.units.Quantity`
        The units of the intensity of the DC.

    Returns
    -------
    - synth_phot: :class:`np.ndarray`
        Array containing the flux estimates expressed in Jy.
    - synth_phot_err: :class:`np.ndarray`
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
            if not isinstance(intensity, u.Quantity):
                intensity = intensity * 1e-16 * u.erg / u.s / u.cm**2 / u.angstrom
                var = var * (1e-16 * u.erg / u.s / u.cm**2 / u.angstrom)**2
            f_nu, f_nu_err = filter.get_fnu(intensity, var**0.5)
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
            if not isinstance(intensity, u.Quantity):
                intensity = intensity * 1e-16 * u.erg / u.s / u.cm**2 / u.angstrom
                var = var * (1e-16 * u.erg / u.s / u.cm**2 / u.angstrom)**2
            f_nu, f_nu_err = filter.get_fnu(intensity, var**0.5)
            synth_photo[ith] = f_nu.value
            synth_photo_err[ith] = f_nu_err.value

    return synth_photo, synth_photo_err

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

    The figure of merit used is defined as:
        :math: `w=e^{-(A+B)/2}`
        where 
        :math: `A=\langle f_{Ap} - \hat{f_{Ap}} \rangle`
        :math: `B=|1 - \frac{\langle f_{Ap} \rangle}{\langle \hat{f_{Ap}} \rangle}|`
        where :math: `f_{Ap}, \hat{f_{Ap}` correspond to the aperture flux in the
        reference (DC) and ancillary data

    Parameters
    ----------
    ref_aperture_flux: np.ndarray
        A set of aperture fluxes measured in the target image.
    ref_aperture_flux_err: np.ndarray
        The associated errors of the aperture fluxes.
    ref_coords: astropy.coordinates.SkyCoord
        The celestial coordinates of each aperture.
    image: np.ndarray
        Pixel data associated to the reference image.
    wcs: astropy.wcs.WCS
        WCS associated to the reference image.
    ra_offset_range: list or tuple, default=[-10, 10],
        The range of offsets in RA to be explored in arcseconds.
        Defaul is +-10 arcsec.
    dec_offset_range: list or tuple, default=[-10, 10],
        The range of offsets in DEC to be explored in arcseconds.
        Defaul is +-10 arcsec.
    offset_step: float, default=0.5
        Offset step size in arcseconds. Default is 0.5.
    aperture_diameter: float, default=1.25
        Aperture diameter size in arcseconds. Default is 1.25.

    Returns
    -------
    results: dict
        Dictionary containing the cross-correlation results.
    """
    vprint("Cross-correlating image to list of apertures")
    vprint(f"Input number of apertures: {len(ref_aperture_flux)}")
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

    vprint("Computing the offset solution")
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
    results: dict
        Dictionary containing the results returned by `crosscorrelate_im_apertures`.

    Returns
    -------
    fig : :class:`plt.Figure`
        Figure containing the plot.
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

def make_plot_astrometry_offset(ref_image, ref_wcs, image, results):
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
