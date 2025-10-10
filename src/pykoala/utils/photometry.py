"""
This module contains tools for creating synthetic photometry from DataContainers
as well as tools for retrieveing and manipulating external imaging data.

""" 
import os
import requests
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.table import Table
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture, ApertureStats

from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

from pykoala import vprint
from pykoala.data_container import Cube, RSS
from pykoala.ancillary import update_wcs_coords
from pykoala.plotting import utils

class QueryMixin:
    """Mixin with common methods for image queries to external databases."""

    def download_image(url, filename):
        """Download an image and write it to a binary file.
        
        Parameters
        ----------
        url : str
            URL to download the image.
        filename : str
            Name of the output binary file.
        
        Returns
        -------
        filename : str
            Output filename. If the download is unsuccessful, the return value
            is None.
        """
        vprint(f"Downloading: {url}")
        try:
            r = requests.get(url)
        except Exception as e:
            vprint(f"ERROR: Download unsuccessful (error: {e})")
            return None
        vprint(f"Saving file at: {filename}")
        with open(filename,"wb") as f:
            f.write(r.content)
        return filename

    def filename_from_pos(ra, dec, filter, survey):
        """Convenience function for creating a filename based on the query information.
        
        This method creates a filename using the central (RA, DEC) values of the
        querie, the filter name associated to the queried image, and the
        survey/database name. The resulting filename uses the following convention:
        ``SURVEY_query_RA_DEC_FILTER.fits``, where negative values of DEC include
        the prefix ``n`` (i.e. `n30` for ``dec=-30``)

        Parameters
        ----------
        ra : float
            Reference RA position in deg.
        dec : float
            Reference DEC position in deg.
        filter : str
            Photometric filter name.
        survey : str
            Survey name.
        
        Returns
        -------
        filename : str
            Output filename
        """
        sign = np.sign(dec)
        if sign == -1:
            sign_str = 'n'
        else:
            sign_str = ''
        filename = f"{survey}_query_{ra:.4f}_{sign_str}{np.abs(dec):.4f}_{filter}.fits"
        return filename


class LegacySurveyQuery(QueryMixin):
    """Utility to query `LegacySurvey <https://www.legacysurvey.org>`_ imaging data.
    
    Attributes
    ----------
    fitscutout : str
        URL to the online cutout service.
    dafault_layer : str, default=``ls-dr10``
        LS layer used to perform the query.
    pixelsize_arcsec : size of the pixels in angstrom.
    """
    fitscutout = "https://www.legacysurvey.org/viewer/fits-cutout"
    default_layer = "ls-dr10"
    pixelsize_arcsec = 0.27 << u.arcsec

    @classmethod
    def getimage(cls, im_coords, size=240, filters="r", images={}, output_dir="."):
        ra, dec = im_coords.ra.to_value("deg"), im_coords.dec.to_value("deg")
        for f in filters:
            url = cls.fitscutout
            url += f"?ra={ra:.5f}&dec={dec:.5f}&layer={cls.default_layer}"
            url += f"&pixscale={cls.pixelsize_arcsec}&bands={f}&size={size}"
            filename = cls.filename_from_pos(ra, dec, f, "ls")
            output = os.path.join(output_dir, filename)
            filename = cls.download_image(url, filename=output)


            if filename is not None:
                ccd = cls.read_image(filename)
                images[f"LS.{f}"] = {"ccd": ccd,
                                     "pix_size": cls.pixelsize_arcsec}

    def read_image(filename):
        """Read a LS fits image.
        
        Parameters
        ----------
        filename : str
            Path to the FITS file containing the image data.

        Returns
        -------
        ccd : ``astropy.nddata.CCDData``
            LS image data. The default data units are Jy.
        """
        vprint("Opening Legacy Survey fits file")
        ccd = CCDData.read(filename, hdu=0, unit="adu")
        # Convert ADU to Jy
        ccd = ccd.multiply(3631e-9 << u.Jy / u.adu)
        return ccd


class PSQuery(QueryMixin):
    """Query to the PANSTARRS (DR2) Survey.
    
    Attributes
    ----------
    ps1filename : str
        URL to the PS query service
    fitscut : str
        URL to the cutour service
    pixelsize_arcsec : size of the pixels in angstrom.
    """

    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    pixelsize_arcsec = 0.25 << u.arcsec

    def getimages(tra, tdec, size=240, filters="grizy", format="fits",
                  imagetypes="stack"):    
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
    def getimage(cls, im_coords, size=240, filters="grizy", format="fits",
                 imagetypes="stack", images={}, output_dir="."):
        
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
        cbuf.write('\n'.join(["{} {}".format(im_coords.ra.to_value("deg"),
                                             im_coords.dec.to_value("deg"))]))
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
            vprint(f"Retrieving cutout: ra={row['ra']}, dec={row['dec']}, filter={row['filter']}")
            sign = np.sign(row['dec'])
            if sign == -1:
                sign_str = 'n'
            else:
                sign_str = ''
            filename = f"ps_query_{row['ra']:.4f}_{sign_str}{np.abs(row['dec']):.4f}_{row['filter']}.fits"
            filename.replace("-", "n")
            output = os.path.join(output_dir, filename)
            filename = cls.download_image(row['url'], filename=output)
            if filename is not None:
                ccd = cls.read_ps_fits(filename)
                images[f"PS1.{row['filter']}"] = {
                        "ccd": ccd, "pix_size": cls.pixelsize_arcsec}

        return images

    def read_ps_fits(filename):
        """Load a PANSTARRS image.
        
        Parameters
        ----------
        filename : str
            Path to the FITS file containing the image data.

        Returns
        -------
        ccd : ``astropy.nddata.CCDData``
            PS image data. The default data units are Jy.
        """
        vprint("Opening PANSTARRS fits file")
        ccd = CCDData.read(filename, hdu=0, unit="adu")
        # Convert ADU to Jy
        ccd = ccd.multiply(3631 * 10**(-0.4 * ccd.header["FPA.ZP"]
                                ) / ccd.header["exptime"] << u.Jy / u.adu)
        return ccd


def get_effective_sky_footprint(data_containers):
    """Compute the effective footprint containing all input DataContainers.
    
    Returns
    -------
    centre: tuple
        Centre coordinates (ra_mean, dec_mean)
    width: tuple
        Window width (ra_max - ra_min, dec_max - dec_min)
    """
    data_containers_footprint = []
    for dc in data_containers:
        data_containers_footprint.append(dc.get_footprint())
    # Select a rectangle containing all footprints
    # TODO: Implement units on DC and remove << deg
    max_ra, max_dec = np.nanmax(data_containers_footprint, axis=(0, 1)) << u.deg
    min_ra, min_dec = np.nanmin(data_containers_footprint, axis=(0, 1)) << u.deg
    centre_coordinates = SkyCoord((max_ra + min_ra) / 2, (max_dec + min_dec) / 2)
    ra_width, dec_width = max_ra - min_ra, max_dec - min_dec
    vprint("Combined footprint Fov: {}, {}".format(ra_width.to("arcmin"),
                                                   dec_width.to("arcmin")))
    return centre_coordinates, (ra_width, dec_width)

def query_image(data_containers, query=PSQuery, filters='r',
                im_extra_size_arcsec=30 << u.arcsec, im_output_dir='.'):
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
    im_coords, im_fov = get_effective_sky_footprint(data_containers)
    # Convert the size to pixels
    ra_pixels = ((im_fov[0] + im_extra_size_arcsec) / query.pixelsize_arcsec
                 ).to_value(u.dimensionless_unscaled)
    dec_pixels = ((im_fov[1] + im_extra_size_arcsec) / query.pixelsize_arcsec
                  ).to_value(u.dimensionless_unscaled)

    im_size_pix = int(np.max([ra_pixels, dec_pixels]))
    vprint(f"Image center sky position (RA, DEC): {im_coords}")
    vprint(f"Image size (pixels): {im_size_pix}")
    # Perform the query
    images = query.getimage(im_coords, size=im_size_pix, filters=filters,
                            output_dir=im_output_dir)
    return images

def get_aperture_photometry(coordinates : SkyCoord, diameters : u.Quantity,
                            image : CCDData):
    """Compute the aperture photometry from an input image.
    
    Convenient method to compute aperture photometry using multiple circular
    apertures. The flux within each aperture is computed as the mean value
    multiplied by the exact number of pixels contained within.

    Parameters
    ----------
    coordinates : :class:`astropy.coordinates.SkyCoord`
        Celestial coordinates where to compute the apertures.
    diameters : :class:`astropy.units.Quantity`
        Diameter size of each aperture. It can be a single or
        multiple values.
    image : :class:`astropy.nddata.CCDData`

    Returns
    -------
    flux_in_ap : :class:`astropy.units.Quantity`
        Flux contained within each aperture.
    flux_in_ap_err : :class:`astropy.units.Quantity`
        Associated error to ``flux_in_ap``.
    """
    apertures = SkyCircularAperture(coordinates, r=diameters / 2)
    table = ApertureStats(image, apertures, sum_method='exact')
    flux_in_ap = table.mean * np.sqrt(table.center_aper_area.value)
    flux_in_ap_err = table.sum_err
    return flux_in_ap, flux_in_ap_err


def get_dc_aperture_flux(data_container, filter_name,
                         aperture_diameter=1.25 << u.arcsec,
                         sample_every=2, rss_fibres_pct=50.0):
    """Compute aperture fluxes from the DataContainers

    This method computes a set of aperture fluxes from an input data container.
    If the input DC is a Cube, a grid of equally-spaced apertures will be
    computed. If the input DC is a RSS, the fibre positions will be used as
    reference apertures.

    Parameters
    ----------
    data_container : :class:`pykoala.data_container.SpectraContainer`
        DataContainer used to compute the synthetic photometry.
    filter_name: str
        Photometric filter name used to initalise a :class:`pst.observables.Filter`.
    aperture_diameter: float
        Diameter size of the circular apertures. In the case of an RSS, this
        will match the size of the fibres.
    sample_every: int, default=2
        Spatial aperture sampling in units of the aperture radius. If
        `sample_every=2`, the aperture will be defined every two aperture
        diameters in the image.
    rss_fibres_pct : float, optional. Default=50
        If the input DataContainer is :class:`RSS`, it determines the
        flux percentile threshold used to select valid fibre apertures.

    Returns
    -------
    dc_photometry: dict
        A dictionary containing the results of the computation

        - synth_photo: The synthetic photometry of the DC, it can be a list
        of fluxes (if the input DC is a RSS) or an image (if DC is a Cube).
        For more details, see the method `get_dc_synthetic_photometry`.
        - synth_photo_err: Associated error fo `synth_photo`.
        - wcs: WCS of the DC. If the DC is a RSS it will be None.
        - coordinates: Celestial position of the apertures stored as a list
        of `astropy.coordinates.Skycoord`.
        - aperture_mask: A mask for invalid aperture fluxes.
        - figs: A list of QC figures showing the .
    
    See also
    --------
    :class:`pst.observables.Filter`
    """
    try:
        from pst.observables import Filter
    except:
        raise ImportError("PST package not found")

    result = {}

    # Compute the synthetic photometry maps    
    photometric_filter = Filter.from_svo(filter_name)
    synth_photo, synth_photo_err = get_dc_synthetic_photometry(
        photometric_filter, data_container)
    result['synth_photo'] = synth_photo
    result['synth_photo_err'] = synth_photo_err

    if isinstance(data_container, Cube):
        vprint("Computing aperture fluxes using Cube synthetic photometry")
        # Create a grid of apertures equally spaced
        pix_size = np.max(
            data_container.wcs.celestial.wcs.cdelt) * 3600 * u.arcsec
        delta_pix = aperture_diameter / pix_size * sample_every
        vprint("Creating a grid of circular aperture "
                + f"(rad={aperture_diameter / 2 / pix_size:.2f}"
                + f" px) every {delta_pix:.1f} pixels")
        rows = np.arange(0, synth_photo.shape[0], delta_pix)
        columns = np.arange(0, synth_photo.shape[1], delta_pix)
        yy, xx = np.meshgrid(rows, columns)
        coordinates = data_container.wcs.celestial.pixel_to_world(
            xx.flatten(), yy.flatten())
        
        flux_in_ap, flux_in_ap_err = get_aperture_photometry(
            coordinates, diameters=aperture_diameter,
            image=CCDData(data=synth_photo, uncertainty=StdDevUncertainty(
                synth_photo_err),
            wcs=data_container.wcs.celestial))
        result['wcs'] = data_container.wcs.celestial.deepcopy()

    elif isinstance(data_container, RSS):
        vprint("Using RSS fibre synthetic photometry as apertures")
        mask = synth_photo > np.nanpercentile(synth_photo, rss_fibres_pct)
        # TODO: this should be a method or an attribute of RSS.
        coordinates = SkyCoord(data_container.info['fib_ra'][mask],
                               data_container.info['fib_dec'][mask],
                               unit='deg')
        flux_in_ap, flux_in_ap_err = synth_photo[mask], synth_photo_err[mask]
        result['wcs'] = None
        aperture_diameter = data_container.fibre_diameter

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

def make_plot_apertures(dc, synth_phot, synth_phot_err, ap_coords,
                            ap_flux, ap_flux_err):
    """Plot the synthetic aperture fluxes measured from a DC."""
    if isinstance(dc, Cube):
        fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True,
                                subplot_kw={'projection': dc.wcs.celestial},
                                constrained_layout=True)
        ax = axs[0, 0]
        mappable = ax.imshow(-2.5 * np.log10(synth_phot.to_value("3631 Jy")), vmin=16, vmax=23,
                                interpolation='none')
        plt.colorbar(mappable, ax=ax, label='SB (mag/pix)')
        ax = axs[0, 1]
        mappable = ax.imshow(synth_phot / synth_phot_err, vmin=0, vmax=10, cmap='jet',
                                interpolation='none')
        plt.colorbar(mappable, ax=ax, label='Flux SNR')
        ax = axs[1, 0]            
        mappable = ax.scatter(ap_coords.ra, ap_coords.dec,
        marker='o', transform=ax.get_transform('world'),
        c= -2.5 * np.log10(ap_flux.to_value("3631 Jy")), vmin=16, vmax=23)
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
        fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True,
                                constrained_layout=True)
        mappable = axs[0].scatter(dc.info['fib_ra'], dc.info['fib_dec'],
                                    c=synth_phot)
        plt.colorbar(mappable, label="Flux")
        mappable = axs[1].scatter(dc.info['fib_ra'], dc.info['fib_dec'],
                                    c=synth_phot / synth_phot_err, label="SNR")
        plt.colorbar(mappable)
        # TODO: once plotting is homogeneized add RSS fibre map
    return fig

def get_dc_synthetic_photometry(filter, dc):
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
    synth_photo = np.full((dc.rss_intensity.shape[0], 1), fill_value=np.nan) << u.Jy
    synth_photo_err = np.full_like(synth_photo, fill_value=np.nan)

    for ith, (intensity, variance) in enumerate(
        zip(dc.rss_intensity, dc.rss_variance)):
        mask = np.isfinite(intensity) & np.isfinite(variance)
        if not mask.any():
                continue
        #TODO: Temporarily until units are implemented in DC.
        if not isinstance(intensity, u.Quantity):
            intensity = intensity * 1e-16 * u.erg / u.s / u.cm**2 / u.angstrom
            variance = variance * (1e-16 * u.erg / u.s / u.cm**2 / u.angstrom)**2
        f_nu, f_nu_err = filter.get_fnu(intensity, variance**0.5)
        synth_photo[ith] = f_nu.to('Jy')
        synth_photo_err[ith] = f_nu_err.to('Jy')

    synth_photo = np.squeeze(dc.rss_to_original(synth_photo))
    synth_photo_err = np.squeeze(dc.rss_to_original(synth_photo_err))

    return synth_photo, synth_photo_err

# def crosscorrelate_im_apertures(ref_aperture_flux, ref_aperture_flux_err,
#                                 ref_coord, image,
#                                 ra_offset_range=[-10, 10],
#                                 dec_offset_range=[-10, 10],
#                                 offset_step=0.5,
#                                 aperture_diameter=1.25 << u.arcsec,
#                                 plot=True):
#     """Cross-correlate an image with an input set of apertures.
    
#     Description
#     -----------
#     This method performs a spatial cross-correlation between a list of input aperture fluxes
#     and a reference image. For example, the aperture fluxes can simply correspond to the
#     flux measured within a fibre or a set of aperture measured in a datacube.
#     First, a grid of aperture position offsets is generated, and every iteration
#     will compute the difference between the two sets. 

#     The figure of merit used is defined as:
#         :math: `w=e^{-(A+B)/2}`
#         where 
#         :math: `A=\langle f_{Ap} - \hat{f_{Ap}} \rangle`
#         :math: `B=|1 - \frac{\langle f_{Ap} \rangle}{\langle \hat{f_{Ap}} \rangle}|`
#         where :math: `f_{Ap}, \hat{f_{Ap}` correspond to the aperture flux in the
#         reference (DC) and ancillary data

#     Parameters
#     ----------
#     ref_aperture_flux: np.ndarray
#         A set of aperture fluxes measured in the target image.
#     ref_aperture_flux_err: np.ndarray
#         The associated errors of the aperture fluxes.
#     ref_coords: astropy.coordinates.SkyCoord
#         The celestial coordinates of each aperture.
#     image: np.ndarray
#         Pixel data associated to the reference image.
#     wcs: astropy.wcs.WCS
#         WCS associated to the reference image.
#     ra_offset_range: list or tuple, default=[-10, 10],
#         The range of offsets in RA to be explored in arcseconds.
#         Defaul is +-10 arcsec.
#     dec_offset_range: list or tuple, default=[-10, 10],
#         The range of offsets in DEC to be explored in arcseconds.
#         Defaul is +-10 arcsec.
#     offset_step: float, default=0.5
#         Offset step size in arcseconds. Default is 0.5.
#     aperture_diameter: float, default=1.25
#         Aperture diameter size in arcseconds. Default is 1.25.

#     Returns
#     -------
#     results: dict
#         Dictionary containing the cross-correlation results.
#     """
#     vprint("Cross-correlating image to list of apertures")
#     vprint(f"Input number of apertures: {len(ref_aperture_flux)}")
#     # Renormalize the reference aperture
#     ref_ap_norm = np.nanmean(ref_aperture_flux)
#     ref_flux = ref_aperture_flux / ref_ap_norm
#     ref_flux_err = ref_aperture_flux_err / ref_ap_norm
#     good_aper = np.isfinite(ref_flux) & (ref_flux > 0)
#     vprint(f"Number of input apertures used: {np.count_nonzero(good_aper)}")
#     # Make a grid of offsets
#     ra_offset = np.arange(*ra_offset_range, offset_step)
#     dec_offset = np.arange(*dec_offset_range, offset_step)

#     # Initialise the results variable

#     offset_sampling = np.meshgrid(dec_offset, ra_offset, indexing='ij')
#     grid_flux_prod = np.full((dec_offset.size, ra_offset.size),
#                               fill_value=np.nan)

#     for i, (ra_offset_arcsec, dec_offset_arcsec) in enumerate(
#         zip(offset_sampling[0].flatten(), offset_sampling[1].flatten())):
#         # Create a set of apertures
#         new_coords = SkyCoord(
#             ref_coord.ra + ra_offset_arcsec * u.arcsec,
#             ref_coord.dec + dec_offset_arcsec * u.arcsec)

#         flux_in_ap, flux_in_ap_err = get_aperture_photometry(
#         new_coords, diameters=aperture_diameter, image=image["ccd"])

#         # Renormalize the flux
#         flux_in_ap_norm = np.nanmean(flux_in_ap)
#         flux_in_ap /= flux_in_ap_norm
#         flux_in_ap_err /= flux_in_ap_norm
#         # Ensure that none of the apertures contain NaN's/inf
#         mask = np.isfinite(flux_in_ap) & good_aper
#         n_valid = np.count_nonzero(mask)
#         idx = np.unravel_index(i, grid_flux_prod.shape)
#         grid_flux_prod[idx] = np.nansum(
#             flux_in_ap[mask] * ref_flux[mask]) / n_valid

#     vprint("Computing the offset solution")
#     weights = np.nanmax(grid_flux_prod) - grid_flux_prod
#     weights = np.exp(grid_flux_prod)
#     weights /= np.nansum(weights)
#     # Minimum
#     min_pos = np.nanargmax(weights)
#     min_pos = np.unravel_index(min_pos, grid_flux_prod.shape)

#     ra_min = offset_sampling[1][min_pos]
#     dec_min = offset_sampling[0][min_pos]
#     ra_mean = np.nansum(offset_sampling[1] * weights)
#     ra_var = np.nansum((offset_sampling[1] - ra_mean)**2 * weights)
#     dec_mean = np.nansum(offset_sampling[0] * weights)
#     dec_var = np.nansum((offset_sampling[0] - dec_mean)**2 * weights)

#     results = {
#         "offset_min": (ra_min, dec_min),
#         "offset_mean": (ra_mean, dec_mean),
#         "offset_var": (ra_var, dec_var),
#         "ra_offset": ra_offset, "dec_offset": dec_offset,
#         "offset_sampling": offset_sampling,
#         "weights": weights}

#     if plot:
#         results['fig'] = make_crosscorr_plot(results)
#     return results

# def make_crosscorr_plot(results):
#     """Make a plot showing the aperture flux cross-correlatation results.
    
#     Parameters
#     ----------
#     results: dict
#         Dictionary containing the results returned by `crosscorrelate_im_apertures`.

#     Returns
#     -------
#     fig : :class:`plt.Figure`
#         Figure containing the plot.
#     """
#     ra_mean, dec_mean = results['offset_mean']
#     ra_min, dec_min = results['offset_min']

#     fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True,
#                            figsize=(4, 4),
#                            sharex=False, sharey=False)

#     mappable = ax.pcolormesh(
#         results['ra_offset'], results['dec_offset'], results['weights'],
#         cmap='Spectral')
#     ax.plot(ra_mean, dec_mean, 'o', ms=10, label='Mean', mec='k', mfc='none')
#     ax.plot(ra_min, dec_min, 'k+', label='Max. like')
#     #ax.legend(framealpha=0.1)
#     ax.set_xlabel("RA offset (arcsec)")
#     ax.set_ylabel("DEC offset (arcsec)")
#     plt.colorbar(mappable, ax=ax, label='W',
#                  orientation='horizontal', pad=0.2)
    
#     vax = ax.inset_axes((1.03, 0, .6, 1), sharey=ax)
#     vax.plot(results['weights'].sum(axis=1), results['dec_offset'], 'k')
#     vax.axhline(dec_min, color='b', label='Max. weight')
#     vax.axhline(dec_mean, color='r', label='Avg.')
#     vax.legend()

#     hax = ax.inset_axes((0, 1.03, 1, .6), sharex=ax)
#     hax.plot(results['ra_offset'], results['weights'].sum(axis=0), 'k')
#     hax.axvline(ra_min, color='b')
#     hax.axvline(ra_mean, color='r')

#     plt.close(fig)
#     return fig

def crosscorrelate_im_apertures(ref_aperture_flux, ref_coord, image,
                                ra_offset_range=[-10., 10.],
                                dec_offset_range=[-10., 10.],
                                aperture_diameter=1.25 << u.arcsec,
                                smooth_image_sigma=1 << u.arcsec):
    """Cross-correlate an image with an input set of apertures.
    
    Description
    -----------
    This method performs a spatial cross-correlation between a list of input aperture fluxes
    and a reference image. For example, the aperture fluxes can simply correspond to the
    flux measured within a fibre or a set of aperture measured in a datacube.

    The objective function is defined as:
        :math: `f(\Delta RA, \Delta DEC)= \frac{1}{\sum f_i \hat{f_i}(\Delta RA, \Delta DEC)}`
        where :math: `f_i` and :math: `\hat{f_i}` correspond to the flux measured
        within the reference and image apertures.


    Parameters
    ----------
    ref_aperture_flux: np.ndarray
        A set of aperture fluxes measured in the target image.
    ref_coords: astropy.coordinates.SkyCoord
        The celestial coordinates of each aperture.
    image: np.ndarray
        Pixel data associated to the reference image.
    ra_offset_range: list or tuple, default=[-10, 10],
        The range of offsets in RA to be explored in arcseconds.
        Defaul is +-10 arcsec.
    dec_offset_range: list or tuple, default=[-10, 10],
        The range of offsets in DEC to be explored in arcseconds.
        Defaul is +-10 arcsec.
    aperture_diameter: :class:`astropy.units.Quantity`, default=1.25 arcsec
        Aperture diameter size in arcseconds. Default is 1.25.
    smooth_image_sigma: :class:`astropy.units.Quantity` or None, default=1 arcsec
        Gaussian smoothing standard deviation applied to the reference image before
        the cross-correlation. If ``None``, no smoothing is applied.

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
    good_aper = np.isfinite(ref_flux) & (ref_flux > 0)
    vprint(f"Number of input apertures used: {np.count_nonzero(good_aper)}")
    # Make a grid of offsets
    if smooth_image_sigma is not None:
        image["ccd"].data = gaussian_filter(
            image["ccd"].data,
            smooth_image_sigma.to_value("arcsec") / image["pix_size"].to_value("arcsec"))

    def objective_function(offsets):
        # Create the new aperture coordinates
        new_coords = SkyCoord(
            ref_coord.ra + offsets[0] * u.arcsec,
            ref_coord.dec + offsets[1] * u.arcsec)

        flux_in_ap, _ = get_aperture_photometry(
        new_coords, diameters=aperture_diameter, image=image["ccd"])

        # Renormalize the flux
        flux_in_ap_norm = np.nanmean(flux_in_ap)
        flux_in_ap /= flux_in_ap_norm
        # Ensure that none of the apertures contain NaN's/inf
        mask = np.isfinite(flux_in_ap) & good_aper
        n_valid = np.count_nonzero(mask)
        value = 1 / (np.nansum(flux_in_ap[mask] * ref_flux[mask]) / n_valid)
        return value

    vprint("Performing minimization...")
    result = minimize(objective_function, x0=[0., 0.],
                      bounds=(ra_offset_range, dec_offset_range),
                      method="Powell",
                      tol=1e-9,
                      )
    vprint(f"Success : {result.success}")
    vprint(f"Status : {result.status}")
    vprint(f"Result : {result.x}")
    vprint(f"Number of evaluations : {result.nfev}")
    results = {"offset_min": result.x << u.arcsec, "opt_results": result}

    return results

def make_plot_astrometry_offset(data_container, dc_synth_photo, image, results):
        """Plot the DC and ancillary data including the astrometry correction.
        
        Parameters
        ----------
        data_container : :class:`DataContainer`
            DataContainer used during the cross-correlation.
        dc_synth_photo : :class:`astropy.units.Quantity`
            DataContainer synthetic photometry.
        image : dict
            Dictionary including the keys ``ccd`` and ``pix_size``, containing
            the :class:`CCDData` and pixel size information of the reference
            image.
        results : dict
            Dictionary containing the results of the cross-correlation.
        
        Returns
        -------
        fig : :class:`plt.Figure`
            Figure containing the plots.
        """

        synt_sb = -2.5 * np.log10(dc_synth_photo.to_value("3631 Jy"))
        im_sb = -2.5 * np.log10(
            image['ccd'].data * image['ccd'].unit.to("3631 Jy")
            / image['pix_size'].to_value("arcsec")**2)

        dc_contour_params = dict(levels=[17, 18, 19, 20, 21, 22], colors='r')
        contour_params = dict(levels=[17, 18, 19, 20, 21, 22], colors='k')

        offset_str = (f"{results['offset_min'][0]:.1f}"
                      + f", {results['offset_min'][1]:.1f}")
        
        fig, axs = utils.new_figure(f"Astrometry Offset ({offset_str})", figsize=(10, 4),
                                    ncols=2, sharex=True, sharey=True,
                                    subplot_kw=dict(projection=image['ccd'].wcs),
                                    gridspec_kw=dict(wspace=0.1),
                                    squeeze=True,
                                    constrained_layout=True,
                                    tweak_axes=False
                                    )
        axs[0].set_title("Original")
        axs[1].set_title("Corrected")

        for ax in axs:
            ax.coords.grid(True, color='orange', ls='solid')
            ax.coords[0].set_format_unit('deg')
            cs = ax.contour(im_sb, **contour_params)

        if isinstance(data_container, Cube):
            cs = axs[0].contour(synt_sb, transform=axs[0].get_transform(
                data_container.wcs.celestial), **dc_contour_params)
            # Compute the correctec WCS
            correct_wcs = update_wcs_coords(wcs=data_container.wcs.celestial,
                                            ra_dec_offset=(
                                                results['offset_min'][0],
                                                results['offset_min'][1]))
            cs = axs[1].contour(synt_sb, transform=axs[1].get_transform(correct_wcs),
                           **dc_contour_params)
        elif isinstance(data_container, RSS):
            # TODO: Temporal until the final rss plotting method is in place
            mappable = axs[0].scatter(data_container.info['fib_ra'],
                                      data_container.info['fib_dec'],
                                      c=synt_sb,
                                      transform=axs[0].get_transform("world"))
            #plt.colorbar(mappable, ax=axs[0])
            mappable = axs[1].scatter(
                data_container.info['fib_ra'] + results['offset_min'][0],
                data_container.info['fib_dec'] + results['offset_min'][1],
                c=synt_sb,
                transform=axs[1].get_transform("world"))
            plt.colorbar(mappable, ax=axs[1], label='Aperture magnitude')
        plt.close(fig)
        return fig
