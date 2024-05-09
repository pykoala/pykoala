# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.centroids import centroid_2dg, centroid_com
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala.exceptions.exceptions import ClassError, FitError, CalibrationError
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.cubing import Cube
from pykoala.ancillary import centre_of_mass, make_dummy_cube_from_rss, interpolate_image_nonfinite
from pykoala.plotting.qc_plot import qc_registration_centroids

class AstrometryCorrection(CorrectionBase):
    """Perform astrometry-related corrections on DataContainers
    
    Description
    -----------
    This class applies astrometry corrections to input DataContainers based
    on different kind of quantities.
    -   RSS offset correction
        Corrects the positions of all fibres using an offset (fib_ra = fib_ra + offset)
    -   Cube offset correction:
        Update the WCS central value according to a new pixel mapping.
    """
    name = 'Astrometry'
    verbose = True

    def __init__(self, *args, **kwargs) -> None:
        self.verbose = kwargs.get('verbose', True)

    def register_centroids(self, data_set, object_name=None, qc_plot=False, **centroid_args):
        """Register a collection of DataContainers.

        Descriptoin
        -----------
        The registration is performed by means of the function
        ``pykoala.corrections.astrometry.find_centroid_in_dc``. The first DataContainer
        will be used as reference, and the offset position of the star will be computed
        in the rest of DataContainers.

        Parameters
        ----------
        data_set: (list)
            Collection of DataContainers.
        object_name: (str, default=None)
            Database querable name for absolute astrometry calibration.
        qc_plot: (bool, default=False)
            If true, a QC plot will be generated by calling
            ``pykoala.plotting.qc_plot.qc_registration_centroids``.
        **centroid_args:
            Extra arguments passed to the `pykoala.corrections.astrometry.find_centroid_in_dc` function.
        
        Returns
        -------
        - offsets: (list)
            A list containing the offsets for each input DataContainer.
        - fig: (matplotlib.pyplot.Figure)
            If `qc_plot=True`, it returns a quality control plot. `None` otherwise.
        """
        if len(data_set) <= 1:
            raise ArithmeticError("Input number of sources must be larger than 1")

        if object_name is not None:
            reference_pos = SkyCoord.from_name(object_name)
        else:
            centroid_args['full_output'] = False
            reference_pos = find_centroid_in_dc(data_set[0], **centroid_args)
        if qc_plot:
            centroid_args['full_output'] = True
            images_list = []
            wcs_list = []
            centroid_list = []

        self.corr_print("Reference star position: ", reference_pos)
        offsets = []
        for data in data_set:
            if qc_plot:
                cube, image, _, centroid_world = find_centroid_in_dc(data, **centroid_args)
                images_list.append(image)
                wcs_list.append(cube.wcs.celestial)
                centroid_list.append(centroid_world)                
            else:
                centroid_world = find_centroid_in_dc(data, **centroid_args)

            offset = [reference_pos.ra - centroid_world.ra, reference_pos.dec - centroid_world.dec]
            offsets.append(offset)
            self.corr_print("Offset found (ra, dec): ", offset[0].to('arcsec'), offset[1].to('arcsec'),
                            " (arcsec)")
        if qc_plot:
            fig = qc_registration_centroids(images_list, wcs_list,
                                            offsets, reference_pos)
            return offsets, fig
        else:
            return offsets, None

    def register_crosscorr(self, data_set, wave_range=None,
                           oversample=100, quick_cube_pix_size=0.3,
                           qc_plot=False):
        """Register a collection of DataContainers.

        """
        self.corr_print("Performing image cross-correlation")
        images = []
        wcs = []
        offsets = [[0 * u.deg, 0 * u.deg]]

        for data_container in data_set:
            if data_container.__class__ is RSS:
                cube = make_dummy_cube_from_rss(data_container, quick_cube_pix_size)
                image = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
            elif data_container.__class__ is Cube:
                cube = data_container
                image = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
                image /= np.nansum(image)

            image = interpolate_image_nonfinite(image)
            images.append(image)
            wcs.append(cube.wcs.celestial)
        results = cross_correlate_images(images, oversample=oversample)
        for i in range(len(results)):
            pixels_shift = results[i][0]
            moving_origin = wcs[i + 1].pixel_to_world(0, 0)
            # TODO: check axis consistency
            reference_origin = wcs[0].pixel_to_world(0 - pixels_shift[1], 0 - pixels_shift[0])
            offsets.append([moving_origin.ra - reference_origin.ra,
                            moving_origin.dec - reference_origin.dec])
        if qc_plot:
            fig = qc_registration_centroids(images, wcs,
                                            offsets,
                                            ref_pos=wcs[i + 1].pixel_to_world(images[0].shape[1] / 2,
                                                                                    images[0].shape[0] / 2))
            return offsets, fig
        else:
            return offsets, None
    
    def apply(self, data_container, offset):
        if data_container.__class__ is RSS:
            self.corr_print("Applying correction to RSS")

            data_container.info['fib_ra'] += offset[0].to('deg').value
            data_container.info['fib_dec'] += offset[1].to('deg').value

            self.log_correction(data_container, status='applied',
                                offset=f"{offset[0].to('arcsec')}{offset[1].to('arcsec')}")
        elif data_container.__class__ is Cube:
            pass

# Registration methods
# - Find centroid
    
def find_centroid_in_dc(data_container, wave_range=None,
                        centroider='com', com_power=1.0,
                        quick_cube_pix_size=0.5, subbox=None,
                        full_output=False):
    """ TODO
    Find the position that corresponds to the centre of light in
    a DataContainer.

    Description
    -----------
    This function finds the centre of light on an image created from a
    DataContainer (RSS or Cube). The position is computed either by using
    the first moment or a gaussian kernel of some power of the light distribution.
    For RSS data, a previous step consists of building a new Cube.

    Parameters
    ----------
    - data_container: (DataContainer)
        Data to extract the centroid.
    - wave_range: (list, default=None)
        Wavelength range to be used to make the white image defined by the
        initial and final wavelengths.
    - centroider: (string, default='com')
        Keyword describing the centroider method to use. 'com' corresponds to
        the first moment of the light distribution, while 'gauss' uses a 2D gaussian
        kernel.
    - com_power: (float, default=1.0)
        Power of the light distribution to weight the centroid position. The centroider
        function will compute the position of ``intensity**com_power``.
    - quick_cube_pix_size: (float, default=0.3 arcsec)
        Only used when the DataContainer is an RSS. Pixel size in arcsec used to interpolate
        the RSS into a Cube.
    - subbox: (list, default=None)
        Subbox containing the array indices of the image to select in the form:
        [[row_in, row_end], [col_in, col_end]].
    Returns
    -------
    - centroid_world: (astropy.coordinates.SkyCoord)
        Sky position of the centroid.
    """
    if centroider == 'com':
        centroider = centroid_com
    elif centroider == 'gauss':
        centroider = centroid_2dg

    if subbox is None:
        subbox = [[None, None], [None, None]]

    if data_container.__class__ is RSS:
        print(
            "[Registration]  Data provided in RSS format --> creating a dummy datacube")
        cube = make_dummy_cube_from_rss(data_container, quick_cube_pix_size)
        image = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
    elif data_container.__class__ is Cube:
        cube = data_container
        image = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
        image /= np.nansum(image)

    # Select a subbox
    image = image[subbox[0][0]:subbox[0][1],
                  subbox[1][0]: subbox[1][1]]
    # Mask bad values
    image = interpolate_image_nonfinite(image)

    # Find centroid
    centroid_pixel  = np.array(centroider(image**com_power))
    centroid_world = cube.wcs.celestial.pixel_to_world(*centroid_pixel)
    if not full_output:
        return centroid_world
    else:
        return cube, image, centroid_pixel, centroid_world 

def cross_correlate_images(list_of_images, oversample=100):
    """Compute image cross-correlation shift.
    
    Description
    -----------
    Apply the skimage.registration.phase_cross_correlation method to find 
    the shift of a list of images with respect to the first one.

    """
    try:
        from skimage.registration import phase_cross_correlation
    except Exception:
        raise ImportError("For using the image crosscorrelation function you have to install scikit-image library")

    results = []
    for i in range(len(list_of_images) - 1):
        # The shift ordering is consistent with the input image shape
        shift, error, diffphase = phase_cross_correlation(
            list_of_images[0], list_of_images[i+1],
            upsample_factor=100)
        results.append([shift, error, diffphase])
    return results

# TODO: implement this method in the correction class
def register_interactive(data_set, quick_cube_pix_size=0.2, wave_range=None):
    """
    Fully manual registration via interactive plot
    """
    import matplotlib
    from matplotlib import pyplot as plt

    images = []

    for data in data_set:
        if type(data) is RSS:
            print(
                "[Registration]  Data provided in RSS format --> creating a dummy datacube")
            cube_size_arcsec = (
                data.info['fib_ra'].max(
                ) - data.info['fib_ra'].min(),
                data.info['fib_dec'].max()
                - data.info['fib_dec'].min())
            x, y = np.meshgrid(
                np.arange(- cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size),
                np.arange(- cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size))
            datacube = np.zeros((data.wavelength.size, *x.shape))
    
            # Interpolate RSS to a datacube
            datacube, _, _ = interpolate_rss(
                data, pixel_size_arcsec=quick_cube_pix_size,
                kernel_size_arcsec=1.,
                datacube=datacube)
            image = make_white_image_from_array(datacube, data.wavelength,
                                                wave_range=wave_range, s_clip=3.0)
            image_pix_size = quick_cube_pix_size

        elif type(data) is Cube:
            image = data.get_white_image(wave_range=wave_range, s_clip=3.0)
            image_pix_size = data.info['pixel_size_arcsec']

        # Mask NaN values
        image = interpolate_image_nonfinite(image)
        images.append(image)

    ### Start the interactive GUI ###
    # Close previus figures
    plt.close()
    plt.ion()
    centres = []

    def mouse_event(event):
        """..."""
        centres.append([event.xdata / 3600, event.ydata / 3600])
        print('x: {} and y: {}'.format(event.xdata, event.ydata))

    n_rss = len(data_set)

    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    for i in range(n_rss):
        ax = fig.add_subplot(1, n_rss, i + 1)
        ax.scatter(data_set[0].info["fib_ra"], data_set[0].info["fib_dec"],
                   c=np.nanmedian(data_set[0].intensity_corrected, axis=1))
    plt.show()
    plt.ioff()
    return centres

# Mr Krtxo \(ﾟ▽ﾟ)/