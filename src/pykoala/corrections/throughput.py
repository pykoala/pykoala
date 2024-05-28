# =============================================================================
# Basics packages
# =============================================================================
from os import path
import numpy as np
import copy
from astropy.io import fits
from scipy.ndimage import median_filter
#from scipy.ndimage import gaussian_filter
# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala import ancillary
from pykoala.ancillary import vprint   #!!!
from pykoala.plotting.rss_plot import rss_image

class Throughput(object):
    def __init__(self, path=None, throughput_data=None, throughput_error=None):
        self.path = path
        self.throughput_data = throughput_data
        self.throughput_error = throughput_error

        if self.path is not None and self.throughput_data is None:
            self.load_fits()
    
    def tofits(self, output_path):
        primary = fits.PrimaryHDU()
        thr = fits.ImageHDU(data=self.throughput_data,
                            name='THROU')
        thr_err = fits.ImageHDU(data=self.throughput_error,
                            name='THROUERR')
        hdul = fits.HDUList([primary, thr, thr_err])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        print(f"[Throughput] Throughput saved at {output_path}")

    def load_fits(self):
        """Load the throughput data from a fits file.
        
        Description
        -----------
        Loads throughput values (extension 1) and associated errors (extension 2) from a fits
        file.
        """
        if not path.isfile(self.path):
            raise NameError(f"Throughput file {self.path} does not exists.")
        print(f"[Throughput] Loading throughput from {self.path}")
        with fits.open(self.path) as hdul:
            self.throughput_data = hdul[1].data
            self.throughput_error = hdul[2].data


class ThroughputCorrection(CorrectionBase):
    """
    Throughput correction class.

    This class accounts for the relative flux loss due to differences on the fibre efficiencies.

    Attributes
    ----------
    - name
    -
    """
    name = "ThroughputCorrection"
    throughput = None
    verbose = False

    def __init__(self, **kwargs):
        super().__init__()
        #print(isinstance(throughput, Throughput))
        self.throughput = kwargs.get('throughput', Throughput())
        #if type(self.throughput) is not Throughput():   #!!! ANGEL    Así me falla, tengo que llamar a str
        #if str(type(self.throughput)) != str(Throughput()):   ## TAMBIEN FALLA ????
        #    raise AttributeError("Input throughput must be an instance of Throughput class")

        self.throughput.path = kwargs.get('throughput_path', None)
        if self.throughput.throughput_data is None and self.throughput.path is not None:
            self.throughput.load_fits(self.throughput_path)
        

    @staticmethod
    def create_throughput_from_rss(rss_set, clear_nan=True,
                                   statistic='median',
                                   medfilt=None,
                                   **kwargs):                                           #!!! ANGEL 
        """Compute the throughput map from a set of flat exposures.

        Given a set of flat exposures, this method will estimate the average
        efficiency of each fibre.

        Parameters
        ----------
        - rss_set: (list)
            List of RSS data.
        - clean_nan: (bool, optional, default=True)
            If True, nan values will be replaced by a nearest neighbour interpolation.
        - statistic: (str, optional, default='median')
            Set to 'median' or 'mean' to compute the throughput function.
        - medfilt: (float, optional, default=None)
            If provided, apply a median filter to the throughput estimate.
        """
        
        if type(rss_set) == list:     #!!! ANGEL
            vprint('\n> Creating throughput from list of RSS Python objects provided using {}, clear_nan = {}, medfilt = {}:'.format(statistic,clear_nan,medfilt), **kwargs)
        else:
            rss_name = rss_set.info["name"]
            rss_set = [rss_set]
            vprint('\n> Creating throughput from RSS Python object {} using clear_nan = {}, medfilt = {}:'.format(rss_name,clear_nan,medfilt), **kwargs)
  
        
        if statistic == 'median':
            stat_func = np.nanmedian
        elif statistic == 'mean':
            stat_func = np.nanmean

        fluxes = []
        for rss in rss_set:
            #vprint('  Reading skyflat/domeflat RSS file {} , exptime = {}'.format(rss.info["path_to_file"],rss.info['exptime']), **kwargs)    #!!! ANGEL
            vprint('  Reading skyflat/domeflat RSS file, exptime = {}'.format(rss.info['exptime']), **kwargs)    #!!! ANGEL
            f = rss.intensity / rss.info['exptime']
            fluxes.append(f)
        # Combine
        combined_flux = stat_func(fluxes, axis=0)
        combined_flux_err = np.nanstd(fluxes, axis=0) / np.sqrt(len(fluxes))

        # Normalize
        reference_fibre = stat_func(combined_flux, axis=0)
        throughput_data = combined_flux / reference_fibre[np.newaxis, :]

        # Estimate the error
        # throughput_error = np.nanstd(np.array(fluxes) / stat_func(fluxes, axis=1)[:, np.newaxis, :], axis=0)
        throughput_error = combined_flux_err / reference_fibre[np.newaxis, :]
        
        if clear_nan:
            print("  Applying nearest neighbour interpolation to remove NaN values ...")
            throughput_data = ancillary.interpolate_image_nonfinite(
                throughput_data)
            throughput_error = ancillary.interpolate_image_nonfinite(
                throughput_error**2)**0.5
        if medfilt is not None:
            print(f"  Applying median filter (size = {medfilt}) ...")
            throughput_data = median_filter(throughput_data, size=medfilt)
            throughput_error = median_filter(
                throughput_error**2, size=medfilt)**0.5

        
        throughput = Throughput(throughput_data=throughput_data,
                                throughput_error=throughput_error)
        
        if kwargs.get('plot'):                                                                      #!!! ANGEL
            rss_image(rss, image=throughput.throughput_data, greyscale = True,  
                      title = "Normalized Throughput",            
                      colorbar_label="Normalized intensity",**kwargs)
        
        return throughput

    def apply(self, rss, throughput=None, plot=True, **kwargs):                                 #!!! ANGEL
        """Apply a 2D throughput model to a RSS.

        Parameters
        ----------
        - throughput
        - rss: (RSS)
        - plot: (bool, optional, default=True)
        """
        
        if throughput is None and self.throughput is not None:
            throughput = self.throughput
        else:
            raise RuntimeError("Throughput not provided!")
        
        #print(isinstance(throughput, Throughput))
        #print(throughput)
        
        if str(type(throughput)) != str(Throughput):        #!!! ANGEL    Así me falla, tengo que llamar a str y usar != en lugar de is not
            raise AttributeError("Input throughput must be an instance of Throughput class")

        # if type(rss) is not RSS:                          #!!! ANGEL    Así me falla, tengo que llamar a str y usar != en lugar de is not
        if str(type(rss)) != str(RSS):
            raise ValueError("Throughput can only be applied to RSS data:\n input {}"
                             .format(type(rss)))
            
        # =============================================================================
        # Verbose if needed   
        # =============================================================================    
        vprint('> Applying throughput to rss object using Python Throughput object ...', **kwargs)    #!!! ANGEL
        # =============================================================================
        # Copy input RSS for storage the changes implemented in the task   
        # =============================================================================
        rss_out = copy.deepcopy(rss)

        rss_out.intensity = rss_out.intensity / throughput.throughput_data
        rss_out.variance = rss_out.variance / throughput.throughput_data**2
        self.log_correction(rss, status='applied')
        return rss_out
