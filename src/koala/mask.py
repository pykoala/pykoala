# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy

# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits

# =============================================================================
# KOALA packages
# =============================================================================
from koala.ancillary import vprint


# =============================================================================
# Mask a value in a rss (ancillary function)
# =============================================================================

def mask_section(rss, 
                 lower,
                 upper,
                 mask_value = 'nan',
                 verbose = True,
                 ):

    """
    This function mask values "mask_value" in a sectio of a RSS between "lower" 
    and "upper" columns. Returns a boolean mask. 
    """
    # Set print verbose
    vprint.verbose = verbose

    vprint("\n> Maskin pixel with value =",mask_value )
    
    
# =============================================================================
# Copy input RSS for storage the changes implemented in the task   
# =============================================================================
    mask = np.zeros_like(rss.intensity)
        
    section = rss.intensity[:,lower:upper]
    
    if mask_value == 'nan': 
        mask[:,lower:upper] = np.isnan(section)
        
    else:        
        mask[:,lower:upper] =  (section == mask_value)


# =============================================================================
# Return new RSS object
# =============================================================================
    return np.array(mask,dtype=bool)






# =============================================================================
# Mask from file
# =============================================================================
class MaskFromFile():
    """
    This task reads a fits file containing a full mask and save it as mask.
    Note that this mask is an IMAGE,
    the default values for mask,following the tast 'get_mask' below, are two vectors
    with the left -mask[0]- and right -mask[1]- valid pixels of the RSS.
    This takes more memory & time to process.

    Parameters
    ----------
    mask :  array[float]
        a mask is read from this Python object instead of reading the fits file
    mask_file : string
        fits file with the mask
    no_nans : boolean
        If False, NaN values are used, otherwise, if True, NaN's will be replaced with zero values
    verbose : boolean (default = True)
        Print results
    plot: boolean (default = True)
        Plot the mask
    include_history  boolean (default = False)
        Include the basic information in the rss.history
    """


    def __init__(self,
                 mask_path, 
                 verbose = True
                 ):
        # Set print verbose
        vprint.verbose = verbose

        # Read mask
        vprint("\n> Reading the mask from fits file : ")
        vprint(" ", mask_path)
        readed_mask = fits.open(mask_path)[0].data
        
        # 0 values means unmasked pixels. Any oher value is converten in 1 wich 
        # means a masked pixel. The value 1 corresponds to a masked value using
        # frokm_file_method
        
        self.mask = np.bool_(readed_mask)*2**0 # This mask has index 0 (2^0, i.e. maks value = 1)


    
    def apply(self,
             rss,
             verbose = True,
             ):
        

        # Set print verbose
        vprint.verbose = verbose

        # Copy of the input RSS for containing the changes implemented by the task   
        rss_out = copy.deepcopy(rss)
        
        rss_out.mask += self.mask    
        rss_out.log['mask from file']['comment']="- Mask obtainted using the RSS file "#", valid range of data:"+"\n"+ str(rss_out.wavelength[mask_max]) + " to " + str(rss_out.wavelength[mask_min]) + ",  in pixels = [ " + str(mask_max) + " , " + str(mask_min) + " ]"

# =============================================================================
# Retur new RSS object
# =============================================================================
        return rss_out
    

"""
log 
     
    if include_history:
        rss_out.history.append("- Mask read from fits file")
        rss_out.history.append("  " + mask_path)

    if include_history:
        rss_out.history.append("  Valid range of data using the mask:")
        rss_out.history.append(
            "  " + np.str(w[mask_max]) + " to " + np.str(w[mask_min]) + ",  in pixels = [ " + np.str(
                mask_max) + " , " + np.str(mask_min) + " ]")

"""



