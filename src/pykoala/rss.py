# =============================================================================
# Basics packages
# =============================================================================
import numpy as np

from pykoala import vprint
from pykoala.data_container import RSS

# =============================================================================
# Combine RSS (e.g., flats, twilights)
# =============================================================================

def combine_rss(list_of_rss, combine_method='nansum'):
    """Combine an input list of DataContainers into a new DataContainer.
    
    This method combines the intensity and variance of an input list of RSS
    frames.

    Parameters
    ----------
    list_of_rss : list
        A list containing the input RSS
    
    Returns
    -------
    stacked_rss : :class:`RSS`
        The resulting RSS from combining all input intensities and variances.
    """
    vprint("Combining input list of RSS data")
    all_intensities = []
    all_variances = []
    finite_masks = []

    for rss in list_of_rss:
        intensity = rss.intensity.copy()
        variance = rss.variance.copy()

        # Ensure nans
        finite_mask = np.isfinite(intensity) & np.isfinite(variance)
        finite_masks.append(finite_mask)

        intensity[~finite_mask] = np.nan
        variance[~finite_mask] = np.nan

        all_intensities.append(intensity)
        all_variances.append(variance)

    if hasattr(np, combine_method):
        combine_function = getattr(np, combine_method)
        new_intensity = combine_function(all_intensities, axis=0
                                         ) << rss.intensity.unit
        new_variance = combine_function(all_variances, axis=0
                                        ) << rss.variance.unit
    else:
        raise NotImplementedError("Implement user-defined combining methods")
    # TODO: Update the metadata as well
    new_rss = RSS(intensity=new_intensity, variance=new_variance,
                  wavelength=rss.wavelength, history=rss.history, info=rss.info,
                  fibre_diameter=rss.fibre_diameter, sky_fibres=rss.sky_fibres)
    return new_rss


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
