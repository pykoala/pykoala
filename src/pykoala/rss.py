# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from pykoala.data_container import RSS
    
# =============================================================================
# Combine RSS (e.g., flats, twilights)
# =============================================================================

def combine_rss(list_of_rss, combine_method='nansum'):
    """Combine an input list of DataContainers into a new DataContainer."""

    all_intensities = []
    all_variances = []
    for rss in list_of_rss:
        intensity = rss.intensity.copy()
        variance = rss.variance.copy()
        # Ensure nans
        finite_mask = np.isfinite(intensity) & np.isfinite(variance)
        intensity[~finite_mask] = np.nan
        variance[~finite_mask] = np.nan

        all_intensities.append(intensity)
        all_variances.append(variance)

    if hasattr(np, combine_method):
        combine_function = getattr(np, combine_method)
        new_intensity = combine_function(all_intensities, axis=0)
        new_variance = combine_function(all_variances, axis=0)
    else:
        raise NotImplementedError("Implement user-defined combining methods")
    # TODO: Update the metadata as well
    new_rss = RSS(intensity=new_intensity, variance=new_variance,
                  wavelength=rss.wavelength, history=rss.history, info=rss.info)
    return new_rss


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
