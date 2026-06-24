import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter
from astropy.io import fits
from photutils.segmentation import detect_sources, deblend_sources

hdul = fits.open("/home/pcorchoc/Research/obs_data/HI-KIDS/raw/20180227/ccd_1/27feb10006.fits")

p5, p95 = np.nanpercentile(hdul[0].data, [5, 95])

min_value = minimum_filter(hdul[0].data, size=5)
max_value = maximum_filter(hdul[0].data, size=5)

back_subtracted = hdul[0].data - min_value
p5, p95 = np.nanpercentile(back_subtracted, [5, 95])

# plt.figure()
#plt.imshow(hdul[0].data, vmin=p5, vmax=p95)
#plt.colorbar()
# plt.imshow(back_subtracted, vmin=p5, vmax=p95)
# plt.colorbar()
# plt.show()

real_n_fibres = 986
frac = 1.
n_pixels = int(max_value.shape[1] * 0.75)
mask = ~np.isfinite(hdul[0].data)
segment_image = detect_sources(hdul[0].data, threshold=min_value * frac, npixels=n_pixels,
                                mask=mask)

# segment_image = deblend_sources(hdul[0].data, segment_image, npixels=n_pixels)
print("N labels: ", segment_image.nlabels)
while segment_image.nlabels < real_n_fibres:
    print("N labels: ", segment_image.nlabels, frac)
    frac += 0.05
    segment_image = detect_sources(hdul[0].data, threshold=min_value * frac,
                                    npixels=n_pixels, mask=mask)
    
plt.figure()
plt.imshow(segment_image.data)
plt.colorbar()
plt.show()
