# =============================================================================
# Basics packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.wcs import WCS
from astropy.io import fits
# =============================================================================
# 1. Reading the data
# =============================================================================
from modular_koala.koala_ifu import koala_rss
# =============================================================================
# 5. Applying throughput
# =============================================================================
from modular_koala.throughput import get_from_sky_flat
from modular_koala.throughput import apply_throughput
# =============================================================================
# 6. Correcting for extinction
# =============================================================================
from modular_koala.extinction import extinction
# This function compute the extinction from the KOALA headers
from modular_koala.ancillary import airmass_from_header
# =============================================================================
# 7. Telluric correction (only for red data) U
# =============================================================================
from modular_koala.sky import Tellurics
# =============================================================================
# Sky substraction
# =============================================================================
from modular_koala.sky import SkyFromObject, uves_sky_lines


file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20025red.fits'
std_rss = koala_rss(file_path)

# combined_skyflat_red --------------------------------------------------------
file_path = 'modular_koala/input_data/sample_RSS/385R/combined_skyflat_red.fits'
skyflat_rss = koala_rss(file_path)

# Object: Tol30 ---------------------------------------------------------------
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20031red.fits'
tol30_rss = koala_rss(file_path)

# Object: He2_100 -------------------------------------------------------------
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20032red.fits'
he2_100_rss = koala_rss(file_path)


throughput_2D = get_from_sky_flat(skyflat_rss)

# We apply the throughput to the std and the science rss


# From 2D array
std_throughput = apply_throughput(std_rss,
                                  throughput=throughput_2D,
                                  verbose=True)
tol30_throughput = apply_throughput(tol30_rss,
                                    throughput=throughput_2D, verbose=True)

# std
airmass = airmass_from_header(std_throughput.header)
std_extinction = extinction(std_throughput,
                            airmass=airmass, verbose=True, plot=True)

airmass = airmass_from_header(tol30_throughput.header)
tol30_extinction = extinction(tol30_throughput, airmass=airmass,
                              verbose=True, plot=True)

# This class compute the telluric correction from a std star. Once computed we 
# can apply this solution to RSS or save the correction to a file.
telluric_correction = Tellurics(std_extinction)
smooth_corr = telluric_correction.telluric_from_smoothed_spec(exclude_wlm=[
                        [6245, 6390], [6450, 6750], [6840, 7000],
                        [7140, 7400], [7550, 7720], [8050, 8450]],
                        weight_fit_median=1,
                        wave_min=6200, plot=False)
# model_corr = telluric_correction.telluric_from_model(width=15, plot=True)

s_0 = np.nansum(std_rss.intensity_corrected, axis=0)
s_1 = np.nansum(std_throughput.intensity_corrected, axis=0)
s_2 = np.nansum(std_extinction.intensity_corrected, axis=0)
s_2 = np.nanmax(std_extinction.intensity_corrected, axis=0)
s_2_var = np.nansum(std_extinction.variance_corrected, axis=0)

w = std_extinction.wavelength
corr = telluric_correction.telluric_correction


obs_norm = np.mean(np.interp([6600, 6700], w, s_2))

width = 30
plt.figure(figsize=(12, 5))
plt.plot(w, s_2, c='k')
plt.plot(w, s_2 * corr, c='darkgreen', alpha=0.5)

tol30_telluric = telluric_correction.apply(tol30_extinction)

# %%
# ------------------------------------------------------------------------------
i = 1000
intensity = tol30_telluric.intensity[:, i]
p16, p50, p84 = np.nanpercentile(intensity, [16, 50, 84])

plt.hist(intensity, bins='auto', cumulative=True)
plt.axvline(p16, c='k')
plt.axvline(p50, c='b')
plt.axvline(2*p50-p16, c='r')
plt.axvline(p84, c='k')
plt.xscale('log')

# %%

tol30_telluric.intensity_corrected[~np.isfinite(tol30_telluric.intensity_corrected)] = np.nan
tol30_telluric.intensity_corrected.clip(
    min=np.nanpercentile(tol30_telluric.intensity_corrected, 99.99))

sky_model = SkyFromObject(tol30_telluric)
sky_model.load_sky_lines(lines_pct=1)

collapsed = np.nansum(tol30_telluric.intensity_corrected, axis=1)
sort_pos = np.argsort(collapsed)
median_pos = sort_pos[sort_pos.size//2]
sky_fibre = tol30_telluric.intensity_corrected[median_pos]
sky_fibre_unc = tol30_telluric.intensity[median_pos]
pct = sky_model.estimate_sky()
sky_estimate = pct[1]
sky_estimate_err = pct[1] - pct[0]

sky_model.fit_continuum(sky_estimate, err=sky_estimate_err, deg=3)

sky_emission_model, sky_emission_lines = sky_model.fit_emission_lines(
    sky_estimate - sky_model.sky_cont,
    errors=sky_estimate_err,
    maxiter=10000)

model_params = sky_emission_model.parameters.reshape(
    (sky_model.sky_lines.size + 1, 3))[1:, :]
median_sky_f = np.mean(sky_model.sky_lines_f)

total_sky_spec = sky_model.sky_cont + sky_emission_lines
# %%
residuals = (sky_estimate - total_sky_spec)**2 / sky_estimate_err**2
pct_res = np.percentile(residuals, [5, 16, 50, 84, 95])


plt.figure(figsize=(12, 4))
plt.subplot(211)
plt.plot(sky_model.rss.wavelength, sky_estimate, c='k', ls='-',
         label='Sky estimate')
plt.fill_between(sky_model.rss.wavelength, sky_estimate - sky_estimate_err,
                 sky_estimate + sky_estimate_err, alpha=0.3, color='k')
plt.plot(sky_model.rss.wavelength, sky_model.sky_cont, c='lime',
         label='Sky continuum model')
# plt.plot(sky_model.rss.wavelength, pct[1] - sky_model.sky_cont, c='orange')
plt.plot(sky_model.rss.wavelength, total_sky_spec, c='r', label='Sky model')
for model_i in model_params:
    plt.axvline(model_i[1], color='fuchsia', linewidth=0.8, zorder=-1, alpha=0.3)
    # plt.plot(sky_model.rss.wavelength, g(sky_model.rss.wavelength,
    #                                      *model_i))
# plt.xlim(6500, 6700)
plt.ylim(100, 1000)
plt.yscale('log')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=3)
plt.subplot(212)
plt.plot(tol30_rss.wavelength, residuals, c='k', lw=0.7)
plt.axhline(pct_res[2], color='Gray', label='P50')
plt.axhline(pct_res[1], ls='--', color='Gray', label='P16')
plt.axhline(pct_res[3], ls='--', color='Gray', label='P84')
plt.yscale('log')
plt.ylim(pct_res[0], pct_res[-1])
plt.ylabel(r'$\chi^2$')
plt.legend(loc='center', bbox_to_anchor=(0.5, -0.3), ncol=3)