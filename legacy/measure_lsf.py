from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from koala.koala_ifu import koala_rss

from koala.corrections.sky import SkyFromObject, SkyCorrection
from koala.corrections.sky import medfilt_continuum

rss = koala_rss("reduced_data/385R/27feb20031red.fits")

skymodel = SkyFromObject(rss)
# skymodel.remove_continuum(window_size=31)

skycorrection = SkyCorrection(skymodel)
rss_substracted, fig = skycorrection.apply(rss, plot=True)

plt.figure(fig)
# %%
fibre = 600
plt.figure()
plt.title("Sky")
plt.plot(rss.wavelength, rss.intensity_corrected[fibre], label='Input RSS')
plt.plot(skymodel.wavelength, skymodel.intensity, label='Background')
plt.plot(rss_substracted.wavelength, rss_substracted.intensity_corrected[fibre],
          label='Sky substracted RSS')
plt.legend()
# %%
plt.figure()
plt.subplot(121)
plt.imshow(rss.intensity_corrected, aspect='auto', interpolation='none',
           norm=LogNorm(vmin=1, vmax=1e5))
plt.subplot(122)
plt.imshow(rss_substracted.intensity_corrected, aspect='auto', interpolation='none',
           norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar()


# %%
for i in range(rss.intensity.shape[0]):
    i = 300
    spec = rss.intensity[i]
    continuum = medfilt_continuum(spec, window_size=31)
    emission_model, emission_spec = skymodel.fit_emission_lines(spec - continuum)
    break
plt.plot(skymodel.wavelength, emission_spec + continuum)
plt.xlim(8500, 8900)

# %%
plt.figure()
for i in range(emission_model.n_submodels):
    mean = getattr(emission_model, f"mean_{i}")
    stddev = getattr(emission_model, f"stddev_{i}")
    amp = getattr(emission_model, f"amplitude_{i}")
    plt.scatter(mean.value, stddev.value, c=amp.value, vmin=0, vmax=50)
plt.colorbar()