def get_mask(self, mask=None, mask_path=None, no_nans=True, plot=True,
             verbose=True, include_history=False):
    # TODO: Merge into a single mask method
    """
    This task reads a fits file containing a full mask and save it as self.mask.
    Note that this mask is an IMAGE, 
    the default values for self.mask, following the task 'get_mask' below, are two vectors
    with the left -self.mask[0]- and right -self.mask[1]- valid pixels of the RSS.
    This takes more memory & time to process.

    Parameters
    ----------
    mask :  array[float]
        a mask is read from this Python object instead of reading the fits file
    mask_path : string
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
    # Read mask from file
    if mask_path not None:
        print("\n> Reading the mask from fits file : ")
        print(" ", mask_file)
        with fits.open(mask_file) as ftf:
            self.mask = ftf[0].data
        if include_history:
            self.history.append("- Mask read from fits file")
            self.history.append("  " + mask_file)
    # Mask provided by the user
    elif mask not None:
        print("\n> Using mask provided by the user")
        self.mask = mask
        if include_history:
            self.history.append("- Using mask provided by the user")
    # Compute mask using RSS data
    else:
        self.mask = np.isfinite(self.intensity)
        if include_history:
            self.history.append("- Mask read using a Python variable")
    # Check edges
    suma_good_pixels = np.nansum(self.mask, axis=0)
    nspec = self.n_spectra
    w = self.wavelength
    # Left edge
    found = 0
    j = 0
    if verbose: print("\n- Checking the left edge of the ccd...")
    while found < 1:
        if suma_good_pixels[j] == nspec:
            first_good_pixel = j
            found = 2
        else:
            j = j + 1
    if verbose: print("  First good pixels is ", first_good_pixel, ", that corresponds to ", w[first_good_pixel],
                      "A")
    if plot:
        ptitle = "Left edge of the mask, valid minimun wavelength = " + np.str(
            np.round(w[first_good_pixel], 2)) + " , that is  w [ " + np.str(first_good_pixel) + " ]"
        plot_plot(w, np.nansum(self.mask, axis=0), ymax=1000, ymin=suma_good_pixels[0] - 10,
                  xmax=w[first_good_pixel * 3], vlines=[w[first_good_pixel]],
                  hlines=[nspec], ptitle=ptitle, ylabel="Sum of good fibres")

    mask_first_good_value_per_fibre = []
    for fibre in range(len(self.mask)):
        found = 0
        j = 0
        while found < 1:
            if no_nans:
                if self.mask[fibre][j] == 0:
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            else:
                if np.isnan(self.mask[fibre][j]):
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2

    mask_max = np.nanmax(mask_first_good_value_per_fibre)
    if plot: plot_plot(np.arange(nspec), mask_first_good_value_per_fibre, ymax=mask_max + 1,
                       hlines=[mask_max], xlabel="Fibre", ylabel="First good pixel in mask",
                       ptitle="Left edge of the mask")

    # Right edge, important for RED 
    if verbose: print("\n- Checking the right edge of the ccd...")
    mask_last_good_value_per_fibre = []
    mask_list_fibres_all_good_values = []

    for fibre in range(len(self.mask)):
        found = 0
        j = len(self.mask[0]) - 1
        while found < 1:
            if no_nans:
                if self.mask[fibre][j] == 0:
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(self.mask[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2
            else:
                if np.isnan(self.mask[fibre][j]):
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(self.mask[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2

    mask_min = np.nanmin(mask_last_good_value_per_fibre)
    if plot:
        ptitle = "Fibres with all good values in the right edge of the mask : " + np.str(
            len(mask_list_fibres_all_good_values))
        plot_plot(np.arange(nspec), mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in mask", ptitle=ptitle)
    if verbose: print("  Minimun value of good pixel =", mask_min, " that corresponds to ", w[mask_min])
    if verbose: print("\n  --> The valid range for these data is", np.round(w[mask_max], 2), " to ",
                      np.round(w[mask_min], 2), ",  in pixels = [", mask_max, " , ", mask_min, "]")

    self.mask_good_index_range = [mask_max, mask_min]
    self.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
    self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

    if verbose:
        print("\n> Mask stored in self.mask !")
        print("  Valid range of the data stored in self.mask_good_index_range (index)")
        print("                             and in self.mask_good_wavelength  (wavelenghts)")
        print("  List of fibres with all good values in self.mask_list_fibres_all_good_values")

    if include_history:
        self.history.append("  Valid range of data using the mask:")
        self.history.append("  " + np.str(w[mask_max]) + " to " + np.str(w[mask_min]) + ",  in pixels = [ " + np.str(
            mask_max) + " , " + np.str(mask_min) + " ]")
    # -----------------------------------------------------------------------------
    
def read_mask_from_fits_file(self, mask=[[]], mask_file="", no_nans=True, plot=True,
                             verbose=True, include_history=False):
    # TODO: Merge into a single mask method
    """
    This task reads a fits file containing a full mask and save it as self.mask.
    Note that this mask is an IMAGE,
    the default values for self.mask, following the task 'get_mask' below, are two vectors
    with the left -self.mask[0]- and right -self.mask[1]- valid pixels of the RSS.
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
    # Read mask
    if mask_file != "":
        print("\n> Reading the mask from fits file : ")
        print(" ", mask_file)
        ftf = fits.open(mask_file)
        self.mask = ftf[0].data
        if include_history:
            self.history.append("- Mask read from fits file")
            self.history.append("  " + mask_file)
    else:
        print("\n> Reading the mask stored in Python variable...")
        self.mask = mask
        if include_history: self.history.append("- Mask read using a Python variable")
    if no_nans:
        print("  We are considering that the mask does not have 'nans' but 0s in the bad pixels")
    else:
        print("  We are considering that the mask DOES have 'nans' in the bad pixels")

    # Check edges
    suma_good_pixels = np.nansum(self.mask, axis=0)
    nspec = self.n_spectra
    w = self.wavelength
    # Left edge
    found = 0
    j = 0
    if verbose: print("\n- Checking the left edge of the ccd...")
    while found < 1:
        if suma_good_pixels[j] == nspec:
            first_good_pixel = j
            found = 2
        else:
            j = j + 1
    if verbose: print("  First good pixels is ", first_good_pixel, ", that corresponds to ", w[first_good_pixel], "A")

    if plot:
        ptitle = "Left edge of the mask, valid minimun wavelength = " + np.str(
            np.round(w[first_good_pixel], 2)) + " , that is  w [ " + np.str(first_good_pixel) + " ]"
        plot_plot(w, np.nansum(self.mask, axis=0), ymax=1000, ymin=suma_good_pixels[0] - 10,
                  xmax=w[first_good_pixel * 3], vlines=[w[first_good_pixel]],
                  hlines=[nspec], ptitle=ptitle, ylabel="Sum of good fibres")

    mask_first_good_value_per_fibre = []
    for fibre in range(len(self.mask)):
        found = 0
        j = 0
        while found < 1:
            if no_nans:
                if self.mask[fibre][j] == 0:
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            else:
                if np.isnan(self.mask[fibre][j]):
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2

    mask_max = np.nanmax(mask_first_good_value_per_fibre)
    if plot: plot_plot(np.arange(nspec), mask_first_good_value_per_fibre, ymax=mask_max + 1,
                       hlines=[mask_max], xlabel="Fibre", ylabel="First good pixel in mask",
                       ptitle="Left edge of the mask")

    # Right edge, important for RED
    if verbose: print("\n- Checking the right edge of the ccd...")
    mask_last_good_value_per_fibre = []
    mask_list_fibres_all_good_values = []

    for fibre in range(len(self.mask)):
        found = 0
        j = len(self.mask[0]) - 1
        while found < 1:
            if no_nans:
                if self.mask[fibre][j] == 0:
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(self.mask[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2
            else:
                if np.isnan(self.mask[fibre][j]):
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(self.mask[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2

    mask_min = np.nanmin(mask_last_good_value_per_fibre)
    if plot:
        ptitle = "Fibres with all good values in the right edge of the mask : " + np.str(
            len(mask_list_fibres_all_good_values))
        plot_plot(np.arange(nspec), mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in mask", ptitle=ptitle)
    if verbose: print("  Minimun value of good pixel =", mask_min, " that corresponds to ", w[mask_min])
    if verbose: print("\n  --> The valid range for these data is", np.round(w[mask_max], 2), " to ",
                      np.round(w[mask_min], 2), ",  in pixels = [", mask_max, " , ", mask_min, "]")

    self.mask_good_index_range = [mask_max, mask_min]
    self.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
    self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

    if verbose:
        print("\n> Mask stored in self.mask !")
        print("  Valid range of the data stored in self.mask_good_index_range (index)")
        print("                             and in self.mask_good_wavelength  (wavelenghts)")
        print("  List of fibres with all good values in self.mask_list_fibres_all_good_values")

    if include_history:
        self.history.append("  Valid range of data using the mask:")
        self.history.append("  " + np.str(w[mask_max]) + " to " + np.str(w[mask_min]) + ",  in pixels = [ " + np.str(
            mask_max) + " , " + np.str(mask_min) + " ]")
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def get_mask(self, verbose=True, plot=True, include_history=False):
    """
    Task for getting the mask using the very same RSS file.
    This assumes that the RSS does not have nans or 0 as consequence of cosmics
    in the edges.
    The task is run once at the very beginning, before applying flat or throughput.
    It provides the self.mask data

    Parameters
    ----------
    no_nans : boolean
        If False, NaN values are used, otherwise, if True, NaN's will be replaced with zero values
    verbose : boolean (default = True)
        Print results
    plot: boolean (default = True)
        Plot the mask
    include_history  boolean (default = False)
        Include the basic information in the rss.history
    """
    if verbose: print("\n> Getting the mask using the good pixels of this RSS file ...")

    #  Check if file has 0 or nans in edges
    if np.isnan(self.intensity[0][-1]):
        no_nans = False
    else:
        no_nans = True
        if self.intensity[0][-1] != 0:
            print(
                "  Careful!!! pixel [0][-1], fibre = 0, wave = -1, that should be in the mask has a value that is not nan or 0 !!!!!")

    w = self.wavelength
    x = list(range(self.n_spectra))

    if verbose and plot: print("\n- Checking the left edge of the ccd...")
    mask_first_good_value_per_fibre = []
    for fibre in range(self.n_spectra):
        found = 0
        j = 0
        while found < 1:
            if no_nans:
                if self.intensity[fibre][j] == 0:
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            else:
                if np.isnan(self.intensity[fibre][j]):
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            if j > 101:
                print(" No nan or 0 found in the fist 100 pixels, ", w[j], " for fibre", fibre)
                mask_first_good_value_per_fibre.append(j)
                found = 2

    mask_max = np.nanmax(mask_first_good_value_per_fibre)
    if plot:
        plot_plot(x, mask_first_good_value_per_fibre, ymax=mask_max + 1, xlabel="Fibre",
                  ptitle="Left edge of the RSS", hlines=[mask_max], ylabel="First good pixel in RSS")

    # Right edge, important for RED
    if verbose and plot: print("\n- Checking the right edge of the ccd...")
    mask_last_good_value_per_fibre = []
    mask_list_fibres_all_good_values = []

    for fibre in range(self.n_spectra):
        found = 0
        j = self.n_wave - 1
        while found < 1:
            if no_nans:
                if self.intensity[fibre][j] == 0:
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == self.n_wave - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2
            else:
                if np.isnan(self.intensity[fibre][j]):
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(self.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2

            if j < self.n_wave - 1 - 300:
                print(" No nan or 0 found in the last 300 pixels, ", w[j], " for fibre", fibre)
                mask_last_good_value_per_fibre.append(j)
                found = 2

    mask_min = np.nanmin(mask_last_good_value_per_fibre)
    if plot:
        ptitle = "Fibres with all good values in the right edge of the RSS file : " + np.str(
            len(mask_list_fibres_all_good_values))
        plot_plot(x, mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in RSS", ptitle=ptitle)

    if verbose: print(
        "\n  --> The valid range for this RSS is {:.2f} to {:.2f} ,  in pixels = [ {} ,{} ]".format(w[mask_max],
                                                                                                    w[mask_min],
                                                                                                    mask_max, mask_min))

    self.mask = [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre]
    self.mask_good_index_range = [mask_max, mask_min]
    self.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
    self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

    if verbose:
        print("\n> Mask stored in self.mask !")
        print("  self.mask[0] contains the left edge, self.mask[1] the right edge")
        print("  Valid range of the data stored in self.mask_good_index_range (index)")
        print("                             and in self.mask_good_wavelength  (wavelenghts)")
        print("  Fibres with all good values (in right edge) in self.mask_list_fibres_all_good_values")

    if include_history:
        self.history.append("- Mask obtainted using the RSS file, valid range of data:")
        self.history.append("  " + np.str(w[mask_max]) + " to " + np.str(w[mask_min]) + ",  in pixels = [ " + np.str(
            mask_max) + " , " + np.str(mask_min) + " ]")
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def apply_mask(self, mask_from_file=False, make_nans=False,
               replace_nans=False, verbose=True):
    """
    Apply a mask to a RSS file.

    Parameters
    ----------
    mask_from_file : boolean (default = False)
        If a full mask (=image) has previously been created and stored in self.mask, apply this mask.
        Otherwise, self.mask should has two lists = [ list_of_first_good_fibres, list_of_last_good_fibres ].
    make_nans : boolean
        If True, apply the mask making nan all bad pixels
    replace_nans : boolean
        If False, NaN values are used, otherwise, if True, NaN's will be replaced with zero values
    verbose : boolean (default = True)
        Print results
    """
    if mask_from_file:
        self.intensity_corrected = self.intensity_corrected * self.mask
    else:
        for fibre in range(self.n_spectra):
            # Apply left part
            for i in range(self.mask[0][fibre]):
                if make_nans:
                    self.intensity_corrected[fibre][i] = np.nan
                else:
                    self.intensity_corrected[fibre][i] = 0
            # now right part
            for i in range(self.mask[1][fibre] + 1, self.n_wave):
                if make_nans:
                    self.intensity_corrected[fibre][i] = np.nan
                else:
                    self.intensity_corrected[fibre][i] = 0
    if replace_nans:
        # Change nans to 0:
        for i in range(self.n_spectra):
            self.intensity_corrected[i] = [0 if np.isnan(x) else x for x in self.intensity_corrected[i]]
        if verbose: print("\n> Mask applied to eliminate nans and make 0 all bad pixels")
    else:
        if verbose:
            if make_nans:
                print("\n> Mask applied to make nan all bad pixels")
            else:
                print("\n> Mask applied to make 0 all bad pixels")