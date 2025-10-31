"""
Core data structures for PyKOALA.

This module defines the base classes that represent the core data structures
used during the reduction of Integral Field Spectroscopy (IFS) data in
:mod:`pykoala`.

At its core, PyKOALA models all sources of data (for example, images,
row-stacked spectra, and data cubes) as instances of :class:`DataContainer`,
a unified structure that standardizes how data and metadata are organized,
accessed, and manipulated.

A specialized subclass, :class:`SpectraContainer`, represents spectroscopic
data, adding a spectral coordinate (:attr:`~SpectraContainer.wavelength`) to
the common attributes (for example, :attr:`~DataContainer.intensity`,
:attr:`~DataContainer.variance`, and :attr:`~DataContainer.mask`). Concrete
implementations include :class:`RSS` for row-stacked spectra and
:class:`Cube` for 3D IFS datacubes.
"""


from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import copy
from datetime import datetime

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants

from pykoala import VerboseMixin, __version__
from pykoala import ancillary
from pykoala.plotting.utils import plot_image, new_figure, plot_fibres
# =============================================================================


class HistoryRecord(object):
    """Atomic log entry used by :class:`DataContainerHistory`.

    Parameters
    ----------
    title : str
        Short title summarizing the record.
    comments : str or list of str
        Comment lines. If a single string is provided, it will be split on
        newline characters into multiple lines.
    tag : str, optional
        Optional free-form tag to categorize the record.

    Attributes
    ----------
    title : str
        Record title.
    comments : list of str
        Comment lines associated with the record.
    tag : str or None
        Optional category/tag.

    Methods
    -------
    to_str(title=True)
        Return a string representation of the entry.

    Notes
    -----
    This class does not enforce any semantics on the content other than basic
    typing. Higher-level consistency is handled by :class:`DataContainerHistory`.
    """
    def __init__(self, title, comments, tag=None) -> None:
        self.title = title
        self.comments = comments
        self.tag = tag

    @property
    def comments(self):
        return self._comments

    @comments.setter
    def comments(self, comments):
        if isinstance(comments, list):
            self._comments = comments
        elif isinstance(comments, str):
            self._comments = comments.split("\n")
        else:
            raise NameError(
                "Input comments must be str or list of strings, not:"
                + f" {comments.__class__}"
            )

    def to_str(self, title=True):
        """Return a string representation of the record.

        Parameters
        ----------
        title : bool, default=True
            If ``True``, prefix the comments with the record title.

        Returns
        -------
        str
            A single newline-joined string for the record.
        """
        comments = "\n".join(self.comments)
        if title:
            comments = f"{self.title}: " + comments
        return comments


# =============================================================================


class DataContainerHistory(VerboseMixin):
    """History/log of data-reduction actions on a :class:`DataContainer`.

    This class stores an ordered sequence of :class:`HistoryRecord` entries
    and provides helpers to search, serialize, and deserialize histories.

    Parameters
    ----------
    list_of_entries : iterable of (:class:`HistoryRecord` or tuple), optional
        Initial entries to populate the history. Tuples must be either
        ``(title, comments)`` or ``(title, comments, tag)``.
    **kwargs
        Passed to :class:`~pykoala.VerboseMixin` (e.g., ``logger``, ``verbose``).

    Attributes
    ----------
    record_entries : list of :class:`HistoryRecord`
        The ordered list of log entries.
    tags : list of str
        Set of unique tags discovered in ``record_entries``.
    logger : str
        Logger name.
    verbose : bool
        Verbosity flag.
    """

    def __init__(self, list_of_entries=None, **kwargs):
        # Initialise the verbose logger
        self.logger = kwargs.get("logger", "pykoala")
        self.verbose = kwargs.get("verbose", True)
        self.record_entries = []
        self.tags = []
        self.verbose = kwargs.get("verbose", True)

        if list_of_entries is not None:
            self.initialise_record(list_of_entries)

    def initialise_record(self, list_of_entries):
        """Populate the history from an iterable of entries.

        Parameters
        ----------
        list_of_entries : iterable
            Iterable of :class:`HistoryRecord` or tuples. Valid tuple forms are
            ``(title, comments)`` or ``(title, comments, tag)``.

        Raises
        ------
        NameError
            If an element is neither a :class:`HistoryRecord` nor a valid tuple.
        """
        for record in list_of_entries:
            if isinstance(record, HistoryRecord):
                self.record_entries.append(record)
                continue
            elif isinstance(record, tuple) or isinstance(record, list):
                if len(record) == 2:
                    title, comments = record
                    tag = None
                elif len(record) == 3:
                    title, comments, tag = record
                else:
                    raise NameError(
                        "Input record must contain two (title, comments) or"
                        + " three (title, comments, tag) elements"
                    )
                record = HistoryRecord(title=title, comments=comments, tag=tag)
            else:
                raise NameError(f"Unrecognized input record of type {record.__class__}")
            self.record_entries.append(record)

    def log_record(self, title, comments, tag=None):
        """Append a new record to the history.

        Parameters
        ----------
        title : str
            Record title.
        comments : str or list of str
            Record comments. If a single string is provided, it is split by
            newline into multiple lines.
        tag : str, optional
            Optional category/tag. New tags are tracked in :attr:`tags`.
        """
        if tag is not None and tag not in self.tags:
            self.tags.append(tag)
        record = HistoryRecord(title=title, comments=comments, tag=tag)
        self.record_entries.append(record)

    def is_record(self, title, comment=None):
        """Check whether an entry exists.

        Parameters
        ----------
        title : str
            Title that must match exactly.
        comment : str, optional
            If provided, the entry must also contain this comment substring.

        Returns
        -------
        bool
            ``True`` if a matching entry is found, otherwise ``False``.
        """
        for record in self.record_entries:
            if record.title == title:
                if comment is not None:
                    if comment in record.comments:
                        return True
                else:
                    return True
        return False

    def find_record(self, title="", comment="", tag=""):
        """Return all records whose fields contain the given substrings.

        Parameters
        ----------
        title : str, default=''
            Substring to match against the record title.
        comment : str, default=''
            Substring to match against the comment text.
        tag : str, default=''
            Substring to match against the tag.

        Returns
        -------
        list of :class:`HistoryRecord`
            All matching entries (possibly empty).
        """
        return [
            record
            for record in self.record_entries
            if (title in record.title)
            and (comment in record.to_str(title=False))
            and (tag in str(record.tag))
        ]

    def dump_to_header(self, header=None):
        """Serialize the history into a FITS header.

        Each entry is written into a sequential ``PYKOALA{index}`` card whose
        **value** stores the comment text and whose **comment** stores the title.

        Parameters
        ----------
        header : astropy.io.fits.Header, optional
            Header to append to. If ``None``, a new header is created.

        Returns
        -------
        astropy.io.fits.Header
            The updated header.

        Notes
        -----
        Existing ``PYKOALA*`` cards in ``header`` are preserved; new entries are
        appended after them.
        """
        if header is None:
            header = fits.Header()
            index = 0
        else:
            index = len(self.get_entries_from_header(header))

        for record in self.record_entries:
            header["PYKOALA" + str(index)] = (record.to_str(title=False), record.title)
            index += 1
        return header

    def dump_to_text(self, file):
        """Write the history to a plain-text file.

        Parameters
        ----------
        file : str or path-like
            Output filename.

        """
        self.vprint("Writting log into text file")
        with open(file, "w") as f:
            for record in self.record_entries:
                f.write(record.to_str() + "\n")

    @classmethod
    def get_entries_from_header(cls, header):
        """Deserialize :class:`HistoryRecord` entries from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            Header containing ``PYKOALA*`` cards created by
            :meth:`dump_to_header`.

        Returns
        -------
        list of :class:`HistoryRecord`
            The reconstructed history entries.
        """
        list_of_entries = []

        for title, key in zip(header.comments["PYKOALA*"], header["PYKOALA*"]):
            list_of_entries.append(
                HistoryRecord(title=title, comments=header[key]))
        return list_of_entries

    @classmethod
    def from_header(cls, header, **kwargs):
        """Construct a :class:`DataContainerHistory` from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            Header previously written by :meth:`dump_to_header`.
        **kwargs
            Passed to the constructor (e.g., ``logger``, ``verbose``).

        Returns
        -------
        DataContainerHistory
            New instance populated with the deserialized entries.
        """
        list_of_entries = cls.get_entries_from_header(header)
        return cls(list_of_entries=list_of_entries, **kwargs)

    def show(self):
        for record in self.record_entries:
            print(record.to_str())

    def __call__(self, *args, **kwargs):
        self.log_record(*args, **kwargs)


# =============================================================================


# class Parameter(object):
#     """Class that represents some parameter and associated metadata"""

#     def __init__(self) -> None:
#         pass


# =============================================================================


class DataMask(object):
    """Bitmask container for pixel flags associated with a :class:`DataContainer`.

    Parameters
    ----------
    shape : tuple of int, optional
        Shape of the mask to initialize. Required if ``bitmask`` is not given.
    flag_map : dict[str, tuple[int, str]], optional
        Mapping from flag name to ``(bit_value, description)``. If omitted,
        a default map with a single ``"BAD"`` flag is used.
    bitmask : numpy.ndarray of int, optional
        Pre-existing integer bitmask. If supplied, ``shape`` is ignored and
        :attr:`masks` are derived from this array.

    Attributes
    ----------
    flag_map : dict[str, tuple[int, str]]
        Mapping of flag names to bit values and description.
    bitmask : numpy.ndarray of int
        Integer bitmask array (power-of-two composition).
    masks : dict[str, numpy.ndarray of bool]
        Boolean masks per flag name, same shape as :attr:`bitmask`.

    Notes
    -----
    A pixel is considered flagged for ``flag_name`` when
    ``bitmask & flag_map[flag_name][0] > 0``.
    """

    def __init__(self, shape=None, flag_map=None, bitmask=None):
        if flag_map is None:
            if bitmask is not None:
                raise AttributeError(
                    "Must provide a flag map to initialise DataMask")
            self.flag_map = {"BAD": (2, "Generic bad pixel flag")}
        else:
            self.flag_map = flag_map
        # Initialise the mask with all pixels being valid
        if bitmask is None:
            self.bitmask = np.zeros(shape, dtype=int)
            self.masks = {}
            for key in self.flag_map.keys():
                self.masks[key] = np.zeros(shape, dtype=bool)
        else:
            self.bitmask = bitmask.astype(int)
            self.masks = {}
            for key in self.flag_map.keys():
                self.masks[key] = self.get_flag_map_from_bitmask(key)

    def __decode_bitmask(self, value):
        return np.bitwise_and(self.bitmask, value) > 0

    def flag_pixels(self, mask, flag_name, desc=""):
        """Set or overwrite a named boolean mask.

        Parameters
        ----------
        mask : numpy.ndarray of bool
            Boolean array with the same shape as :attr:`bitmask`.
        flag_name : str
            Name of the flag to set.
        desc : str, optional
            Description for the flag. If the flag does not exist yet, it is
            created with a new bit value and this description.

        Raises
        ------
        ValueError
            If ``mask`` has a shape incompatible with :attr:`bitmask`.
        """
        if flag_name not in self.flag_map:
            self.add_new_flag(flag_name, desc=desc)
        # Check that the bitmask does not already contain this flag
        bit_flag_map = self.get_flag_map_from_bitmask(flag_name)
        self.bitmask[bit_flag_map] -= self.flag_map[flag_name][0]
        self.bitmask[mask] += self.flag_map[flag_name][0]
        # Store the individual boolean map
        self.masks[flag_name] = mask

    def get_flag_map_from_bitmask(self, flag_name):
        """Return the boolean mask for a given flag decoded from :attr:`bitmask`.

        Parameters
        ----------
        flag_name : str
            Name of the flag to decode.

        Returns
        -------
        numpy.ndarray of bool
            Boolean selection array for the requested flag.
        """
        return self.__decode_bitmask(self.flag_map[flag_name][0])

    def get_flag_map(self, flag_name=None):
        """Return a combined boolean mask for one or more flags.

        Parameters
        ----------
        flag_name : str or iterable of str, optional
            If ``None``, return a mask of any flagged pixel
            (i.e., ``bitmask > 0``). If a string or iterable is given, combine the
            corresponding boolean masks with a logical OR.

        Returns
        -------
        numpy.ndarray of bool
            Boolean mask with the same shape as :attr:`bitmask`.
        """
        if flag_name is not None:
            if type(flag_name) is str:
                return self.masks.get(flag_name, np.zeros_like(self.bitmask, dtype=bool))
            else:
                mask = np.zeros_like(self.bitmask, dtype=bool)
                for flag in flag_name:
                    mask |= self.get_flag_map(flag)
                return mask
        else:
            return self.bitmask > 0

    def add_new_flag(self, name, value=None, desc=""):
        """Register a new flag in :attr:`flag_map`.

        Parameters
        ----------
        name : str
            Flag name.
        value : int, optional
            Power-of-two integer to represent the new flag. If ``None``, the next
            available power-of-two is chosen (twice the current maximum).
        desc : str, optional
            Human-readable description.

        Notes
        -----
        This method does not change :attr:`bitmask`; it only registers metadata and
        prepares the corresponding boolean mask in :attr:`masks`.
        """
        if value is None:
            value = max([v[0] for v in self.flag_map.values()]) * 2
        self.flag_map[name] = (value, desc)
        self.masks[name] = np.zeros(self.bitmask.shape, dtype=bool)

    def plot(self, fig=None, ax=None, show=False,
             vmax=None, vmin=None, title=None,
             max_colorbar_ticks=10):
        """Plot the integer bitmask plus a per-flag summary table.

        If the bitmask has more than 2 dimensions, all leading axes are collapsed
        with a bitwise OR, preserving the last two axes as the image plane.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Existing figure to draw on. If omitted and ``ax`` is ``None``, a
            suitable figure with two rows is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw the image on. If ``None``, new axes are created as part
            of a figure layout that also includes a table panel.
        show : bool, default=False
            If ``True``, call :func:`matplotlib.pyplot.show` at the end.
        vmax, vmin : float, optional
            Color limits for the image. Defaults are inferred from data.
        title : str, optional
            Title for the image panel.
        max_colorbar_ticks : int, default=10
            If the number of unique integer values in the image is less than or
            equal to this threshold, colorbar ticks are placed exactly on those
            values.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax_img : matplotlib.axes.Axes
            Axes with the image.
        ax_tbl : matplotlib.axes.Axes or None
            Axes with the table (``None`` if no table was created).
        """
        if self.bitmask is None:
            raise ValueError("DataMask.bitmask is None.")

        data = self.bitmask

        # Determine collapse axes: collapse all leading axes, keep the final two as image
        if data.ndim < 2:
            raise ValueError(f"plot() expects at least 2D bitmask, got shape {data.shape}")

        if data.ndim > 2:
            axes_to_collapse = tuple(range(0, data.ndim - 2))
            # Collapse by bitwise OR across leading axes
            data2d = np.bitwise_or.reduce(data, axis=axes_to_collapse)
        else:
            axes_to_collapse = ()
            data2d = data

        if fig is None and ax is None:
            aspect = data2d.shape[1] / data2d.shape[0]
            fig = plt.figure(constrained_layout=True,
                             figsize=(8, 8 / aspect * 5 / 3))
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 3],
                                  hspace=0.15)
            ax_img = fig.add_subplot(gs[1, 0])
            ax_tbl = fig.add_subplot(gs[0, 0])
            ax_tbl.axis("off")
        elif ax is not None:
            fig = ax.figure
            ax_img = ax
            ax_tbl = None

        # Build the image
        unique_vals = np.unique(data2d)
        cm = plt.get_cmap("tab20")
        cm.set_under("white")
        cm.set_over("black")
        if vmin is None:
            vmin = 2
        if vmax is None:
            vmax = unique_vals.max()

        im = ax_img.imshow(data2d, origin="lower", interpolation="none",
                           aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
        ax_img.set_xlabel("x [pix]")
        ax_img.set_ylabel("y [pix]")
        if title is None:
            title = "Bitmask"
        ax_img.set_title(title)

        cbar = fig.colorbar(im, ax=ax_img, fraction=0.05, pad=0.04,
                            orientation="horizontal", extend='both')
        if unique_vals.size <= max_colorbar_ticks:
            cbar.set_ticks(unique_vals[unique_vals > 0].astype(int))
        cbar.set_label("Bitmask value")

        # Build the per-flag table (collapse each flag's boolean mask over the same axes)
        rows = []
        total_px = data2d.size
        sorted_flags = sorted(self.flag_map.items(), key=lambda kv: kv[1][0])

        for name, (val, desc) in sorted_flags:
            mask_nd = self.get_flag_map_from_bitmask(name)  # same shape as bitmask
            if mask_nd.ndim > 2 and axes_to_collapse:
                mask_2d = np.any(mask_nd, axis=axes_to_collapse)
            else:
                mask_2d = mask_nd

            count = np.count_nonzero(mask_2d)
            frac = 100.0 * count / total_px if total_px > 0 else 0.0
            rows.append([
                name,
                str(val),
                f"{val:#0{10}b}",
                f"{count}",
                f"{frac:.2f}%",
                (desc or "")
            ])

        any_count = np.count_nonzero(data2d)
        any_frac = 100.0 * any_count / total_px if total_px > 0 else 0.0
        rows.append(["ANY", "--", "--", f"{any_count}", f"{any_frac:.2f}%", "Pixels with one or more flags set"])

        # Render the table
        if ax_tbl is None:
            text_lines = [
                "Bit descriptions:\n",
                "Name | Value | Binary | Count | Frac | Description"
            ]
            for r in rows:
                text_lines.append(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]}")
            ax_img.text(
                1.02, 0.5, "\n".join(text_lines), transform=ax_img.transAxes,
                va="center", ha="left", family="monospace",
                fontsize="x-small")
        else:
            col_labels = ["Flag", "Value", "Binary", "Count", "Frac", "Description"]
            table = ax_tbl.table(cellText=rows, colLabels=col_labels, loc="center")
            table.auto_set_font_size(True)
            # table.scale(1.0, 1.2)
            ax_tbl.set_title("Bit definitions and pixel stats", fontsize=11, pad=6)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax_img, ax_tbl

    def dump_to_hdu(self):
        """Serialize the mask to a FITS :class:`~astropy.io.fits.ImageHDU`.

        The header stores the per-flag metadata in cards named ``FLAG_<NAME>`` with
        the value equal to the integer bit and the card comment equal to the flag
        description.

        Returns
        -------
        astropy.io.fits.ImageHDU
            Image HDU named ``"MASK"`` containing the integer bitmask and metadata.
        """
        header = fits.Header()
        header["COMMENT"] = "Each flag KEY is stored using the convention FLAG_KEY"
        for flag_name, value in self.flag_map.items():
            # Store the value and the description
            header[f"FLAG_{flag_name}"] = value
        header["COMMENT"] = "A value of 0 means unmasked"
        hdu = fits.ImageHDU(name="MASK", data=self.bitmask, header=header)
        return hdu

    @classmethod
    def from_hdu(cls, hdu):
        """Construct a :class:`DataMask` from a FITS HDU.

        Parameters
        ----------
        hdu : astropy.io.fits.ImageHDU
            HDU produced by :meth:`dump_to_hdu` (name ``"MASK"``).

        Returns
        -------
        DataMask
            New instance with :attr:`flag_map` and :attr:`bitmask` restored.
        """
        flag_map = {}
        for k in hdu.header.keys():
            if "FLAG" in k:
                name = k.replace("FLAG_", "")
                value = hdu.header[k]
                description = hdu.header.comments[k]
                flag_map[name] = (value, description)
        return cls(flag_map=flag_map, bitmask=hdu.data)

# =============================================================================


class DataContainer(ABC, VerboseMixin):
    """Abstract base class for PyKOALA data containers.

    Concrete subclasses represent specific data types (for example,
    :class:`RSS` and :class:`Cube`). A data container bundles the primary
    science arrays with metadata, a data-quality mask, WCS, the original
    FITS header, and a reduction history.

    Attributes
    ----------
    intensity : astropy.units.Quantity
        Science data array.
    variance : astropy.units.Quantity
        Per-element variance associated with :attr:`intensity`.
    mask : :class:`DataMask`
        Bit-flag data-quality mask aligned with :attr:`intensity`.
    info : dict
        Auxiliary metadata (for example, name, exposure time, fibre positions).
    history : :class:`DataContainerHistory`
        Log of data-reduction operations.
    header : astropy.io.fits.Header
        Original FITS header (or an empty header).
    wcs : astropy.wcs.WCS or None
        World coordinate system for :attr:`intensity`, if applicable.
    """

    @property
    def intensity(self):
        """
        :class:`astropy.units.Quantity` containing the intensity of each resolution
        element (pixel, fibre, spaxel).
        """
        return self._intensity

    @intensity.setter
    def intensity(self, value : u.Quantity):
        self._intensity = value

    @intensity.deleter
    def intensity(self):
        del self._intensity

    @property
    def variance(self):
        """
        :class:`astropy.units.Quantity` uncertainties associated to the
        ``intensity`` values.
        """
        return self._variance

    @variance.setter
    def variance(self, value : u.Quantity):
        self._variance = value

    @variance.deleter
    def variance(self):
        del self._variance

    @property
    def inverse_variance(self):
        """
        :class:`astropy.units.Quantity` inverse variance associated to the
        ``intensity`` values.
        """
        return 1 / self.variance

    @property
    def snr(self):
        """
        :class:`astropy.units.Quantity` Signal-to-noise ratio defined as
        ``intensity / variance**0.5``.
        """
        return self.intensity / self.variance**0.5

    @property
    def mask(self):
        """:class:`DataMask` associated to ``intensity``."""
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @mask.deleter
    def mask(self):
        del self._mask

    @property
    def info(self):
        """:class:`dict` storing auxiliary data (name, exposure time, fibre position, etc.)."""
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def history(self):
        """:class:`DataContainerHistory` a log recording the data processing steps."""
        return self._history
    
    @history.setter
    def history(self, value):
        self._history = value

    @property
    def header(self):
        """:class:`astropy.fits.Header` associated to the original file."""
        return self._header
    
    @header.setter
    def header(self, value):
        assert isinstance(value, fits.Header), "Header must be an instance of astropy.fits.Header"
        self._header = value

    @property
    def wcs(self):
        """
        :class:`astropy.wcs.WCS` world coordinate system associated to
        ``intensity``.
        """
        return self._wcs
    
    @wcs.setter
    def wcs(self, value):
        assert isinstance(value, WCS) or (value is None), "wcs must be an instance of astropy.wcs.WCS"
        self._wcs = value

    def __init__(self, **kwargs):
        self._intensity = ancillary.check_unit(kwargs["intensity"])
        self._variance = ancillary.check_unit(kwargs.get("variance",
            np.full_like(self._intensity, np.nan, dtype=type(np.nan))))
        self._mask = kwargs.get("mask", None)
        if self._mask is None:
            # Initialise an empty mask
            self._mask = DataMask(shape=self.intensity.shape)
        self.info = kwargs.get("info", dict())
        self.fill_info()
        # Setup datacontainer logging/verbosity and history
        self.logger = kwargs.get("logger", "pykoala.dc")
        self.verbose = kwargs.get("verbose", True)
        self.history = kwargs.get("history",
                                  DataContainerHistory(logger=self.logger,
                                                       verbose=self.verbose))
        self.header = kwargs.get("header", fits.Header())
        self.wcs = kwargs.get("wcs", None)

    def fill_info(self):
        """Check the keywords of info and fills them with placeholders."""
        if "name" not in self.info.keys():
            self.info["name"] = "N/A"

    def copy(self):
        """Return a copy of the DataContainer."""
        return copy.deepcopy(self)

    def is_corrected(self, correction):
        """Check if a ``Correction`` has been applied to the DataContainer."""
        if self.history.is_record(title=correction):
            return True
        else:
            return False

    def _to_hdul(self):
        """Build a FITS :class:`~astropy.io.fits.HDUList` for this container.

        The HDU list includes:
        - ``PRIMARY`` with PyKOALA metadata, original header, and history cards.
        - ``INTENSITY`` with science data and WCS.
        - ``VARIANCE`` with variance data and WCS.
        - ``MASK`` with the serialized :class:`DataMask`.

        Returns
        -------
        astropy.io.fits.HDUList
            HDU list ready to be written to disk.
        """
        primary = fits.PrimaryHDU()
        primary.header['pykoala0'] = __version__, "PyKOALA version"
        primary.header['pykoala1'] = datetime.now().strftime(
            "%d_%m_%Y_%H_%M_%S"), "creation date / last change"
        # Fill the header with the log information
        primary.header = self.history.dump_to_header(primary.header)
        # Include the original header
        primary.header["ORIHEAD"] = len(self.header), "Number of cards of original header"
        primary.header.extend(self.header)
        # Ensure that the name is PRIMARY
        primary.name = "PRIMARY"

        hdu_list = [primary]
        header = self.wcs.to_header()
        header["bunit"] = self.intensity.unit.to_string()
        hdu_list.append(fits.ImageHDU(
            data=self.intensity.value, name='INTENSITY',
            header=header
        )
        )
        header["bunit"] = self.variance.unit.to_string()
        hdu_list.append(fits.ImageHDU(
            data=self.variance.value, name='VARIANCE', header=header))
        # Store the mask information
        hdu_list.append(self.mask.dump_to_hdu())
        hdul = fits.HDUList(hdu_list)
        return hdul

    @classmethod
    def _dc_params_from_hdul(cls, hdul):
        """Extract constructor parameters from a PyKOALA FITS file.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            HDU list produced by :meth:`_to_hdul`.

        Returns
        -------
        dict
            Dictionary with keys ``history``, ``header``, ``intensity``,
            ``variance``, ``wcs``, and ``mask`` suitable for ``cls(**params)``.
        """
        dc_params = {}
        dc_params["history"] = DataContainerHistory.from_header(
            hdul["PRIMARY"].header)
        # Fetch the information of the original header
        if "ORIHEAD" in hdul["PRIMARY"].header:
            star_original_header = hdul["PRIMARY"].header.index("ORIHEAD")
            len_header = hdul["PRIMARY"].header["ORIHEAD"]
            dc_params["header"] = hdul["PRIMARY"].header[
                star_original_header + 1:star_original_header + len_header]
        dc_params["intensity"] = hdul["INTENSITY"].data << u.Unit(
            hdul["INTENSITY"].header.get("BUNIT", 1))
        dc_params["variance"] = hdul["VARIANCE"].data << u.Unit(
            hdul["VARIANCE"].header.get("BUNIT", 1))
        dc_params["wcs"] = WCS(hdul["INTENSITY"].header)
        dc_params["mask"] = DataMask.from_hdu(hdul["MASK"])
        return dc_params
    
    @abstractmethod
    def from_fits():
        """Create an instance from a PyKOALA-compliant FITS file.

        Implementations must read the HDUs created by :meth:`_to_hdul` and return
        a fully initialized instance.
        """
        pass

# =============================================================================


class SpectraContainer(DataContainer):
    """Base class for spectral data containers.

    Extends :class:`DataContainer` by adding a common spectral coordinate
    :attr:`wavelength`. Subclasses include :class:`RSS` and :class:`Cube`.

    Attributes
    ----------
    wavelength : astropy.units.Quantity
        1D array of wavelength samples shared by all spectra.
    """

    @property
    def wavelength(self):
        """:class:`astropy.units.Quantity` wavelength array, common to all spectra."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = ancillary.check_unit(value)

    @wavelength.deleter
    def wavelength(self):
        del self._wavelength

    @property
    def n_wavelength(self):
        """Number of wavekengths in the `wavelength` array"""
        return self._wavelength.size

    @property
    def n_spectra(self):
        """Number of spectra in the `intensity` array."""
        return int(self._intensity.size / self._wavelength.size)

    @property
    @abstractmethod
    def rss_intensity(self):
        """
        :class:`astropy.units.Quantity` ``intensity`` array sorted as
        ``[n_spectra, n_wavelength]``.
        """
        pass

    @rss_intensity.setter
    @abstractmethod
    def rss_intensity(self):
        pass

    @property
    @abstractmethod
    def rss_variance(self):
        """:class:`astropy.units.Quantity` uncertainties associated to ``intensity_rss``"""
        pass

    @rss_variance.setter
    @abstractmethod
    def rss_variance(self):
        pass

    @property
    def rss_snr(self):
        return self.rss_intensity / self.rss_variance**0.5

    @abstractmethod
    def rss_to_original(self, rss_shape_data):
        """Reshape an RSS-like array into the original ``intensity`` shape."""
        pass

    @abstractmethod
    def original_to_rss(self, rss_shape_data):
        """Reshape the original ``intensity`` shape into an RSS-like array."""
        pass

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if "wavelength" in kwargs:
            self.wavelength = ancillary.check_unit(kwargs["wavelength"],
                                                   u.angstrom)
        elif "wcs" in kwargs:
            self.wavelength = kwargs["wcs"].spectral.array_index_to_world(
            np.arange(kwargs["wcs"].spectral.array_shape[0])).to('angstrom')
        else:
            raise AttributeError("Either a wavelength or wcs must be provided")

    def resample_wavelength_grid(self, wavelength, reference_wl=None, mask_threshold=0.0,
                                **interp_kwargs):
        """Resample all spectra onto a new wavelength grid (flux-conserving).

        Intensity and variance are resampled independently using
        :func:`pykoala.ancillary.flux_conserving_interpolation_nd`. If a
        :class:`DataMask` is present, each named boolean layer is propagated via
        :func:`pykoala.ancillary.bool_mask_interpolation`.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Target wavelength grid (1D, monotonically increasing).
        reference_wl : astropy.units.Quantity, optional
            Source wavelength grid. Defaults to :attr:`wavelength` of the
            instance.
        mask_threshold : float, default=0.0
            Threshold in ``[0, 1]`` for boolean mask propagation. Values above
            the threshold are considered ``True`` in the resampled mask.
        **interp_kwargs
            Forwarded to :func:`pykoala.ancillary.flux_conserving_interpolation_nd`.
            If ``return_nan_flag=True``, any output pixel affected by NaNs in the
            inputs is tracked into/with the mask flag ``"interpolated_nans"``.

        Raises
        ------
        ValueError
            If ``reference_wl`` has a size inconsistent with the current spectra.

        Notes
        -----
        The method updates :attr:`wavelength`, :attr:`rss_intensity`,
        and :attr:`rss_variance` in place.
        """
        self.vprint(f"Resampling spectral axis")
        intensity = self.rss_intensity
        variance = self.rss_variance

        if reference_wl is None:
            reference_wl = self.wavelength.copy()
        else:
            if reference_wl.size != self.wavelength:
                raise ValueError(
                    "Reference wavelength dimensions do not match"
                    + "SpectraContainer wavelength attribute")

        new_intensity = np.zeros((intensity.shape[0], wavelength.size)) << intensity.unit
        new_variance = np.zeros((variance.shape[0], wavelength.size)) << variance.unit
        # --- propagate mask (bit-flags) if present
        mask = getattr(self, "mask", None)
        if mask is not None:
            self.vprint("Resampling RSS mask")
            # Create the new mask
            new_mask = DataMask(
                shape=self.rss_to_original(new_intensity).shape,
                flag_map=mask.flag_map)
            for k in mask.flag_map.keys():
                m = ancillary.bool_mask_interpolation(
                    wavelength, reference_wl, mask.masks[k], threshold=mask_threshold)
                new_mask.flag_pixels(m, flag_name=k)

            self.mask = new_mask

        if "return_nan_flag" in interp_kwargs and interp_kwargs["return_nan_flag"]:
            self.vprint("NaNs will be propagated")
            new_intensity, int_nans = ancillary.flux_conserving_interpolation_nd(
                wavelength, reference_wl, intensity, **interp_kwargs)
            new_variance, var_nans = ancillary.flux_conserving_interpolation_nd(
                wavelength, reference_wl, variance, **interp_kwargs)
            interp_nans_mask = int_nans | var_nans
            new_mask = self.mask.get_flag_map("interpolated_nans") | interp_nans_mask
            self.mask.flag_pixels(new_mask, flag_name="interpolated_nans")
        else:
            new_intensity = ancillary.flux_conserving_interpolation_nd(
                wavelength, reference_wl, intensity, **interp_kwargs)
            new_variance = ancillary.flux_conserving_interpolation_nd(
                wavelength, reference_wl, variance, **interp_kwargs)

        self.wavelength = wavelength
        self.rss_intensity = new_intensity
        self.rss_variance = new_variance
    

    def get_spectra_sorted(self, wave_range=None):
        """Return indices that sort spectra by median flux.

        Parameters
        ----------
        wave_range : tuple of (Quantity or float, Quantity or float), optional
            Wavelength limits to compute per-spectrum medians. If omitted, the
            full range is used. Floats are interpreted in the unit of
            :attr:`wavelength`.

        Returns
        -------
        numpy.ndarray of int
            Indices that sort from faintest to brightest median flux.
        """
        if wave_range is None:
            wave_mask = np.ones_like(self.wavelength, dtype=bool)
        else:
            wave_mask = (self.wavelength >= wave_range[0]) & (
                self.wavelength <= wave_range[1])
        median_intensity = np.nanmedian(self.rss_intensity[:, wave_mask], axis=1)
        # Put bad fibres at the bottom
        median_intensity = np.nan_to_num(median_intensity, nan=0.0)
        return np.argsort(median_intensity)

    def plot_spectra(
        self,
        indices,
        ax=None,
        *,
        wave_range=None,
        show_variance=False,
        variance_alpha=0.2,
        labels=None,
        colors=None,
        normalize=None,
        drawstyle="default",
        mask_invalid=True,
        flux_scale=None,
        **plot_kwargs,
    ) -> plt.Axes:
        """
        Plot one or more spectra (RSS ordering) versus wavelength.

        Parameters
        ----------
        indices : int or array-like of int
            Index/indices into ``rss_intensity`` (axis 0).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        wave_range : tuple, optional
            ``(wmin, wmax)`` wavelength limits. Elements can be floats
            (assumed in :attr:`wavelength`.unit) or :class:`~astropy.units.Quantity`.
        show_variance : bool, default=False
            If ``True``, shade +/- 1 sigma  using :attr:`rss_variance`.
        variance_alpha : float, default=0.2
            Alpha for the variance shading patch.
        labels : sequence of str, optional
            One legend label per spectrum. Defaults to ``"idx {i}"``.
        colors : sequence, optional
            Matplotlib-compatible colors. Cycles if shorter than number of spectra.
        normalize : {None, 'median', 'max'}, optional
            Per-spectrum normalization before plotting.
        drawstyle : str, default='default'
            Matplotlib drawstyle (e.g., ``'steps-mid'``).
        mask_invalid : bool, default=True
            If ``True``, mask NaNs in intensity/variance before plotting.
        flux_scale : {'linear','log'}, optional
            Y-axis scale. If ``None``, stays linear.
        **plot_kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object (new or owner of ``ax``).
        ax : matplotlib.axes.Axes
            Axes with the plotted spectra.

        See Also
        --------
        :meth:`get_spectra_sorted`
        """
        # ---- prepare inputs
        if isinstance(indices, int):
            idx = self.get_spectra_sorted()[-indices:]
        else:
            idx = np.atleast_1d(indices).astype(int)

        if idx.ndim != 1:
            raise ValueError("`indices` must be an int or 1D iterable of ints.")
        n = idx.size

        if labels is None:
            labels = [f"idx {i}" for i in idx]
        else:
            labels = list(labels)
            if len(labels) != n:
                raise ValueError("`labels` length must match number of indices.")

        # Axes
        if ax is None:
            fig, ax = plt.subplots()

        if wave_range is None:
            wave_mask = np.ones(self.wavelength.size, dtype=bool)
            wl = self.wavelength
        else:
            wmin, wmax = wave_range
            wmin = ancillary.check_unit(wmin, self.wavelength.unit)
            wmax = ancillary.check_unit(wmax, self.wavelength.unit)
            wave_mask = (self.wavelength >= wmin) & (self.wavelength <= wmax)
            wl = self.wavelength[wave_mask]

        # ---- intensity (and variance) slices
        intensities = self.rss_intensity[idx][:, wave_mask]

        if show_variance:
            variances = self.rss_variance[idx][:, wave_mask]
            if variances is None:
                raise AttributeError("`rss_variance` is not available.")

        # ---- masking invalids
        if mask_invalid:
            mask_good = np.isfinite(intensities)
            if show_variance:
                mask_good &= np.isfinite(variances)
            # keep only wavelengths that are valid in any spectrum, but also honor wave_range
            wave_mask = wave_mask & np.any(mask_good, axis=0)

            intensities = intensities[:, wave_mask]
            if show_variance:
                variances = variances[:, wave_mask]

        # ---- normalization
        def _norm_factor(f):
            if normalize is None:
                return 1.0
            if normalize == "median":
                m = np.nanmedian(f)
                return m if np.isfinite(m) and m != 0 else 1.0
            if normalize == "max":
                m = np.nanmax(f)
                return m if np.isfinite(m) and m != 0 else 1.0
            raise ValueError("`normalize` must be None, 'median', or 'max'.")

        low, up = np.nanpercentile(intensities.value, [50, 95])
        # ---- plotting fibres
        for k, (ik, lab) in enumerate(zip(idx, labels)):
            flux = intensities[k]
            norm = _norm_factor(flux)

            color = None if colors is None else colors[k % len(colors)]
            line = ax.plot(
                wl, flux / norm,
                label=lab,
                color=color,
                drawstyle=drawstyle,
                **plot_kwargs,
            )
            if show_variance:
                sigma = variances[k]**0.5
                ax.fill_between(
                    wl,
                    (flux - sigma) / norm,
                    (flux + sigma) / norm,
                    alpha=variance_alpha,
                    color=line[0].get_color() if color is None else color,
                    linewidth=0
                )

        if flux_scale is not None:
            ax.set_yscale(flux_scale)
        ax.set_ylim(low / 10, up * 1.5)
        ax.set_xlabel(f"Wavelength [{wl.unit}]")
        ax.set_ylabel(f"Intensity [{flux.unit}]")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.3)
        return fig, ax

class RSS(SpectraContainer):
    """Row-stacked spectra container.

    Stores spectra as a 2D array with shape ``(n_fibres, n_wavelength)`` and
    a shared :attr:`~SpectraContainer.wavelength`. Fibre sky positions are
    stored in :attr:`~DataContainer.info` (keys ``'fib_ra'``, ``'fib_dec'``).
    """

    @property
    def rss_intensity(self):
        return self._intensity

    @rss_intensity.setter
    def rss_intensity(self, value : u.Quantity):
        self.intensity = value

    @property
    def rss_variance(self):
        return self._variance

    @rss_variance.setter
    def rss_variance(self, value : u.Quantity):
        self.variance = value

    def rss_to_original(self, rss_shape_data):
        return rss_shape_data

    def original_to_rss(self, original_data):
        return original_data

    @property
    def fibre_diameter(self):
        """:class:`astropy.units.Quantity` angular diameter of the RSS fibres."""
        return self._fibre_diameter
    
    @fibre_diameter.setter
    def fibre_diameter(self, value : u.Quantity):
        assert isinstance(value, u.Quantity) or value is None, (
            "Fibre diameter must be a astropy.units.Quantity")
        self._fibre_diameter = ancillary.check_unit(value, u.arcsec)

    @property
    def sky_fibres(self):
        """Indices of the RSS sky fibres."""
        return self._sky_fibres
    
    @sky_fibres.setter
    def sky_fibres(self, value):
        self._sky_fibres = value

    @property
    def science_fibres(self):
        """Indices of fibres with a science target."""
        return np.delete(np.arange(self.intensity.shape[0]), self.sky_fibres)

    def __init__(self, **kwargs):
        assert ('wavelength' in kwargs)
        assert ('intensity' in kwargs)
        assert ('fibre_diameter' in kwargs)
        if "logger" not in kwargs:
            kwargs['logger'] = "pykoala.rss"

        self.fibre_diameter = kwargs.get("fibre_diameter", None)
        self.sky_fibres = kwargs.get("sky_fibres", [])
        super().__init__(**kwargs)

    def get_centre_of_mass(self, wavelength_step=1, stat=np.nanmedian, power=1.0):
        """Compute the flux-weighted sky center of mass per wavelength bin.

        The COM is measured over fibre positions (RA, Dec) using flux weights from
        :attr:`intensity`. Within each wavelength bin of size ``wavelength_step``,
        a statistic (e.g., median) is applied across the bin.

        Parameters
        ----------
        wavelength_step : int, default=1
            Bin size along the spectral axis (number of wavelength samples).
            ``1`` means evaluate at every wavelength sample.
        stat : callable, default=numpy.nanmedian
            Reduction statistic applied across the bin (vector -> scalar).
        power : float, default=1.0
            If not 1, use ``intensity**power`` as weights.

        Returns
        -------
        ra_com : astropy.units.Quantity
            Flux-weighted RA per wavelength sample (same length as
            :attr:`wavelength`).
        dec_com : astropy.units.Quantity
            Flux-weighted Dec per wavelength sample.

        Notes
        -----
        Requires ``'fib_ra'`` and ``'fib_dec'`` in :attr:`info`.
        """
        ra = self.info["fib_ra"]
        dec = self.info["fib_dec"]
        ra_com = np.empty(self.wavelength.size) << ra.unit
        dec_com = np.empty(self.wavelength.size) << dec.unit
        for wave_range in range(0, self.wavelength.size, wavelength_step):
            # Mean across all fibres
            ra_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[:, wave_range: wave_range +
                               wavelength_step]**power * ra[:, np.newaxis],
                axis=0) / np.nansum(self.intensity[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            # Statistic (e.g., median, mean) per wavelength bin
            ra_com[wave_range: wave_range + wavelength_step] = stat(
                ra_com[wave_range: wave_range + wavelength_step])
            
            dec_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[:, wave_range: wave_range +
                               wavelength_step]**power * dec[:, np.newaxis],
                axis=0) / np.nansum(self.intensity[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            dec_com[wave_range: wave_range + wavelength_step] = stat(
                dec_com[wave_range: wave_range + wavelength_step])
        return ra_com, dec_com

    def update_coordinates(self, new_coords=None, offset=None):
        """Update fibre sky positions.

        Parameters
        ----------
        new_coords : tuple of (Quantity, Quantity), optional
            Absolute coordinates ``(ra, dec)`` for all fibres, both 1D arrays
            with units of degree.
        offset : tuple of (Quantity, Quantity), optional
            Offsets ``(d_ra, d_dec)`` to add to the current positions, with units
            of degree. Ignored if ``new_coords`` is given.

        Raises
        ------
        NameError
            If neither ``new_coords`` nor ``offset`` is provided.

        Notes
        -----
        The original positions are saved into :attr:`info` as
        ``'ori_fib_ra'`` and ``'ori_fib_dec'``. A history entry named
        ``'update_coords'`` is appended.
        """
        self.info['ori_fib_ra'], self.info['ori_fib_dec'] = (self.info["fib_ra"].copy(),
                                                             self.info["fib_dec"].copy())
        if new_coords is not None:
            self.info["fib_ra"] = new_coords[0]
            self.info["fib_dec"] = new_coords[1]
        elif offset is not None:
            self.info["fib_ra"] += offset[0]
            self.info["fib_dec"] += offset[1]
        else:
            raise NameError(
                "Either `new_fib_coord` or `new_fib_coord_offset` must be provided")
        self.history('update_coords', "Offset-coords updated")
        self.vprint("[RSS] Offset-coords updated")

    def to_fits(self, filename=None, overwrite=False, checksum=False):
        """Write the RSS into a FITS file.

        This method allows to store all the information contained in the RSS
        into a FITS file composed of several extensions.
        
        The information is stored in the following HDU extensions

        - ``PRIMARY``: contains the metadata associated to the :class:`DataContainerHistory` as well as the original ``header`` of the RSS.
        - ``INTENSITY``: contains the data associated to the ``intensity`` and the WCS information.
        - ``VARIANCE``: contains the data associated to the ``variance`` and the WCS information.
        - ``MASK``: contains the data associated to the ``mask`` attribute.
        - ``INFO``: contains the data associated to the ``info`` attribute.

        Parameters
        ----------
        filename: str
            Output filename of the FITS file.
        overwrite: bool, optional
            If True, overwrite the output file if it exists.
        checksum: bool, optional
            If True, adds both DATASUM and CHECKSUM cards to the headers of all
            HDU's written to the file.

        """

        hdul = self._to_hdul()

        if filename is None:
            filename = 'rss_{}_{}.fits.gz'.format(
                self.info.get("name", "frame"),
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

        # Fibre information table
        pykoala_info_table = Table(
            names=["fib_ra", "fib_dec"],
            data=[self.info["fib_ra"], self.info["fib_dec"]],
            meta=dict(fib_ra="Fibre RA position (deg)",
                      fib_dec="Fibre DEC position (deg)"))
        info_header = fits.Header()
        info_header["NAME    "] = self.info.get("name", "N/A"), "Object name"
        info_header["EXPTIME "] = self.info.get("exptime", 0 << u.second).to_value(
            "second"), "exposure time (s)"
        info_header["AIRMASS "] = self.info["airmass"], "airmass at centre of FoV"
        info_header["FIBDIAM "] = self.fibre_diameter.to_value("arcsec"), "fibre diameter size (arcsec)"

        hdul.append(fits.BinTableHDU(name="INFO", data=pykoala_info_table,
                                     header=info_header))
        # Save the HDUL into a FITS file.
        hdul.verify('fix')
        hdul.writeto(filename, overwrite=overwrite, checksum=checksum)
        hdul.close()
        self.vprint(f"File saved as {filename}")

    @classmethod
    def from_fits(cls, filename):
        """Read a PyKOALA RSS from a FITS file.

        Parameters
        ----------
        filename : str or path-like
            Path to a FITS file produced by :meth:`to_fits`.

        Returns
        -------
        RSS
            New instance with intensity, variance, wavelength, mask, WCS, history,
            original header, and fibre information loaded.

        Notes
        -----
        Expected extensions:
        - ``PRIMARY``: history and original header
        - ``INTENSITY``: science data + WCS
        - ``VARIANCE``: variance data + WCS
        - ``MASK``: :class:`DataMask`
        - ``INFO``: binary table with ``fib_ra`` and ``fib_dec``, and header
            keywords ``NAME``, ``EXPTIME``, ``AIRMASS``, ``FIBDIAM``.
        """
        with fits.open(filename) as hdul:
            # Extract the basic parameters to initialise a DC
            dc_parameters = cls._dc_params_from_hdul(hdul)
            # Extract RSS-specific information
            info = {}
            info["fib_ra"] = hdul["INFO"].data["fib_ra"] << u.deg
            info["fib_dec"] = hdul["INFO"].data["fib_dec"] << u.deg
            info["name"] = hdul["INFO"].header.get("name")
            info["exptime"] = hdul["INFO"].header.get("exptime") << u.second
            info["airmass"] = hdul["INFO"].header.get("airmass")
            fibre_diameter = hdul["INFO"].header["fibdiam"] << u.arcsec
            wavelength = dc_parameters["wcs"].spectral.array_index_to_world(
                np.arange(dc_parameters["intensity"].shape[1]))

        return cls(info=info, fibre_diameter=fibre_diameter, wavelength=wavelength,
                   **dc_parameters)

    def get_integrated_fibres(self, wavelength_range=None):
        """Integrate each fibre over a wavelength range.

        Parameters
        ----------
        wavelength_range : tuple of (Quantity or float, Quantity or float), optional
            Integration limits ``(wmin, wmax)``. Floats are interpreted in the
            unit of :attr:`wavelength`. If omitted, the full range is used.

        Returns
        -------
        integrated_fibres : astropy.units.Quantity
            1D array with the integrated flux per fibre.
        integrated_variances : astropy.units.Quantity
            1D array with the integrated variance per fibre (assuming independent
            samples; see Notes).

        Notes
        -----
        The implementation uses a mean times the number of samples inside the mask,
        which is equivalent to a rectangle-rule sum on an evenly spaced grid.
        """
        if wavelength_range is not None:

            wavelength_range = [ancillary.check_unit(wl_r, self.wavelength.unit) for wl_r in wavelength_range]
            wave_mask = (self.wavelength >= wavelength_range[0]) & (
                self.wavelength <= wavelength_range[1]

            )
        else:
            wave_mask = np.ones(self.wavelength.size, dtype=bool)

        integrated_fibres = np.nanmean(self.intensity[:, wave_mask], axis=1
                                       ) * np.count_nonzero(wave_mask)
        integrated_variances = np.nanmean(self.variance[:, wave_mask], axis=1
                                       ) * np.count_nonzero(wave_mask)
        return integrated_fibres, integrated_variances

    def get_footprint(self):
        """Return a rectangular sky footprint that encloses all fibre positions.

        Returns
        -------
        astropy.units.Quantity
            Array of shape ``(4, 2)`` with the corners ``(ra, dec)`` in degrees.
        """
        min_ra, max_ra = self.info['fib_ra'].min(), self.info['fib_ra'].max()
        min_dec, max_dec = self.info['fib_dec'].min(), self.info['fib_dec'].max()
        footprint = np.array([[max_ra.to_value("deg"), max_dec.to_value("deg")],
                              [max_ra.to_value("deg"), min_dec.to_value("deg")],
                              [min_ra.to_value("deg"), max_dec.to_value("deg")],
                              [min_ra.to_value("deg"), min_dec.to_value("deg")]],
                              dtype=float) << u.deg
        return footprint

    def plot_rss_image(self, data=None, data_label="", fig_args={}, cmap_args={},
                       fibre_range=None,
                       wavelength_range=None,
                       output_filename=None):
        """Plots a RSS image with optional data, fibre, and wavelength ranges.

        Parameters
        ----------
        data : array-like, optional
            The 2D array data to be plotted. If `None`, `intensity` is used.
        data_label : str, optional
            The color bar label for the data being plotted. Default is an empty string.
        fig_args : dict, optional
            Additional keyword arguments passed to `pykoala.plotting.utils.new_figure` for customizing the figure. 
            Default is an empty dictionary.
        cmap_args : dict, optional
            Additional keyword arguments passed to the `pykoala.plotting.utils.plot_image` function for the colormap
            and normalization.  Default is an empty dictionary.
        fibre_range : tuple of int, optional
            A tuple specifying the range of fibres to include in the plot (start, end).
            If `None`, all fibres are included. Default is `None`.
        wavelength_range : tuple of float, optional
            A tuple specifying the range of wavelengths to include in the plot (start, end). If `None`, all wavelengths are included.
            Default is `None`.
        output_filename : str, optional
            If provided, the plot is saved to the specified file path. Default is `None`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        Notes
        -----
        - The function uses the internal attributes `self.wavelength` and `self.intensity` to obtain default x-values 
        (wavelengths) and y-values (fibre indices) if `data` is not provided.
        - The `new_figure` function is used to create a new figure, and `plot_image` is used to plot the data.
        - If `fibre_range` or `wavelength_range` is specified, the data is sliced accordingly.
        - The plot is saved to `output_filename` if provided, otherwise the figure is returned for display or further manipulation.

        
        """
        x = self.wavelength
        y = np.arange(0, self.intensity.shape[0])
        if data is None:
            data = self.intensity
            data_label = "Intensity"
        if fibre_range is not None:
            fibre_range = range(*fibre_range)
            data = data[fibre_range]
            y = y[fibre_range]
        if wavelength_range is not None:
            wavelength_range = range(*np.searchsorted(self.wavelength.value, wavelength_range))
            data = data[:, wavelength_range]
            x = x[wavelength_range]

        fig, axs = new_figure(self.info['name'], **fig_args)
        im, cb = plot_image(fig, axs[0, 0], cblabel=data_label, data=data,
                            x=x, y=y,
                            xlabel="Wavelength [AA]", ylabel="Fibre",
                            **cmap_args)

        if output_filename is not None:
            fig.savefig(output_filename, bbox_inches="tight")
        return fig

    def plot_mask(self, fig_args={}, cmap_args={}, output_filename=None):
        """Plots a mask image using the bitmask data.

        This method creates a plot of the bitmask data using a predefined colormap and normalization settings.
        It utilizes the `plot_rss_image` method to generate the plot.

        Parameters
        ----------
        fig_args : dict, optional
            Additional keyword arguments passed to the `new_figure` function for customizing the figure.
            Default is an empty dictionary.
        cmap_args : dict, optional
            Additional keyword arguments passed to the `plot_image` function for customizing the colormap.
            If not specified, the colormap is set to "Accent" and normalization to "Normalize". Default is an empty dictionary.
        output_filename : str, optional
            If provided, the plot is saved to the specified file path. Default is `None`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        See Also
        --------
        :func:`plot_rss_image`.
        """

        if "cmap" not in cmap_args:
            cmap_args["cmap"] = "Accent"
        if "norm" not in cmap_args:
            cmap_args["norm"] = "Normalize"
        fig = self.plot_rss_image(data=self.mask.bitmask, data_label="Bitmask",
                            fig_args=fig_args, cmap_args=cmap_args,
                            output_filename=output_filename)
        return fig

    def plot_fibres(self, data=None, cblabel="", fig_args={},
                    cmap_args={}, output_filename=None):
        """
        Plots a fibre map image, showing the spatial distribution of data across fibres.

        This method generates a plot that visualizes the spatial distribution of
        data across fibres, using the Right Ascension (RA) and Declination (Dec)
        of each fibre. If no data is provided, it uses the integrated fibre intensity data.

        Parameters
        ----------
        data : array-like, optional
            The data to be plotted. If `None`, the method calls `self.get_integrated_fibres()`
            to obtain the integrated fibre intensity data. Default is `None`.
        
        cblabel : str, optional
            The label for the color bar representing the data being plotted. Default is an empty string.
        
        fig_args : dict, optional
            Additional keyword arguments passed to the `new_figure` function for
            customizing the figure. Default is an empty dictionary.
        
        cmap_args : dict, optional
            Additional keyword arguments passed to the `plot_fibres` function for
            customizing the colormap. Default is an empty dictionary.
        
        output_filename : str, optional
            If provided, the plot is saved to the specified file path. Default is `None`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        if data is None:
            data, _ = self.get_integrated_fibres()
            cblabel = "Integrated intensity"
        if "figsize" not in fig_args:
            fig_args["figsize"] = (5, 5)
        fig, axs = new_figure(self.info['name'], **fig_args)
        axs[0, 0].set_aspect('auto')

        ax, *_ = plot_fibres(
            fig=fig, ax=axs[0, 0], cblabel=cblabel,
            data=data, rss=self, **cmap_args)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("DEC (deg)")
        if output_filename is not None:
            fig.savefig(output_filename, bbox_inches="tight")
        return fig


class Cube(SpectraContainer):
    """:class:`SpectraContainer` for a 3D IFS datacube.

    The primary arrays have shape ``(n_wave, n_row, n_col)``. The spectral axis
    is the first dimension, compatible with :mod:`astropy.wcs` spectral
    subcomponents.
    """

    # default_hdul_extensions_map = {"INTENSITY": "INTENSITY",
    #                                "VARIANCE": "VARIANCE"}

    # @property
    # def hdul(self):
    #     return self._hdul

    # @hdul.setter
    # def hdul(self, hdul):
    #     assert isinstance(hdul, fits.HDUList)
    #     self._hdul = hdul

    # @property
    # def intensity(self):
    #     return self.hdul[self.hdul_extensions_map['INTENSITY']].data
    
    # @intensity.setter
    # def intensity(self, intensity_corr):
    #     self.vprint("[Cube] Updating HDUL INTENSITY")
    #     self.hdul[self.hdul_extensions_map['INTENSITY']].data = intensity_corr

    # @property
    # def variance(self):
    #     return self.hdul[self.hdul_extensions_map['VARIANCE']].data

    # @variance.setter
    # def variance(self, variance_corr):
    #     self.vprint("[Cube] Updating HDUL variance")
    #     self.hdul[self.hdul_extensions_map['VARIANCE']].data = variance_corr

    @property
    def n_cols(self):
        """Number of spaxel columns (X dimension)."""
        return self.intensity.shape[2]

    @property
    def n_rows(self):
        """Number of spaxel rows (Y dimension)."""
        return self.intensity.shape[1]

    @property
    def rss_intensity(self):
        return self.original_to_rss(self.intensity)

    @rss_intensity.setter   
    def rss_intensity(self, value):
        self.intensity = value.T.reshape(self.intensity.shape)

    @property
    def rss_variance(self):
        return self.original_to_rss(self.variance)

    @rss_variance.setter   
    def rss_variance(self, value):
        self.variance = value.T.reshape(self.variance.shape)

    def __init__(self, **kwargs):

        if "logger" not in kwargs:
            kwargs['logger'] = "pykoala.cube"
        # if "intensity" not in kwargs:
        #     kwargs["intensity"] = self.intensity
        # if "variance" not in kwargs:
        #     kwargs["variance"] = self.variance
        # if "wcs" not in kwargs:
        #     kwargs["wcs"] = WCS(
        #         self.hdul[self.hdul_extensions_map['INTENSITY']].header)

        super().__init__(**kwargs)


    def rss_to_original(self, rss_shape_data):
        return np.reshape(rss_shape_data.T, (rss_shape_data.shape[1],
                                             self.intensity.shape[1],
                                             self.intensity.shape[2]))

    def original_to_rss(self, original_data):
        return np.reshape(original_data, (
            original_data.shape[0],
            original_data.shape[1] * original_data.shape[2])).T

    def get_centre_of_mass(self, wavelength_step=1, stat=np.median, power=1.0):
        """Flux-weighted image-plane center of mass per wavelength bin.

        Parameters
        ----------
        wavelength_step : int, default=1
            Bin size along the spectral axis (number of wavelength samples).
        stat : callable, default=numpy.median
            Reduction statistic across each bin.
        power : float, default=1.0
            If not 1, use ``intensity**power`` as weights.

        Returns
        -------
        x_com : numpy.ndarray of float
            COM along columns for each wavelength sample.
        y_com : numpy.ndarray of float
            COM along rows for each wavelength sample.
        """
        x = np.arange(0, self.n_cols, 1)
        y = np.arange(0, self.n_rows, 1)
        x_com = np.empty(self.n_wavelength)
        y_com = np.empty(self.n_wavelength)
        for wave_range in range(0, self.n_wavelength, wavelength_step):
            x_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[wave_range: wave_range + wavelength_step]**power * x[np.newaxis, np.newaxis, :],
                axis=(1, 2)) / np.nansum(self.intensity[wave_range: wave_range + wavelength_step]**power, axis=(1, 2))
            x_com[wave_range: wave_range + wavelength_step] = stat(x_com[wave_range: wave_range + wavelength_step])
            y_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[wave_range: wave_range + wavelength_step]**power * y[np.newaxis, :, np.newaxis],
                axis=(1, 2)) / np.nansum(self.intensity[wave_range: wave_range + wavelength_step]**power, axis=(1, 2))
            y_com[wave_range: wave_range + wavelength_step] = stat(y_com[wave_range: wave_range + wavelength_step])
        return x_com, y_com

    def get_integrated_light_frac(self, frac=0.5):
        """Return the cumulative-light threshold for a given fraction.

        The cube is collapsed spatially to an image by summing over wavelength.
        Pixels are sorted by brightness, and the cumulative fraction is computed.

        Parameters
        ----------
        frac : float, default=0.5
            Target cumulative fraction in ``(0, 1]``.

        Returns
        -------
        float
            Threshold value in the cumulative array at which ``frac`` is reached.

        Notes
        -----
        This method returns the cumulative statistic at the index where the target
        fraction is first exceeded. It does not return a mask.
        """
        collapsed_intensity = np.nansum(self.intensity, axis=0)
        sort_intensity = np.sort(collapsed_intensity, axis=(0, 1))
        # Sort from highes to lowest luminosity
        sort_intensity = np.flip(sort_intensity, axis=(0, 1))
        cumulative_intensity = np.cumsum(sort_intensity, axis=(0, 1))
        cumulative_intensity /= np.nanmax(cumulative_intensity)
        pos = np.searchsorted(cumulative_intensity, frac)
        return cumulative_intensity[pos]

    def get_white_image(self, wave_range=None, s_clip=3.0, frequency_density=False):
        """Create a white-light image over a wavelength interval.

        Parameters
        ----------
        wave_range : tuple of (Quantity or float, Quantity or float), optional
            Wavelength limits ``(wmin, wmax)``. Floats are interpreted in Angstrom.
            If omitted, the full spectral range is used.
        s_clip : float or None, default=3.0
            If not ``None``, perform symmetric sigma-clipping around the median
            per spaxel using a robust MAD-based standard deviation.
        frequency_density : bool, default=False
            If ``True``, convert to per-frequency density using
            ``nu = c / lambda`` (scaling by ``lambda**2 / c``).

        Returns
        -------
        astropy.units.Quantity
            2D white-light image with the same spatial shape as the cube.
        """
        if wave_range is not None:
            print("Wavelength : ", wave_range, self.wavelength)
            wave_mask = (
                self.wavelength >= ancillary.check_unit(wave_range[0], u.AA)
                ) & (
                self.wavelength <= ancillary.check_unit(wave_range[1], u.AA))
        else:
            wave_mask = np.ones(self.wavelength.size, dtype=bool)
        
        if s_clip is not None:
            std_dev = ancillary.std_from_mad(self.intensity[wave_mask], axis=0)
            median = np.nanmedian(self.intensity[wave_mask], axis=0)
            weights = (
                (self.intensity[wave_mask] <= median[np.newaxis] + s_clip * std_dev[np.newaxis])
                & (self.intensity[wave_mask] >= median[np.newaxis] - s_clip * std_dev[np.newaxis]))
        else:
            weights = np.ones(self.intensity[wave_mask].shape)

        if frequency_density:
            freq_trans = self.wavelength**2 / constants.c
        else:
            freq_trans = np.ones(self.wavelength.size)

        white_image = np.nansum(
            self.intensity[wave_mask]
            * freq_trans[wave_mask, np.newaxis, np.newaxis]
            * weights, axis=0) / np.nansum(weights, axis=0)
        return white_image

    def get_footprint(self):
        """Return the celestial WCS footprint of the datacube.

        Returns
        -------
        numpy.ndarray
            Array of world-coordinate polygon vertices as returned by
            :meth:`astropy.wcs.WCS.calc_footprint`.
        """
        return self.wcs.celestial.calc_footprint()

    def update_coordinates(self, new_coords=None, offset=None):
        """Update the celestial reference of the cube WCS.

        Parameters
        ----------
        new_coords : tuple of (Quantity, Quantity), optional
            Absolute sky coordinates ``(ra, dec)`` to set in the celestial WCS,
            both with units of degree.
        offset : tuple of (Quantity, Quantity), optional
            Offsets ``(d_ra, d_dec)`` to apply to the current celestial reference,
            in degrees. Ignored if ``new_coords`` is provided.

        Notes
        -----
        Only the celestial axes of :attr:`wcs` are modified. A history entry named
        ``'update_coords'`` is appended.
        """
        updated_wcs = ancillary.update_wcs_coords(self.wcs.celestial,
                                        ra_dec_val=new_coords,
                                        ra_dec_offset=offset)
        # Update only the celestial axes
        self.vprint(f"Previous CRVAL: {self.wcs.celestial.wcs.crval}"
                    + f"\nNew CRVAL: {updated_wcs.wcs.crval}")
        self.wcs.wcs.crval[:-1] = updated_wcs.wcs.crval
        self.history('update_coords', "Offset-coords updated")

    def to_fits(self, filename=None, overwrite=False,
                checksum=False):
        """Write the cube to a PyKOALA FITS file.

        See :meth:`DataContainer._to_hdul` for the common HDUs. An additional
        empty ``INFO`` table is appended with basic keywords.

        Parameters
        ----------
        filename : str or path-like, optional
            Output path. If omitted, a timestamped name is generated.
        overwrite : bool, default=False
            Overwrite existing file.
        checksum : bool, default=False
            Add FITS checksums.

        """
        if filename is None:
            filename = 'cube_{}_{}.fits.gz'.format(
                self.info.get("name", "frame"),
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

        hdul = self._to_hdul()
        # Fill the INFO extension
        info_header = fits.Header()
        info_header["NAME    "] = self.info.get("name", "N/A"), "Object name"
        info_header["EXPTIME "] = self.info.get("exptime", 0 << u.second).to_value(
            "second"), "exposure time (s)"
        info_header["AIRMASS "] = self.info.get("airmass"), "airmass at centre of FoV"
        hdul.append(fits.BinTableHDU(name="INFO", data=None, header=info_header))

        # Save the HDUL into a FITS file.
        hdul.verify('fix')
        hdul.writeto(filename, overwrite=overwrite, checksum=checksum)
        hdul.close()
        self.vprint(f"File saved as {filename}")

    # def close_hdul(self):
    #     """Close the underlying HDUList if present."""
    #     if self.hdul is not None:
    #         self.vprint(f"[Cube] Closing HDUL")
    #         self.hdul.close()

    @classmethod
    def from_hdul(cls, hdul):
        """Construct a :class:`Cube` from an open FITS HDU list.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            Open HDU list created by :meth:`to_fits`.

        Returns
        -------
        Cube
            New instance initialized from ``hdul``.
        """
        info = {}
        info["name"] = hdul["INFO"].header.get("name")
        info["exptime"] = hdul["INFO"].header.get("exptime")
        dc_parameters = cls._dc_params_from_hdul(hdul)
        dc_parameters["info"] = info
        return cls(**dc_parameters)

    @classmethod
    def from_fits(cls, filename):
        """Read a PyKOALA cube from a FITS file.

        Parameters
        ----------
        filename : str or path-like
            Path to a FITS file produced by :meth:`to_fits`.

        Returns
        -------
        Cube
            New instance with intensity, variance, wavelength, mask, WCS, history,
            and original header loaded.

        Notes
        -----
        Expected extensions:
        - ``PRIMARY``: history and original header
        - ``INTENSITY``: science data + WCS
        - ``VARIANCE``: variance data + WCS
        - ``MASK``: :class:`DataMask`
        - ``INFO``: optional metadata table for the cube
        """
        hdul = fits.open(filename)
        return cls.from_hdul(hdul)


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
