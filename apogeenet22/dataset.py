"""The dataset used for the APOGEE Net 2 model.

This file includes the AP2_Dataset object and any relevant resources.

MIT License

Copyright (c) 2021 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
from typing import Optional, Tuple

from astropy.io import fits
import numpy as np
import torch


_FLUX_LENGTH = 8575


class AP2_Dataset(torch.utils.data.Dataset):
    """A PyTorch dataset for loading spectra from a .fits table.

    This supports loading metadata and image-data paths from a .fits table with
    the correct columns. It can then return the APOGEE ID, flux spectra,
    metadata, and error in flux imaging for the stars in the table.

    The .fits table is assumed to have the following columns:
    - "APOGEE_ID": The APOGEE ID for each star.
    - "TELESCOPE": The telescope the spectra was captured with.
    - "FIELD": The field the spectra was captured in.
    - "FILE": The name of the file containing the spectra.
    - "GAIAEDR3_PARALLAX": The parallax of the spectra capture.
    - "GAIAEDR3_PHOT_G_MEAN_MAG": The wide optical magnitude for the spectra 
    capture.
    - "GAIAEDR3_PHOT_BP_MEAN_MAG": The blue part magnitude for the spectra
    capture.
    - "GAIAEDR3_PHOT_RP_MEAN_MAG": The red part magnitude for the spectra
    capture.
    - "J": The ~1.1 micron filter used for the spectra capture.
    - "H": The ~1.6 micron filter used for the spectra capture.
    - "K": The ~2.1 micron filter used for the spectra capture.
    For more comprehensive explanations of J, H, and K, see here: https://en.wikipedia.org/wiki/Photometric_system.

    Attributes:
        input:
            A string with the root path to the spectra referenced in the .fits
            table. The path {input}/{table.TELESCOPE}/{table.FIELD}/{table.FILE}
            should yield a .fits file containing the spectra, following this
            format: https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/stars/TELESCOPE/FIELD/apStar.html.
            This should be a directory.
        fits_info:
            A (FITS_TABLE_HEIGHT - BAD_ENTRIES, 4) NumPy array containing the
            APOGEE IDs in column 0, the telescope in column 1, the field in
            column 2, and the file in column 3.
        metadata:
            A (FITS_TABLE_HEIGHT - BAD_ENTRIES, 7) NumPy array containing the
            normalized metadata to the spectra. The columns are arranged in this
            order: [parallax, G, BP, RP, J, H, K].
    """

    _mdata_replacements = np.array([-84.82700,21.40844,24.53892,20.26276,18.43900,24.00000,17.02500])
    _mdata_stddevs = np.array([14.572430555504504,2.2762944923233883,2.8342029214199704,2.136884367623457,
                                1.6793628207779732,1.4888102872755238,1.5848713221149886])
    _mdata_means = np.array([-0.6959113178296891,13.630030428758845,14.5224418320574,12.832448427460813,
                                11.537019017423619,10.858717523536697,10.702106344460235])

    def __init__(self, input: str, fits_table: str = "allField-dr17.fits", star: Optional[str] = None) -> None:
        """Initializes the dataset, pre-processing the table.

        Args:
            input:
                The root directory for the .fits star data files.
            fits_table:
                Optional; The path to the .fits table containing the stars to be
                processed, as described in the class documentation.
            star:
                Optional; The ID or row-index in the .fits table of the only
                star to contain in the dataset.
        """

        self.input: str = input
        self.fits_info, self.metadata = self.process_table(fits_table, star)

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the id, flux spectra, metadata, and error for a star.

        Args:
            See torch.utils.data.Dataset.

        Returns:
            The APOGEE ID of the star at index, the flux spectra with shape 
            (1, 8575), the normalized metadata with shape (1, 7), and the error
            for the flux spectra with shape (1, 8575).    
        """

        metadata = self.metadata[index]
        info = self.fits_info[index]
        id = info[0]
        path = os.path.join(self.input, info[1], info[2], info[3])
        with fits.open(path) as f:
            spectra = self._get_spectral_array(f[1].data)
            error = self._get_spectral_array(f[2].data)
        return id, spectra, metadata, error

    def _get_spectral_array(self, arr: np.ndarray) -> np.ndarray:
        """Properly formats an array from a .fits file for an individual star.

        Returns:
            A (1, 8575) np.float32 array with no NaNs.
        """

        if len(arr.shape) != 1:
            arr = arr[0]
        arr = np.nan_to_num(arr, copy=False)
        return np.reshape(arr.astype(np.float32), [1, _FLUX_LENGTH])

    def __len__(self) -> int:
        """Returns the length of the dataset."""

        return self.fits_info.shape[0]

    def process_table(self, fits_table: str, star: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and pre-processes a .fits table.

        Args:
            fits_table:
                The path to a .fits table, as described in the class' docstring.
            star:
                Optional; The ID or row-index in the .fits table of the only
                star to contain in the dataset.

        Returns:
            The arrays following the rules set out in the class' docstring for 
            self.fits_info and self.metadata, respectively.
        """

        with fits.open(os.path.join(self.input, fits_table)) as f:
            table = f[1].data
            columnize = lambda arr: arr[:, np.newaxis]
            files = columnize(table['FILE'])
            ids = columnize(table['APOGEE_ID'])
            telescopes = columnize(table['TELESCOPE'])
            fields = columnize(table['FIELD'])
            Js = columnize(table['J'])
            Hs = columnize(table['H'])
            Ks = columnize(table['K'])
            parallaxes = columnize(table['GAIAEDR3_PARALLAX'])
            Gs = columnize(table['GAIAEDR3_PHOT_G_MEAN_MAG'])
            BPs = columnize(table['GAIAEDR3_PHOT_BP_MEAN_MAG'])
            RPs = columnize(table['GAIAEDR3_PHOT_RP_MEAN_MAG'])
        fits_info = np.hstack((ids, telescopes, fields, files))
        metadata = np.hstack((parallaxes, Gs, BPs, RPs, Js, Hs, Ks))
        metadata = self.normalize_metadata(metadata).astype(np.float32)
        if star is not None:
            viable_rows = int(star) if star.isnumeric() else fits_info[:, 0] == star
        else:
            viable_rows = np.all(fits_info != "", axis=1)
        return fits_info[viable_rows].reshape(-1, 4), metadata[viable_rows].reshape(-1, 7)

    def normalize_metadata(self, metadata: np.ndarray) -> np.ndarray:
        """Normalizes the metadata and fixes bad values.

        The normalization is done with standard z-score normalization using
        the means and standard deviations used to train the model.

        Non-finite values and values that are too high are replaced with
        appropriate placeholder values.

        Args:
            metadata:
                A (FITS_TABLE_HEIGHT, 4) array containing unnormalized metadata.
        
        Returns:
            The normalized metadata.
        """

        metadata = np.where(metadata < 98, metadata, self._mdata_replacements)
        metadata = np.where(np.isfinite(metadata), metadata, self._mdata_replacements)
        return (metadata - self._mdata_means) / self._mdata_stddevs
