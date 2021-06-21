#! /usr/bin/env/python3

"""A pipeline for APOGEE Net II.

This module contains the Pipeline object, which is the main tool to feed spectra
through the APOGEE Net II model. The corresponding predictions for the spectra
are then written to their output location (either a CSV or stdout). Also 
included is the argument parsing and main method used to run the pipeline 
standalone through the command line.

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

import argparse
import os
import io
import sys
from typing import Optional, Tuple
import warnings
import numpy as np

import torch

from apogeenet2.dataset import AP2_Dataset as Dataset
from apogeenet2.model import Model


class Pipeline():
    """A pipeline for APOGEE Net II that feeds in .fits files.

    The predict() function does much of the heavy lifting in most use cases
    where this object is just used to get predictions from the APOGEE Net II
    model.

    Attributes:
        device:
            A torch.device for which device to use with the model.
        use_file:
            A boolean that is true if output is written to a file.
        model:
            An apogeenet2.Model instance.
        dataset:
            An apogeenet2.AP2_Dataset instance.
        output_path:
            A string for the path to the output file, or "stdout".
        uncertainty_count:
            An int for the number of uncertainties to compute per spectra.
    """

    def __init__(self, input: str, output: str = "predictions.csv", fits_table: str ="allField-dr17.fits",
                 uncertainty_count: int = 0, star: Optional[str] = None) -> None:
        """Initializes the Pipeline object, creating the model and dataset too.

        Args:
            input: 
                The root directory containing the spectra.
            output:
                Optional; The path for the output CSV, or "stdout" if the
                predictions are printed to terminal.
            fits_table:
                Optional; A path to the .fits table containing the data about
                the spectra to be processed.
            uncertainty_count:
                Optional; The number of uncertainties to process per spectra.
            star:
                Optional; The index in the .fits table or APOGEE ID of a star in
                the table to process, with only that star being processed.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_file: bool = output != "stdout"
        self.model: Model = Model(self.device)
        self.dataset: Dataset = Dataset(input, fits_table, star)
        self.output_path: str = output
        self.uncertainty_count: int = uncertainty_count
        self._do_uncertainties = uncertainty_count > 0

    def predict(self) -> None:
        """Creates predictions for spectra in the fits table, and writes them.

        This function does a lot of the work of the class, both feeding the
        predictions through the model and writing them.
        """

        self._write_header()
        f = open(self.output_path, "a+") if self.use_file else sys.stdout
        with torch.set_grad_enabled(False):
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=None, collate_fn=_converter, num_workers=6)
            for id, spectra, mdata, error in loader:
                spectra, mdata, error = spectra.to(self.device), mdata.to(self.device), error.to(self.device)
                predictions = self.model.predict_spectra(spectra, mdata)
                uncertainties = self._calc_uncertainties(spectra, error, mdata) if self._do_uncertainties else tuple()
                self._write_prediction(f, id, predictions, *uncertainties)
        if self.use_file:
            f.close()

    def _calc_uncertainties(
        self, spectra: torch.Tensor, error: torch.Tensor, metadata: torch.Tensor
    ) -> Tuple[float, float, float, float, float, float]:
        """Handles and processes uncertainties.

        Uses the error tensor for a spectra to randomly generate 
        self.uncertainty_count samples within the margin of error. The samples
        are then fed through the model to return statistics about the certainty
        of the model's prediction for the spectra.

        Args:
            spectra:
                The flux data for a star.
            error:
                The error data for an imaging.
            mdata:
                The metadata corresponding to the spectra.

        Returns:
            The median logg value for the uncertainties, the standard deviation
            of logg for the uncertainties, the median logteff value for the 
            uncertainties, the standard deviation of togteff for the
            uncertainties, the median FeH value for the uncertainties, and the 
            standard deviation of FeH for the uncertainties.
        """

        median_error = 5 * torch.median(error)
        error = torch.where(error == 1.0000e+10, spectra, error)
        error = torch.where(error < median_error, error, median_error)
        inputs = torch.randn((self.uncertainty_count, *error.shape[1:]), device=self.device) * error + spectra
        metadata = metadata.repeat(self.uncertainty_count, 1)
        outputs = self.model.predict_spectra(inputs, metadata)
        logg_median, logg_std = get_median_and_stdev(outputs[:, 0])
        logteff_median, logteff_std = get_median_and_stdev(outputs[:, 1])
        feh_median, feh_std = get_median_and_stdev(outputs[:, 2])
        return (logg_median, logg_std, logteff_median, logteff_std, feh_median, feh_std)

    def _write_header(self) -> None:
        """Writes the appropriate CSV-style header for the predictions.

        Writes to stdout if self.output_path is "stdout", otherwise writes to a
        CSV.
        """

        f = open(self.output_path, 'w') if self.use_file else sys.stdout
        if self._do_uncertainties:
            f.write("APOGEE_ID,logg,logTeff,FeH,logg_median,logg_std,logTeff_median,logTeff_std,FeH_median,FeH_std\n")
        else:
            f.write("APOGEE_ID,logg,logTeff,FeH\n")
        if self.use_file:
            f.close()

    def _write_prediction(
        self, f: io.TextIOWrapper, id: str, preds: torch.Tensor, logg_median: float = 0, logg_std: float = 0,
        logteff_median: float = 0, logteff_std: float = 0, feh_median: float = 0, feh_std = 0) -> None:
        """Writes a single spectra's prediction out.
        
        Args:
            f:
                The IO being written to.
            id:
                The APOGEE ID of the prediction.
            preds:
                A (1, 3) tensor containing predicted [logg, logteff, FeH].
            logg_median:
                Optional; The median logg for the uncertainties.
            logg_std:
                Optional; The standard deviation of logg for the uncertainties.
            logteff_median:
                Optional; The median logteff for the uncertainties.
            logteff_std:
                Optional; The standard deviation of logteff for the uncertainties.
            feh_median:
                Optional; The median FeH for the uncertainties.
            feh_std:
                Optional; The standard deviation of FeH for the uncertainties.
        """

        preds = preds[0]
        logg, logteff, feh = preds
        if self._do_uncertainties:
            f.write((f"{id},{logg},{logteff},{feh},{logg_median},{logg_std},{logteff_median},{logteff_std},"
                        f"{feh_median},{feh_std}\n"))
        else:
            f.write(f"{id},{logg},{logteff},{feh}\n")


def _converter(data: np.ndarray) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The collate_fn for the Dataloader.

    Returns:
        The APOGEE ID, a (1, 1, 8575) tensor of the flux spectra, a (1, 7) 
        tensor of the metadata, and a (1, 1, 8575) tensor of the error for the 
        flux spectra.
    """

    spectra = torch.reshape(torch.from_numpy(data[1]), [1, 1, -1])
    metadata = torch.reshape(torch.from_numpy(data[2]), [1, 7])
    error = torch.reshape(torch.from_numpy(data[3]), [1, 1, -1])
    return data[0], spectra, metadata, error

def get_median_and_stdev(arr: torch.Tensor) -> Tuple[float, float]:
    """Returns the median and standard deviation from a tensor."""

    return torch.median(arr).item(), torch.std(arr).item()


# ===== Pipeline as a program functionality =====


def main() -> None:
    """The main method to run the pipeline stand-alone."""

    args = parse_all_args()
    Pipeline(args.input_directory, args.output, args.fits_table, args.uncertainty_count, args.star).predict()


def parse_all_args() -> argparse.Namespace: 
    """Parses the arguments of the file.

    Returns:
        An argparse Namespace containing the parsed arguments. 
    """

    parser = argparse.ArgumentParser()
    input_default = os.getcwd()
    parser.add_argument("--input_directory", type=str, default=input_default,
                        help=("The input directory containing the spectra .fits files. "
                                f"[Default: The current working directory, which is currently '{input_default}']"))
    output_default = "predictions.csv"
    parser.add_argument("--output", type=str, default=output_default,
                        help=("The output CSV filepath for the predictions. Use 'stdout' to print the output."
                                f" [Default: '{output_default}']"))
    fits_default = "allField-dr17.fits"
    parser.add_argument("--fits_table", type=str, default=fits_default,
                        help=('The name of a .fits file in input_directory containing a table with metadata about '
                                f"the spectra. Spectra processed are based on table. [Default: '{fits_default}']"))
    parser.add_argument("--uncertainty_count", type=int, default=0,
                        help="The number of uncertainties to calculate in addition to the prediction. [Default: 0]")
    parser.add_argument("--star", type=str, default=None,
                        help=("The only star to run predictions on. Can be either a row index into the .fits "
                                "table, or an APOGEE ID. [Default: None]"))

    return parser.parse_args()


if __name__ == "__main__":
    main()
