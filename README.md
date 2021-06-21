# APOGEE Net II

The package `apogeenet2` makes predictions of a star's temperature, surface
gravity, and metallicity from spectra.

## Usage

To use the pipeline after installing, run `apogeenet2` on the command line. By
default, the program searches the current directory for the .fits table, uses
the current directory to search for the spectra listed in the table, then writes
its predictions to `predictions.csv` in the current directory.

Alternatively, the program can be used and imported as a package. Inside
`apogeenet2.pipeline`, the `Pipeline` object can be found. The method
`Pipeline.predict()` does most of the work, feeding data through the model and
writing predictions out.

### Options

For help and further options, run `apogeenet2 --help`.

## Installation

To install, run `pip install apogeenet2`. This package requires Python 3.6 or
newer.

## Paper

A paper detailing the significance and architecture of this project is coming
soon.

The summary is that this package uses a CNN to process the spectra, obtaining
previously-unseen results on high-temperature stars.

## Maintenance and Bug Reporting

This package is unmaintained and provided as-is. However, bugs can still be
filed under the "Issues" tab of the project's GitHub page
