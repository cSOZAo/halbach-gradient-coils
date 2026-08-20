# pyCoilGen runtime data

This directory is part of the vendored `pyCoilGen` engine. It is one of the
locations the engine searches for optional mesh surfaces, target fields, and
precomputed solutions.

## Use in this repository

Nothing in this directory needs to be installed separately. Follow the setup
instructions in the [project README](../README.md); the local editable package
and application automatically resolve this directory from the repository.

Add a file here only when it is reusable engine input rather than a result from
one Halbach design run. Generated coils, shells, metrics, and sweep tables belong
under `halbach_coils/resultados/`, which is ignored by Git.

The printable Halbach shell halves used by the GUI are application assets and
live in `halbach_coils/assets/shells/`, not here.

## Upstream project

`pyCoilGen` generates MRI/NMR coil layouts on three-dimensional support surfaces
using a boundary-element target-field method. Upstream documentation is
available at [pycoilgen.readthedocs.io](https://pycoilgen.readthedocs.io/) and
the upstream source is at [kev-m/pyCoilGen](https://github.com/kev-m/pyCoilGen).

This repository vendors the source so that a fresh clone uses the tested engine
version. Installing `pycoilgen` or `pycoilgen_data` globally is neither required
nor recommended for this application because it can select a different version.

## Licence

The vendored engine and this repository are covered by the GPL terms in the
repository's [LICENSE.txt](../LICENSE.txt). Preserve upstream attribution when
redistributing modified data or engine code.
