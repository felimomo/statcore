# Project notepad

## Big-picture conceptual stuff

## Benchmarks

## Failure / bug log

## Chronological Notes

### 28 Sept. 2026

- conda-forge [compilers](https://anaconda.org/channels/conda-forge/packages/compilers/overview) installs C++ & FORTRAN compilers compatible with other conda-forge dependencies. If I ever want this to be conda-installable, I should list the dependency using a "Jinja template function", that is, the dependency line in the conda environment.yml file should be something like `- {{ compiler('cxx') }}`.