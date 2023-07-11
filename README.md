# Loop tracing in python.

LoopTrace is a Python package for performing procedures of chromatin tracing in Python. Current version: 0.3.

## Installation

The simplest way to install the package and all dependencies is to clone the repository and create a new environment using the 
environment.yml file. There are several optional dependencies, these can be installed separately (see below), or directly during installation by using environment_full.yml as the environment file instead. 
In a terminal (e.g. a miniconda/Anaconda prompt):

```bash
git clone https://git.embl.de/grp-ellenberg/looptrace
cd looptrace
conda env create -f environment.yml
python setup.py install
conda activate looptrace
```
There are several optional packages that can be added depending on use case:

Deconvolution of spots during tracing:
```bash
pip install flowdec[tf_gpu]
```

See also https://git.embl.de/grp-ellenberg/tracebot for fluidics control system.

## Basic usage (tracing):
First edit the config YAML file to provide input and output directories and other parameters. A documented example file is found in examples/example_tracing_processing_config.yml.
Once the config file is appropriately edited (this can be updated during the pipeline), proceed to run the CLI programs (found in bin/cli folder) in a workflow as shown in the jupyter notebooks (examples/looptrace_image_processing_example_single_computer.ipynb or looptrace_image_processing_example_HPC.ipynb for running the worflow on a (SLURM) HPC cluster).

Once the tracing is done, the data can be further analyzed for example using iPython notebooks. See examples of the full analysis as well as several full datasets at https://www.ebi.ac.uk/biostudies/studies/S-BIAD59

## Authors
Written and maintained by Kai Sandvold Beckwith (kai.beckwith@embl.de), Ellenberg group, CBB, EMBL Heidelberg.
See https://www-ellenberg.embl.de/. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation
Please cite our paper: https://www.biorxiv.org/content/10.1101/2021.04.12.439407 
