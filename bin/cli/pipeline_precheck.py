"""Validation of the main looptrace configuration file"""

import argparse
from pathlib import Path
from typing import *

import yaml

from gertils import ExtantFile

from looptrace.Deconvolver import REQ_GPU_KEY
from looptrace import Drifter, LOOPTRACE_JAR_PATH, MINIMUM_SPOT_SEPARATION_KEY, ZARR_CONVERSIONS_KEY
from looptrace.SpotPicker import DetectionMethod, CROSSTALK_SUBTRACTION_KEY, DETECTION_METHOD_KEY as SPOT_DETECTION_METHOD_KEY
from looptrace.Tracer import MASK_FITS_ERROR_MESSAGE


class ConfigFileError(Exception):
    """Base class for violations of configuration file rules."""
    pass


class ConfigFileCrash(Exception):
    """Class aggregating nonempty collection of config file errors"""
    def __init__(self, errors: Iterable[ConfigFileError]):
        super().__init__(f"{len(errors)} error(s):\n{'; '.join(map(str, errors))}")


class MissingJarError(Exception):
    """For when the project's JAR is not an extant file"""
    def __init__(self, path: Path):
        super().__init__(str(path))
        if path.is_file():
            raise ValueError(f"Alleged missing JAR is a file: {path}")


def find_config_file_errors(config_file: ExtantFile) -> List[ConfigFileError]:
    """
    Parse the given looptrace main processing configuration file, and build a collection of any errors.

    Parameters
    ----------
    config_file : ExtantFile
        Path to the main looptrace processing configuration file to parse

    Returns
    -------
    list of ConfigFileError
        A collection of violations of prohibitions, or other faults, found in the parsed config data
    """
    with open(config_file.path, 'r') as fh:
        conf_data = yaml.safe_load(fh)
    
    errors = []
    
    if not conf_data.get(ZARR_CONVERSIONS_KEY):
        errors.append(ConfigFileError(
            f"Conversion of image folders should be specified as mapping from src to dst folder, with key {ZARR_CONVERSIONS_KEY}."
            ))
    
    errors.extend([
        ConfigFileError(f"Missing or null value for key: {k}") 
        for k in ("decon_input_name", "decon_output_name") if not conf_data.get(k)
        ])
    
    if not conf_data.get(REQ_GPU_KEY, False):
        errors.append(ConfigFileError(f"Requiring GPUs for deconvolution with key {REQ_GPU_KEY} is currently required."))
    
    dc_method = Drifter.get_method_name(conf_data)
    if dc_method and not Drifter.Methods.is_valid_name(dc_method):
        errors.append(ConfigFileError(f"Invalid drift correction method ({dc_method}); choose from: {', '.join(Drifter.Methods.values())}"))
    
    if conf_data.get(CROSSTALK_SUBTRACTION_KEY, False):
        errors.append(ConfigFileError(f"Crosstalk subtraction ('{CROSSTALK_SUBTRACTION_KEY}') isn't currently supported."))
    
    spot_detection_method = conf_data.get(SPOT_DETECTION_METHOD_KEY)
    if spot_detection_method is None:
        errors.append(ConfigFileError(f"No spot detection method ('{SPOT_DETECTION_METHOD_KEY}') specified!"))
    elif spot_detection_method == DetectionMethod.INTENSITY.value:
        errors.append(ConfigFileError(f"Prohibited (or unsupported) spot detection method: '{spot_detection_method}'"))
    
    try:
        min_sep = conf_data[MINIMUM_SPOT_SEPARATION_KEY]
    except KeyError:
        errors.append(ConfigFileError(f"No minimum spot separation ('{MINIMUM_SPOT_SEPARATION_KEY}') specified!"))
    else:
        if not isinstance(min_sep, (int, float)) or min_sep < 0:
            errors.append(ConfigFileError(f"Illegal minimum spot separation ('{MINIMUM_SPOT_SEPARATION_KEY}') value: {min_sep}"))

    if conf_data.get("mask_fits", False):
        errors.append(ConfigFileError(MASK_FITS_ERROR_MESSAGE))
    
    return errors


def workflow(config_file: ExtantFile) -> None:
    if not LOOPTRACE_JAR_PATH.is_file():
        raise MissingJarError(LOOPTRACE_JAR_PATH)
    errors = find_config_file_errors(config_file=config_file)
    if errors:
        raise ConfigFileCrash(errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the provided main looptrace configuration file.")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    args = parser.parse_args()
    workflow(config_file=args.config_path)