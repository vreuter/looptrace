"""Script to generate the pipeline execution control documentation"""

import argparse
import os
from pathlib import Path
import sys
from typing import *

from expression import Option
from gertils import ExtantFile, ExtantFolder

__author__ = "Vince Reuter"

from run_processing_pipeline import PIPE_NAME, SPOT_DETECTION_STAGE_NAME, TRACING_QC_STAGE_NAME, LooptracePipeline


def _parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the pipeline exceution control documentation.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("-O", "--outfile", required=True, help="Path to output file")
    return parser.parse_args(cmdl)


DOCTEXT = f"""<!--- DO NOT EDIT THIS GENERATED DOCUMENT DIRECTLY; instead, edit {os.path.basename(__file__)} --->
# Controlling pipeline execution

## Overview
The main `looptrace` processing pipeline is built with [pypiper](https://pypi.org/project/piper/).
One feature of such a pipeline is that it may be started and stopped at arbitrary points.
To do so, the start and end points must be specified by name of processing stage.

To __start__ the pipeline from a specific point, use `--start-point <stage name>`. Example:
```
python run_processing_pipeline.py \\
    --rounds-config rounds.json \\
    --params-config params.yaml \\
    --images-folder images_folder \\
    --pypiper-folder pypiper_output \\
    --start-point {SPOT_DETECTION_STAGE_NAME} \\
    --stop-before {TRACING_QC_STAGE_NAME}
```

To __stop__ the pipeline...<br>
* ...just _before_ a specific point, use `--stop-before <stage name>`.
* ...just _after_ a specific point, use `--stop-after <stage name>`.

## Rerunning the pipeline...
When experimenting with different parameter settings for one or more stages, it's common to want to rerun the pipeline from a specific point.
Before rerunning the pipeline with the appropriate `--start-point` value, take care of the following:

1. __Analysis folder__: It's wise to create a new analysis / output folder for a rerun, particularly if it corresponds to updated parameter settings.
1. __Parameters configuration file__: It's wise to create a new parameters config file for a rerun if the rerun includes updated parameter settings. 
Regardless of whether that's done, ensure that the `analysis_path` value corresponds to the output folder you'd like to use.
1. __Imaging rounds configuration file__: If a new analysis for the same experimental data affects something about the imaging rounds configuration, 
e.g. the minimum separation distance required between regional spots, you may want to create this config file anew, copying the old one and updating 
the relevant parameter(s).
1. __Pipeline (pypiper) folder__: You should create a new pypiper folder _for a rerun with new parameters_. 
This is critical since the semaphore / checkpoint files will influence pipeline control flow.
You should copy to this folder any checkpoint files of any stages upstream of the one from which you want the rerun to begin.
Even though `--start-point` should allow the rerun to begin from where's desired, if that's forgotten the checkpoint files should save you.
For a _restart_--with the same parameters--of a stopped/halted/failed pipeline run, though, you should generally reuse the same pypiper folder as before. 

Generate an empty checkpoint file for each you'd like to skip. 
Simply create (`touch`) each such file `{PIPE_NAME}_<stage>.checkpoint` in the desired pypiper output folder.
Below are the sequential pipeline stage names.

### Pipeline stage names
"""


def get_stage_predecessors_text(stage: str, preds: List[str]) -> List[str]:
    return [f"### ...`{stage}`"] + [f"* {name}" for name in preds]


def main(cmdl):
    opts = _parse_cmdl(cmdl)
    temporary_pipeline = LooptracePipeline(
        rounds_config=ExtantFile(Path.cwd() / "shell.nix"), # dummy, just to get ExtantFile 
        params_config=ExtantFile(Path.cwd() / "shell.nix"), # dummy, just to get ExtantFile
        signal_config=Option.Nothing(),
        images_folder=ExtantFolder(Path.cwd()),
        output_folder=ExtantFolder(Path.cwd()), 
    )
    stage_names = [stage.name for stage in temporary_pipeline.stages()]
    full_text = DOCTEXT + "\n".join(f"* {sn}" for sn in stage_names)
    print(f"Writing docs file: {opts.outfile}")
    with open(opts.outfile, 'w') as out:
        out.write(full_text)
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
