## Command-line programs
This folder contains some programs which are run as part of the main [processing pipeline](./run_processing_pipeline.py). 
It also contains some utility programs which may be helpful in general or when something goes awry with the pipeline and diagnostics or remediative measures are needed. 

Here is a (hopefully complete and up-to-date, but nop guarantees) list of the ancillary/utility programs:
* `analyse_bead_discard_reasons.py`, [a command-line program](./analyse_detected_bead_rois.py) to examine aggregation of failure reason(s) for fiducial bead ROIs, i.e. why certain beads were discarded and ineligible for use for drift correction. This program also provides suggestions about which FOVs will be impossible to make have a sufficient number of beads for drift correction (i.e., which FOVs should be removed from the analysis), and how to adjust (downward) the minimum threshold for a bead ROI's intensity needed to make it eligble for drift correction use.
