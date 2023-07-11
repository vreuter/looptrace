
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("--config_path", help="Config file path")
    parser.add_argument("--image_path", help="Path to folder with images to read.")
    args = parser.parse_args()
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, pos_id = array_id)
    T = Tracer(H)
    T.trace_all_rois()