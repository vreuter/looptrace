"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nucleus detection on images.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", help="(Optional): Path to folder to save images to.", default=None)
    args = parser.parse_args()
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, image_save_path=args.image_save_path, pos_id = array_id)
    N = NucDetector(H)
    N.segment_nuclei()
    H.read_images()
    N.classify_nuclei()