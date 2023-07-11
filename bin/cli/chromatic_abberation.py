"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.Drifter import Drifter
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nucleus detection on images.')
    parser.add_argument("--config_path", help="Config file path")
    parser.add_argument("--image_path", help="Path to folder with images to read.")
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    D = Drifter(H)
    D.chrom_shift()