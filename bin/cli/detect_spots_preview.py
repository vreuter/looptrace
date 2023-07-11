"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preview spot detection in given position.')
    parser.add_argument("--config_path", help="Config file path")
    parser.add_argument("--image_path", help="Path to folder with images to read.")
    parser.add_argument('--position', help='(Optional): Index of position to view.', default='0')

    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, pos_id=int(args.position))
    S = SpotPicker(H)
    S.rois_from_spots(preview_pos=int(args.position))