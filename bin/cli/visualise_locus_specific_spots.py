"""Visualisation of retained (after QC filters) locus-specific FISH spots."""

import argparse
from typing import *

import napari
import numpy as np
import pandas as pd

from gertils import ExtantFile, ExtantFolder
from looptrace import RoiImageSize, read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.image_io import multi_ome_zarr_to_dask
from looptrace.napari_helpers import \
    SIGNAL_TO_QUIT, add_points_to_viewer, prompt_continue_napari, shutdown_napari

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

# TODO: feature ideas -- see https://github.com/gerlichlab/looptrace/issues/259
# Include annotation and/or coloring based on the skipped reason(s).

POSITION_COLUMN = "position"
ROI_NUMBER_COLUMN = "roi_number"
FRAME_COLUMN = "frame"
QC_PASS_COLUMN = "qcPass"
COORDINATE_COLUMNS = ["z_px", "y_px", "x_px"]


def workflow(config_file: ExtantFile, images_folder: ExtantFolder):
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    T = Tracer(H)
    extra_columns = [POSITION_COLUMN, ROI_NUMBER_COLUMN, FRAME_COLUMN, QC_PASS_COLUMN]
    print(f"Reading ROIs file: {H.traces_file_qc_unfiltered}")
    # NB: we do NOT use the drift-corrected pixel values here, since we're interested 
    #     in placing each point within its own ROI, not relative to some other ROI.
    # TODO: we need to add the columns for frame and ROI ID / ROI number to the 
    #       list of what we pull, because we need these to add to the points coordinates.
    point_table = read_table_pandas(H.traces_file_qc_unfiltered)
    if POSITION_COLUMN not in point_table:
        # TODO -- See: https://github.com/gerlichlab/looptrace/issues/261
        print(f"DEBUG -- Column '{POSITION_COLUMN}' is not in the spots table parsed in package-standard pandas fashion")
        print(f"DEBUG -- Retrying the spots table parse while assuming no index: {H.traces_file_qc_unfiltered}")
        point_table = pd.read_csv(H.traces_file_qc_unfiltered, index_col=None)[COORDINATE_COLUMNS + extra_columns]
    data_path = T.all_spot_images_zarr_root_path
    print(f"INFO -- Reading image data: {data_path}")
    images, positions = multi_ome_zarr_to_dask(data_path)
    for img, pos in zip(images, positions):
        _, num_times, _, _, _ = img.shape
        cur_pts_tab = point_table[point_table.position == pos]
        points_data = compute_points(cur_pts_tab, num_times=num_times, roi_size=H.roi_image_size)
        visibilities, point_symbols, qc_passes, points = zip(*points_data)
        viewer = napari.view_image(img)
        add_points_to_viewer(
            viewer=viewer, 
            # TODO: use info about frame and ROI ID / ROI number to put each point in the proper 3D array (z, y, x).
            points=points, 
            properties={QC_PASS_COLUMN: qc_passes},
            size=1,
            shown=visibilities,
            symbol=point_symbols,
            edge_color="transparent",
            face_color=QC_PASS_COLUMN,
            face_color_cycle=["blue", "red"], 
            )
        napari.run()
        if prompt_continue_napari() == SIGNAL_TO_QUIT:
            break
        # Here we don't call viewer.close() programmatically since it's expected that the user closes the window.
        print("DEBUG: Continuing...")
    shutdown_napari()


def compute_points(cur_pts_tab, *, num_times: int, roi_size: RoiImageSize):
    bad_shape = "o"
    points_data = []
    for roi_idx, roi_group in cur_pts_tab.groupby(ROI_NUMBER_COLUMN):
        lookup = {row[FRAME_COLUMN]: (row[QC_PASS_COLUMN], row[COORDINATE_COLUMNS].to_numpy()) for _, row in roi_group.iterrows()}
        for t in range(num_times):
            try:
                qc_pass, coords = lookup[t]
            except KeyError:
                coords = np.zeros(3)
                visible = False
                point_shape = bad_shape
                qc_pass = False
            else:
                visible = True
                qc_pass = bool(qc_pass)
                point_shape = "star" if qc_pass else "hbar"
            if coords[0] < 0 or coords[0] > roi_size.z or coords[1] < 0 or coords[1] > roi_size.y or coords[2] < 0 or coords[2] > roi_size.x:
                if qc_pass:
                    print(f"WARN -- spot point passed QC! {coords}")
                coords = np.array([0, 0, 0])
                visible = False
                point_shape = bad_shape
            point = np.concatenate(([roi_idx, t], coords)).astype(np.float32)
            points_data.append((visible, point_shape, qc_pass, point))
    return points_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation of retained (after QC filters) locus-specific FISH spots", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path)