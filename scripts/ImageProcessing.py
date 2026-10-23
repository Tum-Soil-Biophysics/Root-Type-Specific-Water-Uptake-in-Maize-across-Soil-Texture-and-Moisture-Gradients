r"""
Process neutron radiography scans:
- Read OB/DC reference beams (FITS)
- Normalize raw images, median filter, stitch into full scans
- Read segmented root image tiles, stitch, label roots, compute skeleton + diameter
- Compute CD2O along roots over time
- Export plots, Excel, and PDF of figures

To run (from the repository folder; all paths are arguments, nothing is hard-coded):
    python scripts/ImageProcessing.py \
        --input-dir      Data/day_D2O/s_d_2/D2O_compartment2 \
        --dir-ob         Data/reference_beams/ob_d2t2315_N \
        --dir-dc         Data/reference_beams/dc_d2t2310_N \
        --dir-color-code Data
    Results are written to <input-dir>/FinalResults (or to --output-dir if given).
"""

#%%
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import matplotlib as mpl
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, median_filter
from skimage.morphology import skeletonize

warnings.simplefilter("ignore", category=UserWarning)


# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class Config:
    input_dir: Path
    dir_ob: Path
    dir_dc: Path
    dir_color_code: Path
    output_dir: Path

    output_folder_name: str = "FinalResults"

    beamline: str = "NEUTRA"  # 'ICON' or 'NEUTRA'
    pixel_size_cm: float = 0.0083  # [cm/px]
    soil_thickness_cm: float = 1.0  # [cm]

    mu_h2o: float = 2.42  # [1/cm]; ICON: 3.18; NEUTRA: 2.42
    mu_d2o: float = 0.55  # [1/cm]; ICON: 0.61; NEUTRA: 0.55
    mu_soil_al: float = 0.18  # [1/cm]; NEUTRA Sand: 0.18, Loam: 0.33

    nr_grid_x: int = 3
    nr_grid_y: int = 2
    scan_direction_x: str = "Right2Left"  # 'Right2Left', 'Left2Right'
    scan_direction_y: str = "Up2Low"  # 'Up2Low', 'Low2Up'
    overlapped_fraction_pct: float = 60.0  # [%]

    nr_edge_y: int = 350
    nr_edge_x: int = 350

    # Root labeling / filtering
    min_root_component_size_px: int = 200


# ---------------------------
# Plot settings 
# ---------------------------

def setup_matplotlib() -> None:
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["figure.figsize"] = (8, 11)
    mpl.rcParams["lines.markersize"] = 14
    mpl.rcParams["lines.linewidth"] = 6
    plt.close("all")


# ---------------------------
# IO helpers
# ---------------------------

def tif_read(path: Path) -> np.ndarray:
    """Read a TIFF file into a numpy array."""
    return np.array(tiff.imread(str(path)))


def seg_read(path: Path) -> np.ndarray:
    """Read segmented RGB image and convert to gray uint8."""
    im = np.array(tiff.imread(str(path)))
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)


def read_fits_image(
    path: Path,
    edge_pos: Sequence[int],
    flip_vertical: bool = True,
) -> np.ndarray:
    """Read FITS image, crop by edge_pos, optionally flip vertically, and remove first row/col."""
    with fits.open(str(path)) as hdul:
        hdul.verify("fix")
        tem = np.array(hdul[0].data[edge_pos[0]:edge_pos[1], edge_pos[2]:edge_pos[3]])
    if flip_vertical:
        tem = cv2.flip(tem, 0)
    tem = tem[1:, 1:]
    return tem


def read_beam_current(header, beamline: str) -> float:
    """Extract beam current from FITS header (kept compatible with your original slicing)."""
    if beamline == "ICON":
        return float(header["detector_phys/beam_current"][2:-1])
    if beamline == "NEUTRA":
        return float(header["detector/beam_current"][2:-1])
    raise ValueError(f"Unknown beamline: {beamline}")


def parse_date_taken_to_epoch_seconds(date_str: str) -> float:
    """Parse FITS DATE field and return epoch seconds."""
    # Original format: "%Y-%m-%dT%H:%M:%S"
    return float(time.mktime(dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").timetuple()))


# ---------------------------
# Visualization
# ---------------------------

def show_image(
    im: np.ndarray,
    fig_name: Optional[str] = None,
    roi: Optional[Sequence[int]] = None,
    colorbar_label: Optional[str] = None,
) -> None:
    """
    Visualize a 2D image with contrast based on percentiles and a properly sized colorbar.
    Keeps your behaviour (roi currently unused for cropping, but kept as parameter).
    """
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(num=fig_name)
    low_pct = 0.35
    high_pct = 99.65

    im_roi = im  
    im_flat = im_roi[np.isfinite(im_roi)]
    vmin = np.percentile(im_flat, low_pct)
    vmax = np.percentile(im_flat, high_pct)

    cmap = mpl.colormaps["gray_r"] if roi is None else mpl.colormaps["gray"]
    im_display = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im_display, cax=cax)

    if colorbar_label:
        cbar.set_label(colorbar_label)

    tick_vals = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_vals)
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in tick_vals])

    cbar.set_label(colorbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Stitching
# ---------------------------

def overlap_finder_horizontal(img_pair: Sequence[np.ndarray], scan_direction_x: str, overlapped_fraction_pct: float) -> int:
    """Find overlap pixels in X direction using mean absolute difference."""
    shift = int(img_pair[0].shape[1] * overlapped_fraction_pct / 100.0)
    diffs: List[float] = []

    if scan_direction_x == "Right2Left":
        for i in range(1, shift):
            diffs.append(float(np.mean(np.abs(img_pair[0][:, :i] - img_pair[1][:, -i:]))))
    elif scan_direction_x == "Left2Right":
        for i in range(1, shift):
            diffs.append(float(np.mean(np.abs(img_pair[0][:, -i:] - img_pair[1][:, :i]))))
    else:
        raise ValueError(f"Unknown scan_direction_x: {scan_direction_x}")

    x_overlap = int(np.argmin(diffs[1:]) + 1)
    return x_overlap


def overlap_finder_vertical(img_pair: Sequence[np.ndarray], scan_direction_y: str, overlapped_fraction_pct: float) -> int:
    """Find overlap pixels in Y direction using mean absolute difference."""
    shift = int(img_pair[0].shape[0] * overlapped_fraction_pct / 100.0)
    diffs: List[float] = []

    if scan_direction_y == "Low2Up":
        for i in range(1, shift):
            diffs.append(float(np.nanmean(np.abs(img_pair[0][:i, :] - img_pair[1][-i:, :]))))
    elif scan_direction_y == "Up2Low":
        for i in range(1, shift):
            diffs.append(float(np.nanmean(np.abs(img_pair[0][-i:, :] - img_pair[1][:i, :]))))
    else:
        raise ValueError(f"Unknown scan_direction_y: {scan_direction_y}")

    y_overlap = int(np.argmin(diffs[1:]) + 1)
    return y_overlap


def stitch_vertically(img_pair: Sequence[np.ndarray], nr_grid_y: int, scan_direction_y: str, y_overlap: int) -> np.ndarray:
    if nr_grid_y <= 1:
        return img_pair[0]

    if scan_direction_y == "Low2Up":
        return np.concatenate((img_pair[1][:-y_overlap, :], img_pair[0][:, :]))
    if scan_direction_y == "Up2Low":
        return np.concatenate((img_pair[0][:-y_overlap, :], img_pair[1][:, :]))

    raise ValueError(f"Unknown scan_direction_y: {scan_direction_y}")


def stitch_horizontally(imgs: Sequence[np.ndarray], nr_grid_x: int, scan_direction_x: str, x_overlap: int) -> np.ndarray:
    """
    2 or 3 image stitching logic.
    """
    if nr_grid_x <= 1:
        return imgs[0]

    if scan_direction_x == "Right2Left" and nr_grid_x in (2, 3):
        tem = np.concatenate((imgs[1], imgs[0][:, x_overlap:]), axis=1)
        if nr_grid_x == 3:
            tem = np.concatenate((imgs[2], tem[:, x_overlap:]), axis=1)
        return tem

    if scan_direction_x == "Left2Right" and nr_grid_x in (2, 3):
        tem = np.concatenate((imgs[0], imgs[1][x_overlap:, :]), axis=1)
        if nr_grid_x == 3:
            tem = np.concatenate((tem, imgs[1][x_overlap:, :]), axis=1)
        return tem

    raise ValueError(f"Unsupported combination: nr_grid_x={nr_grid_x}, scan_direction_x={scan_direction_x}")


def montage(
    img_tiles: Sequence[np.ndarray],
    nr_grid_x: int,
    nr_grid_y: int,
    scan_direction_x: str,
    scan_direction_y: str,
    x_overlap: Optional[int],
    y_overlap: Optional[int],
) -> np.ndarray:
    if nr_grid_x == 1 and nr_grid_y == 1:
        return img_tiles[0] if isinstance(img_tiles, list) else img_tiles  # defensive

    if nr_grid_x > 1 and nr_grid_y == 1:
        assert x_overlap is not None
        return stitch_horizontally(img_tiles[:nr_grid_x], nr_grid_x, scan_direction_x, x_overlap)

    if nr_grid_y > 1 and nr_grid_x == 1:
        assert y_overlap is not None
        return stitch_vertically(img_tiles[:nr_grid_y], nr_grid_y, scan_direction_y, y_overlap)

    if nr_grid_x > 1 and nr_grid_y == 2:
        assert x_overlap is not None and y_overlap is not None
        row1 = stitch_horizontally(img_tiles[:nr_grid_x], nr_grid_x, scan_direction_x, x_overlap)
        row2 = stitch_horizontally(img_tiles[nr_grid_x:], nr_grid_x, scan_direction_x, x_overlap)
        return stitch_vertically([row1, row2], nr_grid_y, scan_direction_y, y_overlap)

    if nr_grid_x == 2 and nr_grid_y == 3:
        assert x_overlap is not None and y_overlap is not None
        row1 = stitch_horizontally(img_tiles[:2], 2, scan_direction_x, x_overlap)
        row2 = stitch_horizontally(img_tiles[2:4], 2, scan_direction_x, x_overlap)
        row3 = stitch_horizontally(img_tiles[4:], 2, scan_direction_x, x_overlap)
        return stitch_vertically([row1, row2, row3], nr_grid_y, scan_direction_y, y_overlap)

    raise ValueError("Montage case not covered by this script (e.g. 4 pictures in Y).")


# ---------------------------
# Special y-shift correction (kept)
# ---------------------------

def find_y_shift(img_original: np.ndarray, img_shifted: np.ndarray) -> int:
    shift = int(img_original.shape[0] * 0.1)
    diffs: List[float] = []
    for i in range(1, shift):
        diffs.append(float(np.nanmean(np.abs(img_shifted[:-i, :] - img_original[i:, :]))))
    return int(np.argmin(diffs[1:]) + 1)


def fix_y_shift(img_original: np.ndarray, img_shifted: np.ndarray, y_shift: int) -> np.ndarray:
    cut_image = img_shifted[:-y_shift, :]
    corrected = np.full((img_original.shape[0], img_shifted.shape[1]), np.nan)
    corrected[y_shift:, :] = cut_image
    return corrected


# ---------------------------
# Export helpers
# ---------------------------

def save_figures_to_pdf(file_path: Path) -> None:
    """Save all open Matplotlib figures to a single PDF, roughly fitted to A4."""
    a4_width = 8.27
    a4_height = 11.69
    aspect_a4 = a4_width / a4_height

    with pdf_backend.PdfPages(str(file_path)) as pdf:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            w, h = fig.get_size_inches()
            if w > h:  # landscape
                target_width = a4_height
                target_height = target_width / aspect_a4
                if target_height > a4_height:
                    target_height = a4_width
                    target_width = h * aspect_a4
            else:  # portrait
                target_height = a4_height
                target_width = target_height * aspect_a4
                if target_width > a4_width:
                    target_width = a4_width
                    target_height = target_width / aspect_a4

            fig.set_size_inches(target_width, target_height)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight", dpi=300)


# ---------------------------
# Main pipeline
# ---------------------------

def run(cfg: Config) -> None:
    logger = logging.getLogger("process_scan")

    output_folder = cfg.output_dir / cfg.output_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    edge_pos = [cfg.nr_edge_y, -cfg.nr_edge_y, cfg.nr_edge_x, -cfg.nr_edge_x]
    number_positions = cfg.nr_grid_x * cfg.nr_grid_y

    logger.info("Reading OB references from %s", cfg.dir_ob)
    ob_files = sorted(cfg.dir_ob.glob("*.fits"))
    ob_stack: List[np.ndarray] = []
    ob_beams: List[float] = []

    for p in ob_files:
        with fits.open(str(p)) as hdul:
            hdul.verify("fix")
            tem = np.array(hdul[0].data[edge_pos[0]:edge_pos[1], edge_pos[2]:edge_pos[3]])
            tem = cv2.flip(tem, 0)[1:, 1:]
            beam = read_beam_current(hdul[0].header, cfg.beamline)
        ob_stack.append(tem)
        ob_beams.append(beam)

    ob = median_filter(np.median(ob_stack, axis=0), size=5)

    logger.info("Reading DC references from %s", cfg.dir_dc)
    dc_files = sorted(cfg.dir_dc.glob("*.fits"))
    dc_stack: List[np.ndarray] = []
    for p in dc_files:
        tem = read_fits_image(p, edge_pos=edge_pos, flip_vertical=True)
        dc_stack.append(tem)
    dc = median_filter(np.median(dc_stack, axis=0), size=5)

    logger.info("Reading raw images from %s", cfg.input_dir)
    raw_files = sorted(cfg.input_dir.glob("*.fits"))
    if not raw_files:
        raise FileNotFoundError(f"No FITS files found in {cfg.input_dir}")

    # First pass: full scan reference positions (first grid)
    x_pos_full: List[int] = []
    y_pos_full: List[int] = []
    for i in range(min(number_positions, len(raw_files))):
        with fits.open(str(raw_files[i])) as hdul:
            hdul.verify("fix")
            x_position = float(hdul[0].header["sample_position_2/sp2_tx"][2:-3])
            y_position = float(hdul[0].header["sample_position_2/sp2_ty"][2:-3])
        x_pos_full.append(round(x_position))
        y_pos_full.append(round(y_position))

    # Second pass: load & normalize all frames
    imgs: List[np.ndarray] = []
    x_pos: List[int] = []
    y_pos: List[int] = []
    t_obs: List[float] = []

    for idx, p in enumerate(raw_files):
        logger.info("Opening image %d / %d: %s", idx + 1, len(raw_files), p.name)
        with fits.open(str(p)) as hdul:
            hdul.verify("fix")
            x_position = float(hdul[0].header["sample_position_2/sp2_tx"][2:-3])
            y_position = float(hdul[0].header["sample_position_2/sp2_ty"][2:-3])

            tem = np.array(hdul[0].data[edge_pos[0]:edge_pos[1], edge_pos[2]:edge_pos[3]])
            tem = cv2.flip(tem, 0)[1:, 1:]

            tem = -np.log((tem - dc) / (ob - dc))

            tem[np.isinf(tem)] = np.nan
            tem = median_filter(tem, size=5)

            date_taken = hdul[0].header["DATE"]
            epoch_s = parse_date_taken_to_epoch_seconds(date_taken)

        x_pos.append(round(x_position))
        y_pos.append(round(y_position))
        t_obs.append(epoch_s)
        imgs.append(tem)

    t_obs_arr = np.array(t_obs)
    t_obs_min = np.round((t_obs_arr - t_obs_arr[0]) / 60.0, 1)

    # Find overlaps
    if cfg.nr_grid_x > 1:
        _, idx_sel_x = np.unique(x_pos[:number_positions], return_index=True)
        idx_sel_x = np.sort(idx_sel_x[:2])
        selected = [imgs[i] for i in idx_sel_x]
        x_overlap = overlap_finder_horizontal(selected, cfg.scan_direction_x, cfg.overlapped_fraction_pct)
    else:
        x_overlap = None

    if cfg.nr_grid_y > 1:
        _, idx_sel_y = np.unique(y_pos[:number_positions], return_index=True)
        idx_sel_y = np.sort(idx_sel_y[:2])
        selected = [imgs[i] for i in idx_sel_y]
        y_overlap = overlap_finder_vertical(selected, cfg.scan_direction_y, cfg.overlapped_fraction_pct)
    else:
        y_overlap = None

    # Fix y-shift in compartment scans 
    y_shift = 0
    for n in range(len(imgs)):
        if y_pos[n] not in y_pos_full:
            if y_shift == 0:
                y_shift = find_y_shift(imgs[0], imgs[n])
            imgs[n] = fix_y_shift(imgs[0], imgs[n], y_shift)
            # align reported position to the corresponding fullscan
            y_pos[n] = y_pos_full[min(n % len(y_pos_full), len(y_pos_full) - 1)]

    # Build full scans by stitching tiles
    full_scans: List[np.ndarray] = []
    t_obs_new: List[float] = []

    scan_tiles: List[np.ndarray] = [np.empty_like(imgs[0]) * np.nan for _ in range(number_positions)]
    grid_counter = 0

    for n in range(len(imgs)):
        idx_next = min(n + 1, len(imgs) - 1)

        if grid_counter == 0:
            scan_tiles = [np.empty_like(imgs[0]) * np.nan for _ in range(number_positions)]

        if x_pos[n] == x_pos_full[grid_counter] and y_pos[n] == y_pos_full[grid_counter]:
            scan_tiles[grid_counter] = imgs[n]

        looped_back = (
            (x_pos[idx_next] == x_pos_full[0] and y_pos[idx_next] == y_pos_full[0])
            or (
                (np.array(x_pos_full[:grid_counter]) == x_pos[idx_next])
                * (np.array(y_pos_full[:grid_counter]) == y_pos[idx_next])
            ).any()
        )

        if looped_back:
            if cfg.nr_grid_x == 1 and cfg.nr_grid_y == 1:
                stitched = imgs[n]
            else:
                stitched = montage(
                    scan_tiles,
                    cfg.nr_grid_x,
                    cfg.nr_grid_y,
                    cfg.scan_direction_x,
                    cfg.scan_direction_y,
                    x_overlap,
                    y_overlap,
                )
            full_scans.append(stitched)
            t_obs_new.append(t_obs_min[n])
            grid_counter = 0
        else:
            grid_counter += 1

        if grid_counter >= number_positions:
            grid_counter = 0

    # Free memory
    del imgs, scan_tiles

    t_obs_min = np.array(t_obs_new)

    # Segmented image montage
    seg_prefix = "seg_"
    seg_files = sorted(cfg.input_dir.glob(f"*{seg_prefix}*.tif"))
    if not seg_files:
        raise FileNotFoundError(f"No segmented tif files matching '*{seg_prefix}*.tif' in {cfg.input_dir}")

    seg_tiles: List[np.ndarray] = []
    for p in seg_files:
        seg_tem = tif_read(p)[edge_pos[0]:edge_pos[1], edge_pos[2]:edge_pos[3]]
        seg_tem = seg_tem[1:, 1:]
        seg_tem = cv2.cvtColor(seg_tem, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        seg_tiles.append(seg_tem)

    seg = montage(
        seg_tiles,
        cfg.nr_grid_x,
        cfg.nr_grid_y,
        cfg.scan_direction_x,
        cfg.scan_direction_y,
        x_overlap,
        y_overlap,
    )

    # Color template
    color_key_path = cfg.dir_color_code / "ColorTemplate.tif"
    color_key = tif_read(color_key_path)
    color_key = cv2.cvtColor(color_key, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    color_key[color_key == color_key[0, 0]] = 0

    # --- Root color extraction ---
    index_noise = int(np.delete(np.unique(color_key[110:165, 450:]), 0)[0])

    idx_root1 = int(np.delete(np.unique(color_key[:85, 0:145]), 0)[0]) 
    idx_root2 = int(np.delete(np.unique(color_key[85:130, 0:145]), 0)[0])
    idx_root3 = int(np.delete(np.unique(color_key[130:190, 0:145]), 0)[0])
    idx_root4 = int(np.delete(np.unique(color_key[190:, 0:145]), 0)[0])
    idx_root5 = int(np.delete(np.unique(color_key[:85, 145:230]), 0)[0])
    idx_root6 = int(np.delete(np.unique(color_key[85:130, 145:230]), 0)[0])
    idx_root7 = int(np.delete(np.unique(color_key[130:190, 145:230]), 0)[0])
    idx_root8 = int(np.delete(np.unique(color_key[190:, 145:230]), 0)[0])
    idx_root9 = int(np.delete(np.unique(color_key[:90, 230:365]), 0)[0]) 
    idx_root10 = int(np.delete(np.unique(color_key[90:150, 230:365]), 0)[0])
    idx_root11 = int(np.delete(np.unique(color_key[150:190, 230:365]), 0)[0])
    idx_root12 = int(np.delete(np.unique(color_key[190:, 230:365]), 0)[0])

    idx_root13 = int(np.delete(np.unique(color_key[:90, 365:440]), 0)[0])
    idx_root14 = int(np.delete(np.unique(color_key[90:150, 365:440]), 0)[0])
    idx_root15 = int(np.delete(np.unique(color_key[150:190, 365:440]), 0)[0])
    idx_root16 = int(np.delete(np.unique(color_key[190:, 365:440]), 0)[0])
    idx_root17 = int(np.delete(np.unique(color_key[:85, 450:]), 0)[0])
    idx_root18 = int(np.delete(np.unique(color_key[85:110, 450:]), 0)[0])

    root_types = {
        idx_root1: "CR1", idx_root2: "CR2", idx_root3: "CR3", idx_root4: "CR4",
        idx_root5: "CR5", idx_root6: "CR6", idx_root7: "CR7", idx_root8: "CR8",
        idx_root9: "CR9", idx_root10: "CR10", idx_root11: "CR11", idx_root12: "CR12",
        idx_root13: "SR1", idx_root14: "SR2", idx_root15: "SR3", idx_root16: "SR4",
        idx_root17: "SR5", idx_root18: "SR6",
    }

    root = (
        idx_root1 * (seg == idx_root1)
        + idx_root2 * (seg == idx_root2)
        + idx_root3 * (seg == idx_root3)
        + idx_root4 * (seg == idx_root4)
        + idx_root5 * (seg == idx_root5)
        + idx_root6 * (seg == idx_root6)
        + idx_root7 * (seg == idx_root7)
        + idx_root8 * (seg == idx_root8)
        + idx_root9 * (seg == idx_root9)
        + idx_root10 * (seg == idx_root10)
        + idx_root11 * (seg == idx_root11)
        + idx_root12 * (seg == idx_root12)
        + idx_root13 * (seg == idx_root13)
        + idx_root14 * (seg == idx_root14)
        + idx_root15 * (seg == idx_root15)
        + idx_root16 * (seg == idx_root16)
        + idx_root17 * (seg == idx_root17)
        + idx_root18 * (seg == idx_root18)
    )
    noise = np.where(seg == index_noise, 1, 0).astype(np.uint8)

    # Root labeling
    kernel = np.ones((1, 1), np.uint8)
    root_thin = cv2.erode(np.array(root != 0, dtype=np.uint8), kernel, iterations=1)

    nr_root, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(root_thin != 0), connectivity=4
    )

    # remove small components
    for i in range(1, nr_root):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.min_root_component_size_px:
            labels[labels == i] = 0

    nr_root, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(labels != 0), connectivity=4
    )

    dilation_factor = 10
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    dilated_image = cv2.dilate(np.array(labels, dtype=np.uint8), kernel, iterations=1)
    key = (dilated_image - labels).copy()
    del dilated_image

    output: Dict[str, Dict] = {}
    count_root = 0

    for i in range(1, nr_root):
        label_size = stats[i, cv2.CC_STAT_AREA]
        if label_size < cfg.min_root_component_size_px:
            root[labels == i] = 0
            labels[labels == i] = 0
            labels[key == i] = 0
            continue

        count_root += 1
        rid = f"R{count_root}"
        output[rid] = {}

        values, counts = np.unique(root[labels == i], return_counts=True)
        root_color = int(values[np.argmax(counts)])

        output[rid]["RootCount"] = root_types.get(root_color, "UNKNOWN")
        output[rid]["RootLabel"] = count_root
        output[rid]["tObs [min]"] = np.array(t_obs_min)

        root[labels == i] = count_root
        labels[labels == i] = count_root

        cv2.putText(
            key,
            text=rid,
            org=(int(centroids[i][0]), int(centroids[i][1])),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=2,
            color=(15, 15, 15),
            thickness=3,
        )

    key[key > 0] = 1
    nr_root_final = int(np.max(labels))
    logger.info("Number of Roots: %d", nr_root_final)

    # ROI for contrast adjustment / display
    roi1 = np.where(np.sum(labels, axis=0) > 10)[0]
    roi2 = np.where(np.sum(labels, axis=1) > 10)[0]
    roi = [int(roi2[0]), int(roi2[-1]), int(roi1[0]), int(roi1[-1])]

    # --- Visualization ---
    show_image((full_scans[0] - cfg.mu_soil_al) / cfg.mu_h2o, "Overview of samples", None, "Water content [vol.]")
    median_filtered = median_filter(full_scans[-1] / full_scans[0], size=3)
    show_image(median_filtered, "Ratio of Img(tn)/Img(t0)", roi, "D2O Signal")
    show_image(median_filter(key + (full_scans[-1] / full_scans[0]), size=4), "Root Labeling Key", roi, "D2O Signal")

    # Skeleton and diameter
    skeleton = skeletonize(root != 0).astype(np.uint8) * 1
    for i in range(1, nr_root_final + 1):
        distance_map = np.max(distance_transform_edt(labels == i) * (skeleton > 0), axis=1)
        distance_map[distance_map == 0] = np.nan
        output[f"R{i}"]["Droot"] = 2 * distance_map * cfg.pixel_size_cm

    # Compute delta attenuation
    ac_t0 = full_scans[0].copy()
    ac = list(full_scans)
    del_ac = list(full_scans)
    del full_scans

    for t in range(len(ac)):
        logger.info("Computing change in Neutron Attenuation %d/%d", t + 1, len(ac))
        tem = ac[t] - ac[0]
        del_ac[t] = median_filter(tem, size=3)

    # CD2O per root
    for count, rid in enumerate(list(output.keys())):
        logger.info("Computing CD2O %d/%d (%s)", count + 1, len(output), rid)

        # Soil vicinity mask: dilate in/out
        dilation_in = 5
        kernel_in = np.ones((1, dilation_in), np.uint8)
        mask_root = np.where(labels == output[rid]["RootLabel"], 1, 0).astype(np.uint8)
        dilated_in = cv2.dilate(mask_root, kernel_in, iterations=1)

        dilation_out = 60
        kernel_out = np.ones((1, dilation_out), np.uint8)
        dilated_out = cv2.dilate(mask_root, kernel_out, iterations=1)

        dilated_ring = dilated_out - dilated_in
        dilated_ring = dilated_ring * (noise == 0) * (labels == 0)
        dilated_ring[np.sum(dilated_ring, axis=1) < 10, :] = 0

        # AC soil at t=0 (row-wise median)
        tem = ac_t0 * dilated_ring
        masked_tem = np.ma.masked_where((dilated_ring == 0) | np.isnan(tem), tem)
        ac_soil_t0 = np.ma.median(masked_tem, axis=1).filled(np.nan)

        # Delta AC in soil
        del_ac_soil = np.zeros((ac[0].shape[0], len(ac)))
        for t in range(len(ac)):
            tem = del_ac[t] * dilated_ring
            masked_tem = np.ma.masked_where((dilated_ring == 0) | np.isnan(tem), tem)
            del_ac_soil[:, t] = np.ma.median(masked_tem, axis=1).filled(np.nan)

        # Root pixels AC (erode)
        erosion_factor = 3
        kernel_er = np.ones((1, erosion_factor), np.uint8)
        eroded = cv2.erode(mask_root, kernel_er, iterations=1)

        tem = ac_t0 * eroded
        masked_tem = np.ma.masked_where((eroded == 0) | np.isnan(tem), tem)
        ac_rootsoil_t0 = np.ma.median(masked_tem, axis=1).filled(np.nan)

        d_soil = cfg.soil_thickness_cm - float(np.nanmedian(output[rid]["Droot"]))
        ac_root = (ac_rootsoil_t0 - d_soil * ac_soil_t0)
        ac_root[np.isnan(ac_rootsoil_t0)] = np.nan
        l_liq_t0 = float(np.nanmedian(ac_root / cfg.mu_h2o))

        cd2o_root = np.zeros((ac[0].shape[0], len(ac)))
        for t in range(1, len(del_ac)):
            tem = del_ac[t] * eroded
            masked_tem = np.ma.masked_where((eroded == 0) | np.isnan(tem), tem)
            tem_row = np.ma.median(masked_tem, axis=1).filled(np.nan)
            cd2o_root[:, t] = (tem_row - d_soil * del_ac_soil[:, t]) / (cfg.mu_d2o - cfg.mu_h2o) / l_liq_t0

        cd2o_root[:, 0] = 0

        # Select only root segment
        idx_root = ~np.isnan(output[rid]["Droot"])
        start = int(np.min(np.where(idx_root)))
        end = int(np.max(np.where(idx_root)))
        cd2o_root = cd2o_root[start:end, :]
        cd2o_root = cd2o_root[10:-10, :]  # keep your trimming

        output[rid]["Droot"] = float(np.nanmedian(output[rid]["Droot"]))
        output[rid]["DistanceFromDistalPart"] = np.linspace(cd2o_root.shape[0] - 1, 0, cd2o_root.shape[0]) * cfg.pixel_size_cm
        output[rid]["CD2O_Root"] = cd2o_root

    # Plot results
    for rid in output.keys():
        plt.figure(num=rid, figsize=(10, 8))
        for t in range(output[rid]["CD2O_Root"].shape[1]):
            plt.plot(output[rid]["DistanceFromDistalPart"], output[rid]["CD2O_Root"][:, t], ".")
        plt.xlabel("Distance From Distal Part [cm]")
        plt.ylabel("CD2O [-]")

    plt.figure(num="Overview", figsize=(10, 8))
    for rid in output.keys():
        plt.plot(output[rid]["tObs [min]"], np.nanmean(output[rid]["CD2O_Root"], axis=0), ".", label=rid)
    plt.xlabel("Time [min]")
    plt.ylabel("CD2O [-]")
    plt.legend()
    plt.savefig(str(output_folder / "overview.png"), dpi=300, bbox_inches="tight")

    # Export Excel
    xlsx_path = output_folder / "OutPuts.xlsx"
    if xlsx_path.exists():
        xlsx_path.unlink()

    with pd.ExcelWriter(str(xlsx_path), engine="xlsxwriter") as writer:
        for rid in output.keys():
            df = pd.DataFrame({"Root Diameter [cm]": [output[rid]["Droot"]], "Root Type": [output[rid]["RootCount"]]})

            for item, val in output[rid].items():
                if item in {"Droot", "RootLabel", "RootCount"}:
                    continue
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    df = pd.concat([df, pd.DataFrame({item: val})], axis=1)
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    for t in range(val.shape[1]):
                        df = pd.concat([df, pd.DataFrame({f"CD2O_Root_t{t}": val[:, t]})], axis=1)

                length = output[rid]["DistanceFromDistalPart"].shape[0]
                df = df.iloc[:length]

            df = df.replace([np.nan, "nan"], "")
            df.to_excel(writer, sheet_name=rid, header=True, startrow=0, index=False)

    # Export PDF of figures
    pdf_path = output_folder / "All_figures.pdf"
    save_figures_to_pdf(pdf_path)
    logger.info("Saved PDF: %s", pdf_path)
    logger.info("Saved Excel: %s", xlsx_path)
    logger.info("Output folder: %s", output_folder)


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process scan and compute CD2O along roots.")
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing FITS and seg_*.tif files.")
    p.add_argument("--dir-ob", type=Path, required=True, help="Folder containing OB *.fits files.")
    p.add_argument("--dir-dc", type=Path, required=True, help="Folder containing DC *.fits files.")
    p.add_argument("--dir-color-code", type=Path, required=True, help="Folder containing ColorTemplate.tif.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: input-dir")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    setup_matplotlib()

    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir

    cfg = Config(
        input_dir=args.input_dir,
        dir_ob=args.dir_ob,
        dir_dc=args.dir_dc,
        dir_color_code=args.dir_color_code,
        output_dir=output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
