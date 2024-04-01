"""Given the filepath to suite2/.../stat.npy, timestamps, and stimuli, load in the registered
movie, and save the relevant stimulus-aligned movies.
"""
from pathlib import Path
import re
import numpy as np
import tifffile
from typing import Union, List, Any, Dict
import xarray as xr
from scipy.ndimage import percentile_filter


def get_suite2p_folder(file):
    """ Returns base suite2p directory in file path

    Args:
        file (Union[str, Path]): filepath to anything in suite2p directory

    Returns:
        (Path): Path to top 'suite2p' folder.
    """
    file = Path(file)

    for folder in file.parents:
        if folder.name == 'suite2p':
            return folder
    return None


def path_to_plane(plane_folder):
    """ Extracts plane number from file path containing **/suite2p/plane{int}/...

    Args:
        plane_folder (Union[str, Path]): Path like
        "/local/storage/Remy/natural_mixtures/processed_data/
                                    2022-02-11/3/kiwi_ea_eb_only/downsampled_3/suite2p/plane6"

    Returns:
        (int): plane # in file path
    """
    pattern = '.+/suite2p/(plane(\d+))$'
    x = re.search(pattern, str(plane_folder))

    folder_name, plane_num = x.groups()
    plane_num = int(plane_num)

    return plane_num


def get_suite2p_plane_folders(suite2p_folder):
    """Returns all plane subdirectories in a suite2p folder (.../suite2p)

    Args:
        suite2p_folder (Path): Suite2p parent folder
    Returns:
        (List[Path]): suite2p/plane{:d} folders, sorted by plane number
    """
    plane_folders = sorted(list(suite2p_folder.glob("plane*")),
                           key=lambda x: path_to_plane(x))

    return plane_folders


def get_reg_tiff_index(tiff_file):
    x = re.search("file(\d+)_chan\d", tiff_file.name)

    tiff_idx = int(x.groups()[0])
    return tiff_idx


def load_single_plane_reg_tiffs_as_array(reg_dir, channel=0):
    """Load registered tiff stacks from suite2p/plane** folder.

    Args:
        reg_dir (Path): Path to `reg_tif`, contains tiffs named file{:03d}_chan0.tif
        channel (int): channel index (default 0)

    Returns:
         (np.ndarray): registered movie (TYX axis order)
    """
    tiff_files = sorted(list(reg_dir.joinpath('reg_tif').glob(f"file*_chan{channel}.tif")),
                        key=lambda x: get_reg_tiff_index(x))

    stacks = []
    for file in tiff_files:
        with tifffile.TiffFile(file) as tif:
            img = np.stack([page.asarray() for page in tif.pages], axis=0)
            stacks.append(img)

    reg_plane = np.concatenate(stacks, axis=0)
    return reg_plane


def load_single_plane_reg_tiffs(stat_file, channel=0, expand_z_dim=True):
    """
    Load 2d registered movie as xr.DataArray.

    Args:
        stat_file:
        channel:
        expand_z_dim:

    Returns:

    """
    reg_plane = load_single_plane_reg_tiffs_as_array(stat_file.with_name('reg_tif'),
                                                     channel=channel)
    T, Lz, Ly, Lx = get_dims(stat_file)

    if expand_z_dim:   # add Z-dimension
        reg_plane = np.expand_dims(reg_plane, axis=1)
        reg_plane = xr.DataArray(data=reg_plane,
                                 dims=['T', 'Z', 'Y', 'X'],
                                 coords=dict(
                                         X=range(Lx),
                                         Y=range(Ly),
                                         ),
                                 attrs=dict(stat_file=str(stat_file))
                                 )
    else:
        reg_plane = xr.DataArray(data=reg_plane,
                                 dims=['T', 'Y', 'X'],
                                 coords=dict(
                                         X=range(Lx),
                                         Y=range(Ly),
                                         ),
                                 attrs=dict(stat_file=str(stat_file))
                                 )
    return reg_plane


def load_combined_reg_tiffs(stat_file, channel=0):
    """Load 3d registered movie as xarray, with only good planes (not in 'ignore_flyback')
    included."""
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    nplanes = ops['nplanes']
    ignore_flyback = ops['ignore_flyback']
    good_planes = [plane for plane in range(nplanes) if plane not in ignore_flyback]

    time, Lz, Ly, Lx = get_dims(stat_file, without_flyback_planes=True)

    # load registered tiffs into (frames, z, y, x) np.array
    plane_folders = [stat_file.parent.with_name(f"plane{plane}") for plane in good_planes]

    reg_stack = \
        np.stack([load_single_plane_reg_tiffs_as_array(folder, channel=channel)
                  for folder in plane_folders], axis=1)

    reg_stack = xr.DataArray(data=reg_stack,
                             dims=['time', 'Z', 'Y', 'X'],
                             coords=dict(
                                     Z=good_planes,
                                     Y=range(Ly),
                                     X=range(Lx)
                                     ),
                             name='reg_stack',
                             attrs=dict(stat_file=str(stat_file))
                             )

    return reg_stack


def is_3d(stat_file, method='filepath'):
    """Check if stat.npy file is from a 3D recording.

    Args:
        stat_file (Path): path to stat.npy file
        method (str): 'filepath' or 'iplane'.
                       If 'filepath', then check if 'parent folder name' is in the filepath.
                       If 'iplane', then load `stat_file` and check if 'iplane' is a key.
   Returns:
       (bool): whether or not stat_file belongs to a 3D (multiplane) movie
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    if method == 'filepath':
        is_multiplane = 'combined' in stat_file.parent.name
    elif method == 'iplane':
        # load stat.npy file into a n_cells x 1 array of dicts
        stat = np.load(stat_file, allow_pickle=True)
        is_multiplane = 'iplane' in stat[0].keys()
    return is_multiplane


def load_suite2p_reg_tiffs(stat_file, channel=0):
    if is_3d(stat_file):
        reg_stack = load_combined_reg_tiffs(stat_file, channel=channel)
    else:
        reg_stack = load_single_plane_reg_tiffs_as_array(stat_file.with_name('reg_tif'),
                                                         channel=channel)
        reg_stack = np.expand_dims(reg_stack, axis=1)
    return reg_stack


# functions for extracting roi masks, taken from suite2p source code

def create_masks(stats: List[Dict[str, Any]], Ly, Lx, ops):
    """ create cell and neuropil masks """

    cell_masks = [create_cell_mask(stat, Ly=Ly, Lx=Lx, allow_overlap=ops['allow_overlap']) for stat
                  in stats]

    return cell_masks


def create_cell_pix(stats: List[Dict[str, Any]], Ly: int, Lx: int,
                    lam_percentile: float = 50.0) -> np.ndarray:
    """Returns Ly x Lx array of whether pixel contains a cell (1) or not (0).

    Taken from the `suite2p` package (see github)

    lam_percentile allows some pixels with low cell weights to be used,
    disable with lam_percentile=0.0

    """
    cell_pix = np.zeros((Ly, Lx))
    lammap = np.zeros((Ly, Lx))
    radii = np.zeros(len(stats))
    for ni, stat in enumerate(stats):
        radii[ni] = stat['radius']
        ypix = stat['ypix']
        xpix = stat['xpix']
        lam = stat['lam']
        lammap[ypix, xpix] = np.maximum(lammap[ypix, xpix], lam)
    radius = np.median(radii)
    if lam_percentile > 0.0:
        filt = percentile_filter(lammap, percentile=lam_percentile, size=int(radius * 5))
        cell_pix = ~np.logical_or(lammap < filt, lammap == 0)
    else:
        cell_pix = lammap > 0.0

    return cell_pix


def create_cell_mask(stat, Ly, Lx, allow_overlap):
    """
    Creates cell masks for ROIs in stat and computes radii (taken from suite2p source code.

    Taken from the `suite2p` package (see github)

    Args:
        stat : dictionary 'ypix', 'xpix', 'lam'
        Ly : y size of frame
        Lx : x size of frame
        allow_overlap : whether or not to include overlapping pixels in cell masks

    Returns
        cell_masks : pixels belonging to each cell and weights
        lam_normed
    """
    mask = ... if allow_overlap else ~stat['overlap']
    cell_mask = np.ravel_multi_index((stat['ypix'], stat['xpix']), (Ly, Lx))
    cell_mask = cell_mask[mask]
    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    return cell_mask, lam_normed


def get_dims(stat_file, without_flyback_planes=True):
    """Get dimensions of recording from stat.npy file.

    Note: if stat.npy file is in a `plane**` folder from a 3D recording, dimensions returned are
    for the single plane (i.e. Lz = 1).
    """

    if is_3d(stat_file):
        ops_3d = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
        ops_2d = np.load(stat_file.parent.with_name('plane0').joinpath('ops.npy'),
                         allow_pickle=True).item()
        Ly = ops_2d['Ly']
        Lx = ops_2d['Lx']

        nframes = ops_3d['nframes']

        # Z dimension
        nplanes = ops_3d['nplanes']
        ignore_flyback = ops_3d['ignore_flyback']
        good_planes = [plane for plane in range(nplanes) if plane not in ignore_flyback]

        if without_flyback_planes:
            Lz = len(good_planes)
        else:
            Lz = nplanes

        dims = (nframes, Lz, Ly, Lx)

    else:
        ops_2d = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
        Ly = ops_2d['Ly']
        Lx = ops_2d['Lx']
        nframes = ops_2d['nframes']

        dims = (nframes, 1, Ly, Lx)

    return dims
