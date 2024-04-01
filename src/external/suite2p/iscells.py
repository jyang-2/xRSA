from pathlib import Path
import numpy as np
from datetime import datetime
import re
from attrs import define, field
from datetime import datetime


@define(kw_only=True)
class IsCell:
    iscell_file: Path = field(converter=Path)
    iscell: np.array = field(init=False)
    n_total_rois: int = field(init=False)
    n_iscell_rois: int = field(init=False)
    fraction_iscell: float = field(init=False)
    st_ctime: datetime = field(init=False)
    st_mtime: datetime = field(init=False)

    def __attrs_post_init__(self):
        roi_stats = get_iscell_roistats(self.iscell_file)
        self.iscell = np.load(self.iscell_file)
        self.n_total_rois = roi_stats['n_total_rois']
        self.n_iscell_rois = roi_stats['n_iscell_rois']
        self.fraction_iscell = self.n_iscell_rois / self.n_total_rois

        file_stats = get_iscell_filestats(self.iscell_file)
        self.st_ctime = file_stats['st_ctime']
        self.st_mtime = file_stats['st_mtime']


def get_iscell_filestats(iscell_file: Path) -> dict:
    """Given a file path to an iscell.npy file, get the file creation and modification times.

    """

    st = iscell_file.stat()
    file_stats = {'st_ctime': datetime.fromtimestamp(st.st_ctime),
                  'st_mtime': datetime.fromtimestamp(st.st_mtime),
                  }
    return file_stats


def get_iscell_roistats(iscell_file: Path) -> dict:
    """Get the # of selected rois and total rois from an iscell.npy file."""

    iscell = np.load(iscell_file)
    return {'n_total_rois': iscell.shape[0],
            'n_iscell_rois': int(iscell[:, 0].sum())}


def get_iscell_stats(iscell_file: Path) -> dict:
    return {**get_iscell_filestats(iscell_file), **get_iscell_roistats(iscell_file)}


# def make_iscell_fluorescence(iscell_file, parent_folder='rastermap'):
#     """Create a folder w/ the same name as iscell_file in rastermap folder. Copy over selected
#     Fc_zscore.npy where iscell == 1, and """
#     x = re.search("file(\d+)_chan\d", tiff_file.name)


def get_iscell_suffix(iscell_filename: str) -> str:
    """Given an iscell*.npy file, get the suffix of the filename."""

    pattern = 'iscell_(?P<suffix>.+).npy'
    if iscell_filename == 'iscell.npy' or iscell_filename == 'iscell':
        suffix = None
    else:
        if iscell_filename.endswith('.npy'):
            pattern = 'iscell_(?P<suffix>.+).npy'
        else:
            pattern = 'iscell_(?P<suffix>.+)'

        suffix = re.search(pattern, iscell_filename).groupdict()['suffix']

    return suffix


def all_iscell_stats_from_stat_file(stat_file: Path, astype='list'):
    iscell_files = sorted(list(stat_file.parent.glob("iscell*.npy")))

    iscell_stats = [get_iscell_stats(file) for file in iscell_files]

    if astype == 'list':
        # add iscell_filename to dict
        fstats = [
            {'iscell_filename': iscell_file.name, **stat}
            for iscell_file, stat in zip(iscell_files, iscell_stats)
            ]

    elif astype == 'dict':
        fstats = {iscell_file.name: stat for iscell_file, stat in zip(iscell_files, iscell_stats)}
    else:
        raise ValueError(astype)

    return fstats
