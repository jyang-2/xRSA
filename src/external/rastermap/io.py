"""Code for copying and converting Fc_zscore files with different iscell values to rmap folder"""
from pathlib import Path
import numpy as np
from ..suite2p.iscells import get_iscell_suffix


def move_fluorescence_with_iscell(stat_file, F_filename, iscell_filename):
    """Copy 'Fc_zscore.npy' and 'iscell_{{iscell_suffix}}.npy' files to /rmap/{{iscell_suffix}}
    directory.

    Only good cells in `iscell_{{iscell_suffix}}.npy` are included in the new
    'rmap/iscell_{{iscell_suffix}}/Fc_zscore.npy' file.

    Run rastermap on '/rmap/{{iscell_suffix}}/{{F_filename}}

    Args:
        stat_file (Path): Path to 'stat.npy' file (suite2p output file)
        F_filename (str): Name of 'Fc_zscore.npy' (or similar fluorescence) file,
          in same directory as `stat_file`.
        iscell_filename (str): Name of iscell_{{iscell_suffix}}.npy file, \
          in same directory as `stat_file`

    Returns:
        rmap_fluorescence_file (Path): Path to '/rmap/iscell_{{iscell_suffix}}/{{F_filename}}'

    """
    rmap_dir = stat_file.with_name('rmap')
    rmap_dir.mkdir(exist_ok=True)

    F_stem = stat_file.with_name(F_filename).stem

    iscell_filepath = stat_file.joinpath(iscell_filename)
    iscell_stem = iscell_filepath.stem

    rmap_iscell_dir = rmap_dir.joinpath(iscell_stem)
    rmap_iscell_dir.mkdir(exist_ok=True)

    iscell = np.load(stat_file.with_name(iscell_filename))
    F = np.load(stat_file.with_name(F_filename))

    F_cropped = F[iscell[:, 0] == 1, :]

    suffix = get_iscell_suffix(iscell_filename)
    F_new_filename = f"{F_stem}_{suffix}.npy"

    np.save(rmap_iscell_dir.joinpath(iscell_filename), iscell)
    np.save(rmap_iscell_dir.joinpath(F_new_filename), F_cropped)

    return rmap_iscell_dir.joinpath(F_new_filename)


def load_fluoresence_and_iscell(stat_file, fluorescence_filename, iscell_filename):
    """Load fluorescence and

    Args:
        stat_file:
        fluorescence_filename:
        iscell_filename:

    Returns:

    """
    F = np.load(stat_file.with_name(fluorescence_filename))
    iscell = np.load(stat_file.with_name(iscell_filename))
    return F, iscell


