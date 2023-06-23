from pathlib import Path
import numpy as np
import numpy.typing as npt
import xarray as xr
from typing import List


def add_timestamps_to_suite2p_outputs(ds_suite2p_outputs, timestamps):
    """Add timestamps to the output of `external.suite2p.convert.outputs_2_xarray_base`

    Args:
        ds_suite2p_outputs (xr.Dataset): suite2p output data contents
        timestamps (np.array): 1d array of timestamps for each suite2p timepoint

    Returns:
        xr.Dataset: suite2p data with time dimension

    Examples:
        >>> stat_file = Path("./combined/stat.npy")
        >>> timestamps = np.load("timestamps.npy")
        >>> ds_suite2p_outputs = external.suite2p.convert.outputs_2_xarray_base(stat_file)
        >>> ds_suite2p_outputs = add_timestamps_to_suite2p_outputs(ds_suite2p_outputs, timestamps)
    """

    return ds_suite2p_outputs.assign_coords(time=timestamps)


def to_trial_tensor(ds_suite2p_outputs, stim_ict, stim_list, trial_ts=None):
    """Convert (cells, time) dataset  to (cells, time, trials), from stim. times and identifiers.

    Args:
        ds_suite2p_outputs (xr.Dataset): suite2p outputs with timestamps
        stim_ict (Union[list, np.ndarray]): stimulus onset times, from ThorSync line 'olf_ict'
        stim_list (list): stimulus identifier strings, same size as `stim_ict`
        trial_ts (np.ndarray): timestamps relative to `stim_ict` times to use for each trial

    Returns:
        xr.Dataset: (trials x cells x time) with `stim_ict` and `stim_list` stored in `attrs`
    """

    if trial_ts is None:
        trial_ts = np.arange(-5, 20, 0.05).round(3)

