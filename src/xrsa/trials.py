"""Functions for working with trial-structured neural response timeseries."""

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import ryeutils

xr.set_options(keep_attrs=True)


def timeseries_2_trials(ds_timeseries, stim_ict, stim_list, trial_ts, index_stimuli=False,
                        stimulus_index_keys=None):
    """Converts timeseries dataset (cells x time) to a (trials, cells, time) tensor dataset.

    Args:

        ds_timeseries (xr.Dataset): suite2p outputs with timestamps
        stim_ict (Union[list, np.ndarray]): stimulus onset times, from ThorSync line 'olf_ict'
        stim_list (list): stimulus identifier strings, same size as `stim_ict`
        trial_ts (np.ndarray): timestamps relative to `stim_ict` times to use for each trial
        index_stimuli (bool): whether to include additional stimulus info (repeats, runs, etc.)
        stimulus_index_keys (list): which keys to keep from indexed stimuli returned by
            `ryeutils.index_stimuli`.  Default value is `['stim', 'stim_occ', 'run_idx',
            'idx_in_run', 'run_occ']`. Only used if `index_stimuli=True`.

    Returns:
        xr.Dataset: (trials x cells x time) with `stim_ict` and `stim_list` stored in `attrs`

    """

    trials = []

    # split cells x time xr.Dataset by trials, with stimulus onset at time=0
    for trial_idx, ict in enumerate(stim_ict):
        ds0 = ds_timeseries.interp(time=trial_ts + ict)
        ds0 = ds0.assign_coords(
                trials=trial_idx,
                time=trial_ts
                )
        trials.append(ds0)

    ds_trials0 = xr.concat(trials, 'trials')

    # index stimuli (get additional information about occurrence, consecutive runs, etc.
    if index_stimuli:
        stim_idx = ryeutils.index_stimuli(stim_list, include_trial_idx=True)

        # keep stimulus_index_keys in stim_idx, if stimulus_index_keys is specified
        if stimulus_index_keys is not None:
            stim_idx = {k: stim_idx[k] for k in stimulus_index_keys}

        ds_trials0 = ds_trials0.assign_coords(
                {k: ('trials', v) for k, v in stim_idx.items()}
                )
        ds_trials0 = ds_trials0.set_index(trials=list(stim_idx.keys()))
    else:
        ds_trials0 = ds_trials0.assign_coords(stim=('trials', stim_list))

    # add attributes to
    ds_trials0.attrs['trials.stim_ict'] = list(stim_ict)
    ds_trials0.attrs['trials.stim_list'] = list(stim_list)

    return ds_trials0


def baseline_correct_trials(ds_trials, baseline_win=(-5, 0), baseline_method='quantile',
                            baseline_quantile=0.5):
    """Baseline-corrects trials by subtracting the mean/baseline quantile of the baseline window.

    Args:
        ds_trials (xr.Dataset): trial dataset, with timestamps centered on 0 (stimulus onset time)
        baseline_win (tuple): time window of baseline
        baseline_method (str): 'mean' or 'quantile'
        baseline_quantile (float): used only if baseline_method='quantile'

    Returns:
        ds_bc_trials (xr.Dataset): baseline-corrected dataset, with parameters added to `attrs`
    """

    if baseline_method == 'quantile':
        ds_baseline = (ds_trials
                       .sel(time=slice(*baseline_win))
                       .quantile(0.5, dim='time')
                       .drop('quantile')
                       )
    elif baseline_method == 'mean':
        # da_baseline = da_trials.sel(time=slice(-5, 0)).mean(dim='time')
        ds_baseline = (ds_trials
                       .sel(time=slice(*baseline_win))
                       .mean(dim='time'))

    ds_bc_trials = ds_trials - ds_baseline

    # add information about baselining to attrs
    ds_bc_trials.attrs['baseline.baseline_win'] = baseline_win
    ds_bc_trials.attrs['baseline.baseline_method'] = baseline_method
    if baseline_method == 'quantile':
        ds_bc_trials.attrs['baseline.baseline_quantile'] = baseline_quantile

    return ds_bc_trials
