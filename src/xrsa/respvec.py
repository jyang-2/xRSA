import xarray as xr
import numpy as np
from typing import Union, List


def filter_cells_by_coord(ds_respvec, good_xid, xid_coord='xid0'):
    """Filter cells by xid_coord values.

    Args:
        ds_respvec (Union[xr.Dataset, xr.DataArray]): must contain `xid_coord` as a coordinate
        good_xid (Union[np.array, List]): cluster IDs to include
        xid_coord (str): name of cell coordinate to filter

    Returns:
        ds_respvec_filt (Union[xr.Dataset, xr.DataArray]): ds_respvec w/ filtered cells
    """
    ds_respvec_filt = ds_respvec.where(ds_respvec[xid_coord].isin(good_xid), drop=True)
    return ds_respvec_filt


def filter_cells_by_attr(ds_respvec, attr_name='good_xid', xid_coord='xid0'):
    """Filter cells by `xid_coord`, based on an item in ds_respvec.attrs

    Args:
        ds_respvec (Union[xr.Dataset, xr.DataArray]): must contain `xid_coord` as a coordinate
        attr_name (str): ds_respvec.attrs key, for allowed values
        xid_coord (str): name of cell coordinate to filter

    Returns:
        ds_respvec_filt (Union[xr.Dataset, xr.DataArray]): ds_respvec w/ filtered cells
    """
    if attr_name not in ds_respvec.attrs.keys():
        raise AttributeError(f"Attribute `{attr_name}` does not exist.")

    good_xid = ds_respvec.attrs[attr_name]

    return filter_cells_by_coord(ds_respvec, good_xid, xid_coord=xid_coord)


def peak_amp(ds_trials, peak_win, peak_method='mean', peak_quantile=None,
             subtract_baseline=False,
             baseline_win=None, baseline_method='mean', baseline_quantile=None):
    """To perform baseline subtraction, baseline_win must be set."""


    ds_trials_peak = ds_trials.sel(time=slice(*peak_win)).copy(deep=True)

    # compute peak
    if peak_method == 'mean':
        ds_peak_amp = ds_trials_peak.mean(dim='time')
    elif peak_method == 'min':
        ds_peak_amp = ds_trials_peak.min(dim='time')
    elif peak_method == 'max':
        ds_peak_amp = ds_trials_peak.max(dim='time')
    elif peak_method == 'quantile':
        if peak_quantile is None:
            raise ValueError("`peak_quantile` must be provided.")
        ds_peak_amp = (ds_trials_peak
                       .quantile(peak_quantile, dim='time')
                       .drop('quantile')
                       )

    # add attributes of peak amplitude
    ds_peak_amp.attrs['respvec.peak_win'] = peak_win
    ds_peak_amp.attrs['respvec.peak_method'] = peak_method

    if peak_method == 'quantile':
        ds_peak_amp.attrs['respvec.peak_quantile'] = peak_quantile

    ###############################################
    # only if you want to run baseline subtraction
    ###############################################
    if subtract_baseline:
        # get baseline time chunk
        if baseline_win is None:
            raise ValueError("Cannot run baseline subtraction, `baseline_win` must be provided.")
        else:
            ds_trials_baseline = ds_trials.sel(time=slice(*baseline_win)).copy(deep=True)

        # compute baseline value
        if baseline_method == 'mean':
            ds_baseline_amp = ds_trials_baseline.mean(dim='time')
        elif baseline_method == 'min':
            ds_baseline_amp = ds_trials_baseline.min(dim='time')
        elif baseline_method == 'max':
            ds_baseline_amp = ds_trials_baseline.max(dim='time')
        elif baseline_method == 'quantile':
            if baseline_quantile is None:
                raise ValueError("`baseline_quantile` must be provided.")
            ds_baseline_amp = (ds_trials_baseline
                               .quantile(baseline_quantile, dim='time')
                               .drop('quantile')
                               )
        ds_peak_amp = ds_peak_amp - ds_baseline_amp

        # add baselining attributes
        ds_peak_amp.attrs['respvec.baseline_win'] = baseline_win
        ds_peak_amp.attrs['respvec.baseline_method'] = baseline_method

        if peak_method == 'quantile':
            ds_peak_amp.attrs['respvec.baseline_method'] = baseline_method

    return ds_peak_amp
