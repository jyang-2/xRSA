import xarray as xr
from sklearn import metrics
import pandas as pd
import numpy as np


def compute_rdm(ds_respvec, metric='correlation', input_dim_ord=None,
                output_dim_names=None, output_suffixes=None):
    """Compute representation dissimilarity matrix w/ specified dimension order.

    Args:
        ds_respvec (Union[xr.Dataset, xr.DataArray]):
        metric (str): pairwise distance metric (see `sklearn.metrics.pairwise_distances`)
        input_dim_ord (List[str]): input dimension order
          - if input_dim_ord has dims (trials, cells), then output_dim_ord has dims (trials, trials)
        output_dim_names (List[str]): output dimension names for RDM with shape `(input_dim_ord[0],
            input_dim_ord[0])`
        output_suffixes (List[str]): suffixes used to generate output dim. names
            - default ['_row', '_col']
            - used only if output_dim_names = `None`
    Returns:
        ds_rdm (Union[xr.Dataset, xr.DataArray]):

    Notes:
        If `input_dim_ord = ['trials', 'cells']` and `output_suffixes = ['_row', '_col']`, then
        output_dim_ord = ['trials_row', 'trials_col']
    """
    if input_dim_ord is None:  # input dimensions default to the first 2
        dims = list(ds_respvec.dims.keys())
        input_dim_ord = dims[:2]
    if output_dim_names is None:  # construct output_dim_names
        if output_suffixes is None:
            output_suffixes = ['_row', '_col']
        output_dim_names = [f"{input_dim_ord[0]}{suffix}" for suffix in output_suffixes]
    # construct output dimension names

    # print(input_dim_ord)
    # print(output_dim_names)

    ds_rdm = xr.apply_ufunc(
            metrics.pairwise_distances,
            ds_respvec,
            input_core_dims=[input_dim_ord],
            output_core_dims=[output_dim_names],
            vectorize=True,
            kwargs=dict(metric=metric, force_all_finite=False),
            keep_attrs=True)

    # copy coordinates along the 1st intput dimension to the output dimensions
    coords = {output_dim: (output_dim, ds_respvec.indexes[input_dim_ord[0]].copy())
              for output_dim in output_dim_names}

    ds_rdm = ds_rdm.assign_coords(coords)

    ds_rdm.attrs['rdm.metric'] = metric
    return ds_rdm


def check_input_to_respvec_rdm(ds_input):
    """Check that the input to `compute_trial_respvec_rdm` has the required structure.

    If `trials` is single index, check that
    """
    # check if `trials is a MultiIndex
    # multiindex_trials = ds_input.indexes.is_multi('trials')

    # check that it has a `trials` dimension
    has_trials = 'trials' in ds_input.indexes.keys()

    if not has_trials:
        raise ValueError('Missing `trials` dimension coordinate')

    has_stim = 'stim' in ds_input['trials'].coords.keys()

    if not has_stim:
        raise ValueError('Missing `stim` coordinate along `trials` dimension')

    # check if `trials` is a MultiIndex
    # multiindex_trials = ds_input.indexes.is_multi('trials')
    #
    # if multiindex_trials:
    #     'stim' in ds_trials['trials'].coords.keys()


def compute_trial_respvec_rdm(ds_respvec, metric='correlation'):
    """Compute RDM w/ dims (..., trial_row, trial_col) from a (..., cells, time) dataset.

    `ds_respvec` must have dimension `trials`.

    If `trials` is a MultiIndex, copy all the MultiIndex columns to `trial_row` and `trial_col`
    with prefixes "row_" and "col_".

    Simple trials index:
    --------------------
    If `trials` is a simple index, it must also have coordinate `stim` along the `trials`
    dimension, and

        trials --> trial_row
        trials --> trial_col

        stim --> row_stim (along `trial_row` dimension)
        stim --> col_stim (along `trial_col` dimension)



        <xarray.Dataset>
        Dimensions:        (time: 500, trial_row: 51, trial_col: 51)
        Coordinates:
          * time           (time) float64 -5.0 -4.95 -4.9 -4.85 ... 19.85 19.9 19.95
            row_stim       (trial_row) <U12 'ep @ -3.0' 'ep @ -3.0' ... 'benz @ -3.0'
            row_trial_idx  (trial_row) int64 0 1 2 3 4 5 6 7 ... 43 44 45 46 47 48 49 50
            col_stim       (trial_col) <U12 'ep @ -3.0' 'ep @ -3.0' ... 'benz @ -3.0'
            col_trial_idx  (trial_col) int64 0 1 2 3 4 5 6 7 ... 43 44 45 46 47 48 49 50
        Dimensions without coordinates: trial_row, trial_col

    MultiIndex trials:
    ------------------
    If `trials` is a MultiIndex, `trial_row` and `trial_col` will also be MultiIndex.
    All coordinates along `trials` will be copied to dimensions `trial_row` and `trial_col`,


    Examples:
        >>>
        <xarray.Dataset>
        Dimensions:        (time: 500, trial_row: 51, trial_col: 51)
        Coordinates:
          * time           (time) float64 -5.0 -4.95 -4.9 -4.85 ... 19.85 19.9 19.95
            row_stim       (trial_row) <U12 'ep @ -3.0' 'ep @ -3.0' ... 'benz @ -3.0'
            row_trial_idx  (trial_row) int64 0 1 2 3 4 5 6 7 ... 43 44 45 46 47 48 49 50
            col_stim       (trial_col) <U12 'ep @ -3.0' 'ep @ -3.0' ... 'benz @ -3.0'
            col_trial_idx  (trial_col) int64 0 1 2 3 4 5 6 7 ... 43 44 45 46 47 48 49 50
        Dimensions without coordinates: trial_row, trial_col
        Data variables:
            Fc             (time, trial_row, trial_col) float64 0.0 0.5505 ... 1.472 0.0
            F              (time, trial_row, trial_col) float64 0.0 0.5505 ... 1.472 0.0
            Fneu           (time, trial_row, trial_col) float64 0.0 nan nan ... nan 0.0
            spks           (time, trial_row, trial_col) float64 0.0 0.5924 ... 0.0
            F_zscore       (time, trial_row, trial_col) float64 0.0 1.095 ... 1.247 0.0
            Fc_zscore      (time, trial_row, trial_col) float64 0.0 1.095 ... 1.247 0.0
        Attributes:
            baseline.baseline_win:       (-5, 0)
            baseline.baseline_method:    quantile
            baseline.baseline_quantile:  0.5
            distance_metric:             correlation

    """
    # compute RDM with dims (..., trial_row, trial_col)
    ds_rdm = xr.apply_ufunc(
            metrics.pairwise_distances,
            ds_respvec,
            input_core_dims=[['trials', 'cells']],
            output_core_dims=[['trial_row', 'trial_col']],
            vectorize=True,
            kwargs=dict(metric=metric, force_all_finite=False),
            keep_attrs=True
            )

    if ds_respvec.indexes.is_multi('trials'):
        # copy all multiindex columns to trial_row and trial_col
        mi = ds_respvec['trials'].to_index()
        mi = pd.MultiIndex.from_frame(mi.to_frame(index=False).convert_dtypes())

        mi_row = mi.set_names({k: f"row_{k}" for k in mi.names})
        mi_col = mi.set_names({k: f"col_{k}" for k in mi.names})

        ds_rdm = (ds_rdm
                  .assign_coords(trial_row=mi_row, trial_col=mi_col)
                  .reset_index('trial_row')
                  .reset_index('trial_col')
                  )

        row_dtypes = [type(item) for item in mi_row.to_flat_index()[0]]
        for n, d in zip(mi_row.names, row_dtypes):
            ds_rdm[n] = ds_rdm[n].astype(d)

        col_dtypes = [type(item) for item in mi_col.to_flat_index()[0]]
        for n, d in zip(mi_col.names, col_dtypes):
            ds_rdm[n] = ds_rdm[n].astype(d)

        ds_rdm = (ds_rdm
                  .set_xindex(mi_row.names)
                  .set_xindex(mi_col.names)
                  )

    else:
        trial_idx = ds_respvec.trials.to_numpy()
        ds_rdm = ds_rdm.assign_coords(
                row_stim=('trial_row', ds_respvec.stim.to_numpy().tolist()),
                row_trial_idx=('trial_row', trial_idx),
                col_stim=('trial_col', ds_respvec.stim.to_numpy().tolist()),
                col_trial_idx=('trial_col', trial_idx)
                )

    ds_rdm.attrs['rdm.metric'] = metric

    return ds_rdm


def sort_trial_rdm_by_stim_ord(ds_rdm, stim_ord, use_stim_occ=True):
    """Sort ds_rdm to match desired stimulus ordering, in coordinates row_stim and col_stim.

    ds_rdm (xr.Dataset): has dimensions ('trial_row', 'trial_col'), and coords ('row_stim',
      'col_stim')
    stim_ord (list): stimulus order for coords ('row_stim', 'col_stim')

    """
    if use_stim_occ:
        df_row = ds_rdm.trial_row.to_dataframe().reset_index(drop=True)
        df_row['row_stim'] = pd.Categorical(df_row['row_stim'], ordered=True,
                                            categories=stim_ord)
        row_idx = df_row.sort_values(['row_stim', 'row_stim_occ']).index.to_list()

        df_col = ds_rdm.trial_col.to_dataframe().reset_index(drop=True)
        df_col['col_stim'] = pd.Categorical(df_col['col_stim'], ordered=True,
                                            categories=stim_ord)
        col_idx = df_col.sort_values(['col_stim', 'col_stim_occ']).index.to_list()
    else:
        row_idx = np.argsort([stim_ord.index(item) for item in ds_rdm.row_stim.to_numpy()])
        col_idx = np.argsort([stim_ord.index(item) for item in ds_rdm.col_stim.to_numpy()])

    ds_rdm_sorted = ds_rdm.isel(trial_row=row_idx).isel(trial_col=col_idx)
    return ds_rdm_sorted


def sort_stim_rdm_by_stim_ord(ds_stim_rdm, stim_ord, row_coord='stim_row', col_coord='stim_col'):
    """Sort ds_stim_rdm to match desired stimulus ordering."""

    row_idx = np.argsort([stim_ord.index(item) for item in ds_stim_rdm[row_coord].to_numpy()])
    col_idx = np.argsort([stim_ord.index(item) for item in ds_stim_rdm[col_coord].to_numpy()])

    ds_stim_rdm_sorted = ds_stim_rdm[{row_coord: row_idx}][{col_coord: col_idx}]
    return ds_stim_rdm_sorted


def prepare_to_align(ds_rdm):
    """Set indexes and drop coords before using xr.align"""
    ds_rdm_prepared = (ds_rdm
                       .reset_index(['trial_row', 'trial_col'])
                       .reset_coords(['row_trial_idx', 'col_trial_idx'], drop=True)
                       .set_xindex(coord_names=['row_stim', 'row_stim_occ'])
                       .set_xindex(coord_names=['col_stim', 'col_stim_occ'])
                       .sortby('trial_row')
                       .sortby('trial_col')
                       )
    return ds_rdm_prepared


def acq_attrs_2_coords(ds_rdm):
    """Copy Acquisition fields from attrs (use before concatenating acquisitions)"""
    attrs = ds_rdm.attrs.copy()

    return ds_rdm.assign_coords(
            date_imaged=attrs['acq.date_imaged'],
            fly_num=attrs['acq.fly_num'],
            thorimage_name=attrs['acq.thorimage_name']
            )
