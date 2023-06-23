import xarray as xr
from sklearn import metrics
import pandas as pd


def check_input_to_rdm(ds_input):
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
            kwargs=dict(metric=metric),
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

    ds_rdm.attrs['distance_metric'] = metric

    return ds_rdm
