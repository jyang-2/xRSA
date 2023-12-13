import xarray as xr
import numpy as np
import pandas as pd


def add_spectral_coclusters_to_stim_respvec(da_respvec, model,
                                            row_label_coord='stim',
                                            col_label_coord='cells',
                                            row_label_coordname='stim_labels',
                                            col_label_coordname='cell_labels'
                                            ):
    """
    Adds cell and stimulus clusters to response vector dataarray, as `stim_labels` and
    `cell_labels`.
    Model parameters added to attrs.

    Args:
        da_respvec (xr.DataArray): da_mean_peak_stim['Fc_zscore'], with dims ('cells', 'stim')
        model: fitted
            sklearn.clustering.SpectralBiclustering

    Returns:
        da_coclust (xr.DataArray), dims ('cells', 'stim'), has additional coords `stim_labels`
        and `cell_labels`

    Examples:
        To run spectral coclustering::

            n_biclusters = (16, 16)
            clustering = (SpectralBiclustering(n_clusters=n_biclusters,
                                       method='scale',
                                       n_init=5)
                  .fit(ds_mean_peak_stim['Fc_zscore']))

            da_coclust = xrsa.cluster.spectral.add_spectral_coclusters_to_stim_respvec(
                ds_mean_peak_stim['Fc_zscore'],
                clustering
                )
            df_cell_cluster_counts = xrsa.cluster.spectral.get_cell_cluster_counts(da_coclust)
    """

    da_coclust = da_respvec.assign_coords(
            {
                row_label_coordname: (row_label_coord, model.row_labels_),
                col_label_coordname: (col_label_coord, model.column_labels_)
                },
            )
    da_coclust.attrs = model.get_params()
    return da_coclust


def get_cell_cluster_counts(da_coclust):
    df_cell_cluster_counts = da_coclust['cell_labels'].to_pandas().value_counts().sort_index()
    df_cell_cluster_counts = (df_cell_cluster_counts.reset_index()
                              .rename(columns={'index': 'cell_cluster', 0: 'n_cells'})
                              )
    n_cells = df_cell_cluster_counts['n_cells']
    df_cell_cluster_counts['fraction'] = df_cell_cluster_counts['n_cells'] / n_cells
    return df_cell_cluster_counts
