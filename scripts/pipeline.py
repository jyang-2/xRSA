"""Run all steps to generate RDMs from suite2p outputs."""
from pathlib import Path
import numpy as np
import xarray as xr
import attrs
from attrs import define, field
from typing import List, Union

import external
from expt.acquisition import Acquisition
import xrsa
import matplotlib.pyplot as plt

# %%

if __name__ == "__main__":
    # %%
    stat_file = Path(
            "/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/2022-10-26/2"
            "/megamat1_singleplane/suite2p/plane0/stat.npy")

    acq = Acquisition.from_stat_file(stat_file)
    acq.load_timestamps()
    acq.load_stim_list()

    # %% make cells x time dataset (timeseries)

    ds_suite2p_outputs = external.suite2p.convert.outputs_2_xarray_base(stat_file)

    # add timestamps, where `timestamps` is a 1-D numpy array
    ds_suite2p_outputs = xrsa.timeseries.add_timestamps_to_suite2p_outputs(
            ds_suite2p_outputs, timestamps=acq.timestamps['stack_times'])

    # drop cells where `iscell=0`
    ds_suite2p_outputs = ds_suite2p_outputs.where(ds_suite2p_outputs.iscell == 1, drop=True)

    # %% Convert timeseries (cells x time) to trials (trials x cells x time) and baseline correct

    # ds_suite2p_outputs = xr.load_dataset(
    #         "/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/2023-05-10/3"
    #         "/megamat1_calyx/source_extraction_s2p/suite2p/plane0/iscell_calyx"
    #         "/xrds_suite2p_outputs.nc")

    ds_trials = xrsa.trials.timeseries_2_trials(ds_suite2p_outputs,
                                                stim_ict=acq.timestamps['olf_ict'],
                                                stim_list=acq.stim_list,
                                                trial_ts=np.arange(-5, 20, 0.05).round(3),
                                                index_stimuli=True,
                                                stimulus_index_keys=['stim', 'stim_occ',
                                                                     'trial_idx'])

    # baseline-correct traces
    #   for PN boutons/KC claws, use baseline_method='quantile')
    #   for KC soma, use baseline_method = 'mean', baseline_quantile is ignored

    ds_bc_trials = xrsa.trials.baseline_correct_trials(ds_trials,
                                                       baseline_win=(-5, 0),
                                                       baseline_method='quantile',
                                                       baseline_quantile=0.5
                                                       )

    # %% compute RDM
    ds_rdm = xrsa.rdm.compute_trial_respvec_rdm(ds_bc_trials, metric='correlation')

    # %%
    ds_rdm_sorted = \
        (ds_rdm
         .sortby(['row_stim', 'row_stim_occ'])
         .sortby(['col_stim', 'col_stim_occ'])
         )

    # %% plot RSM
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1)
    da_rsm = 1 - ds_rdm_sorted['Fc_zscore']
    img_rdm = (da_rsm.sel(time=0.5)).plot.imshow(
            x='row_stim',
            y='col_stim',
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            center=0,
            yincrease=False,
            xticks=list(range(1, 51, 3)),
            yticks=list(range(1, 51, 3)),
            ax=ax
            )

    new_xticklabels = [label.get_text().split(' @ ')[0] for label in img_rdm.axes.get_xticklabels()]
    img_rdm.axes.set_xticklabels(new_xticklabels,
                                 rotation=90
                                 )
    img_rdm.axes.set_box_aspect(1)

    plt.tight_layout()
    plt.show()
