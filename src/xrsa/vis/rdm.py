import xarray as xr
import matplotlib as plt


def plot_rsm_heatmap(da_rsm):
    ds_rdm_sorted = (da_rsm
                     .assign_coords(row_occ=('trial_row', range(da_rsm.dims['trial_row'])),
                                    col_occ=('trial_col', range(da_rsm.dims['trial_col']))
                                    )
                     .set_index(trial_row=['row_stim', 'row_occ'],
                                trial_col=['col_stim', 'col_occ'])
                     .sortby(['col_stim', 'col_occ'])
                     .sortby(['row_stim', 'row_occ'])
                     )
    ds_rdm_mean = ds_rdm_sorted.drop('acq').mean(dim='acq')
    img_rdm = (1 - ds_rdm_sorted['Fc_zscore']).plot.imshow(
            x='col_stim',
            y='row_stim',
            cmap='RdBu_r',
            # vmin=-1, vmax=1,
            center=0,
            yincrease=False,
            xticks=list(range(1, 51, 3)),
            yticks=list(range(1, 51, 3)),
            )

    img_rdm.axes.set_xticklabels(img_rdm.axes.get_xticklabels(), rotation=90)
    img_rdm.axes.set_box_aspect(1)

    plt.tight_layout()
    plt.show()
