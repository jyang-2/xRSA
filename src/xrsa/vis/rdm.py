import xarray as xr
import matplotlib.pyplot as plt
import ryeutils
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})


def plot_rsm_heatmap(da_rsm, row_coord='row_stim', col_coord='col_stim', ax=None, cbar_ax=None,
                     heatmap_kws=None):
    # ds_rdm_sorted = (da_rsm
    #                  .assign_coords(row_occ=('trial_row', range(da_rsm.dims['trial_row'])),
    #                                 col_occ=('trial_col', range(da_rsm.dims['trial_col']))
    #                                 )
    #                  .set_index(trial_row=['row_stim', 'row_occ'],
    #                             trial_col=['col_stim', 'col_occ'])
    #                  .sortby(['col_stim', 'col_occ'])
    #                  .sortby(['row_stim', 'row_occ'])
    #                  )
    # ds_rdm_mean = ds_rdm_sorted.drop('acq').mean(dim='acq')
    # img_rdm = (1 - ds_rdm_sorted['Fc_zscore']).plot.imshow(
    #         x='col_stim',
    #         y='row_stim',
    #         cmap='RdBu_r',
    #         # vmin=-1, vmax=1,
    #         center=0,
    #         yincrease=False,
    #         xticks=list(range(1, 51, 3)),
    #         yticks=list(range(1, 51, 3)),
    #         )
    #
    # img_rdm.axes.set_xticklabels(img_rdm.axes.get_xticklabels(), rotation=90)
    # img_rdm.axes.set_box_aspect(1)
    #
    # plt.tight_layout()
    # plt.show()

    row_tick_labels, row_tick_locs = ryeutils.main.get_tick_labels_and_locs(
            da_rsm[row_coord].to_numpy())
    col_tick_labels, col_tick_locs = ryeutils.main.get_tick_labels_and_locs(
            da_rsm[col_coord].to_numpy())

    # %%
    df_plot = da_rsm.to_pandas()

    row_coords_to_drop = list(set(df_plot.index.names) - {row_coord})
    col_coords_to_drop = list(set(df_plot.columns.names) - {col_coord})

    df_plot = (da_rsm.to_pandas()
               .droplevel(row_coords_to_drop, axis='index')
               .droplevel(col_coords_to_drop, axis='columns')
               )
    # %%
    default_heatmap_kws = dict(
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            )
    if heatmap_kws is None:
        use_heatmap_kws = default_heatmap_kws.copy()
    else:
        use_heatmap_kws = default_heatmap_kws.copy()
        use_heatmap_kws.update(**heatmap_kws)
    ax = sns.heatmap(df_plot,
                     **use_heatmap_kws,
                     xticklabels=True, yticklabels=True,
                     square=True,
                     ax=ax, cbar_ax=cbar_ax)

    ax.set_xticks(row_tick_locs, row_tick_labels, rotation=90, fontsize=10)
    ax.set_yticks(col_tick_locs, col_tick_labels, fontsize=10)

    return ax


def plot_individual_and_mean_rdms(da_rdm_concat, row_coord='row_stim', col_coord='col_stim',
                                  title_coord=None, figsize=None):


    n_acqs = da_rdm_concat.shape[da_rdm_concat.dims.index('acq')]

    if figsize is None:
        figsize = (30, 10)
    fig = plt.figure(figsize=(figsize))
    # if n_acqs >= 8:
    #     fig = plt.figure(figsize=(30, 10))
    # else:
    #     fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, n_acqs),  # creates 2x2 grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     cbar_mode='edge',
                     cbar_size='5%'
                     )

    da_rdm_mean = da_rdm_concat.mean(dim='acq')
    da_rdm_std = da_rdm_concat.std(dim='acq')

    # plot mean RDM
    # ----------------
    ax_mean = grid.axes_row[0][0]
    ax_mean = plot_rsm_heatmap(da_rdm_mean, row_coord=row_coord, col_coord=col_coord,
                               ax=ax_mean, cbar_ax=ax_mean.cax)
    ax_mean.set_title('mean')

    # plot stdev RDM
    # ----------------
    ax_std = grid.axes_row[0][1]
    ax_std = plot_rsm_heatmap(da_rdm_std, row_coord=row_coord, col_coord=col_coord,
                              ax=ax_std, cbar_ax=ax_std.cax)
    ax_std.set_title('std')

    # plot individual RDMs
    # ------------------------
    da_rdm_list = [da for iaq, da in da_rdm_concat.groupby('acq')]
    # da_rdm_list.append(da_rdm_concat.mean('acq'))
    # df_rdm_list = [da.to_pandas() for da in da_rdm_list]

    for i, (ax, da) in enumerate(zip(grid.axes_row[1],
                                     da_rdm_list)
                                 ):
        ax = plot_rsm_heatmap(da, row_coord=row_coord, col_coord=col_coord,
                              ax=ax, cbar_ax=ax.cax)
        ax.set(xlabel="", ylabel="", facecolor='0.7')

        if title_coord is not None:
            ax.set_title(da[title_coord].item())
    return fig


def plot_trial_and_stim_rdms(da_trial_rsm, da_stim_rsm, heatmap_kws=None):
    fig, axarr = plt.subplots(1, 2,
                              figsize=(12, 5),
                              tight_layout=True,
                              gridspec_kw=dict(
                                      width_ratios=[1, 1],
                                      # height_ratios=[1, 1]
                                      )
                              )

    ax1 = axarr[0]
    ax1 = plot_rsm_heatmap(da_trial_rsm,
                           ax=ax1,
                           heatmap_kws=heatmap_kws
                           )
    # ax1.set_title(f"metric={da_trial_rsm.attrs['rdm.metric']}")
    ax1.set(xlabel='', ylabel='')

    ax2 = axarr[1]
    ax2 = plot_rsm_heatmap(da_stim_rsm,
                           row_coord='stim_row',
                           col_coord='stim_col',
                           ax=ax2,
                           heatmap_kws=heatmap_kws
                           )
    ax2.set(xlabel='', ylabel='')
    # ax2.set_title(f"metric={da_stim_rsm.attrs['rdm.metric']}")

    fig.suptitle("")
    return axarr
