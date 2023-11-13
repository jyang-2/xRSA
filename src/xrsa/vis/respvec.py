import xarray as xr
import matplotlib.pyplot as plt
import ryeutils
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})


def plot_trials(da_respvec, stim_coord='stim',
                cell_dim='cells', trial_dim='trials',
                metric='correlation', method='average'):
    """

    Args:
        da_respvec (xr.DataArray): response vectors (such as mean peak amplitudes), cells x trials
        stim_coord (str): coord name containing stimuli (default 'stim')
        cell_dim (str): cell dimension name (default 'cells')
        trial_dim (str): trial dimension name (default 'stim')
        metric (str): distance metric used for clustering (see scipy.spatial.distance.pdist)
        method (str): Linkage method to use for calculating clusters.

    Returns:
        g (sns.ClusterGrid): Respvec clustermap
    """

    stim_tick_labels, stim_tick_locs = ryeutils.main.get_tick_labels_and_locs(
            da_respvec[stim_coord].to_numpy())

    df_plot = da_respvec.transpose(cell_dim, trial_dim).to_pandas()

    g = sns.clustermap(df_plot, cmap='vlag',
                       center=0,
                       robust=True,
                       xticklabels=True, yticklabels=False,
                       figsize=(8.5, 11),
                       metric=metric,
                       method=method,
                       dendrogram_ratio=(0.3, 0.2),
                       cbar_pos=(.05, .825, .025, .15)
                       # ax=ax, cbar_ax=cbar_ax
                       )
    g.ax_heatmap.set_title(f'metric={metric}, linkage={method}',
                           fontsize=10)

    return g
