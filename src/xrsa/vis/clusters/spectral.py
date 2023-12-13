import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np

def plot_cell_cluster_profiles(df_mean_cell_clusters, df_cell_cluster_counts, stim_ord=None):
    """

    Args:
        df_mean_cell_clusters (pd.DataFrame): should have dims ("cell_labels", "stim")
        df_cell_cluster_counts (pd.DataFrame): should have columns
        stim_ord:

    Returns:

    """
    fig_biclusters, (ax_spectral_1, ax_spectral_2) = \
        plt.subplots(1, 2, figsize=(12, 6.4))

    # mean cluster heatmap
    sns.heatmap(df_mean_cell_clusters,
                robust=True,
                center=0,
                cmap='vlag',
                square=True,
                ax=ax_spectral_1,
                xticklabels=True,
                )

    ax_spectral_1.set_xticklabels(ax_spectral_1.get_xticklabels(), fontsize=8)
    ax_spectral_1.set_yticklabels(ax_spectral_1.get_yticklabels(), rotation=0)

    sns.despine(ax=ax_spectral_1, left=True, bottom=True)
    ax_spectral_1.set_title("averaged cell clusters")

    # barplot
    sns.barplot(df_cell_cluster_counts,
                y='cell_cluster',
                x='fraction',
                orient='horizontal',
                color='tab:blue',
                ax=ax_spectral_2
                )
    for i, bar in enumerate(ax_spectral_2.containers):
        ax_spectral_2.bar_label(bar, fmt='%.3f',
                                padding=2,
                                fontsize=9
                                )

    ax_spectral_2.set_yticklabels(ax_spectral_2.get_yticklabels(), rotation=0)
    sns.despine(ax=ax_spectral_2, trim=True)
    ax_spectral_2.set_title("cell cluster counts")

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(bottom=0.25, top=0.88)
    return
