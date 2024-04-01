from pathlib import Path
import numpy as np
from attrs import define, field


@define(kw_only=True)
class RastermapEmbedding:
    """
    Args:
        filename (Path): path to 'Fc_zscore_embedding.npy' saved by Rastermap
    """
    filename: Path = field(converter=Path)
    emb: dict = field(init=False)
    rmap_arr: np.array = field(init=False)
    n_total_rois: int = field(init=False)
    n_cluster_rois: int = field(init=False)
    n_clusters: int = field(init=False)

    def __attrs_post_init__(self):
        self.emb = np.load(self.filename, allow_pickle=True).item()
        self.rmap_arr = rmap_2_rmap_arr(self.emb)

        self.n_total_rois = self.emb['embedding'].shape[0]
        self.n_clusters = len(self.emb['user_clusters'])
        self.n_cluster_rois = int((self.rmap_arr[:, 0] > 0).sum())

        rmap_isclust = self.rmap_arr[:, 0][self.rmap_arr[:, 0] > 0]
        self.n_clusters = np.unique(rmap_isclust).size

    def save_rmap_arr(self, filename=None):
        if filename is None:
            filename = self.filename.with_name('rmap.npy')

        np.save(filename, self.rmap_arr)

        return filename


def rmap_2_rmap_arr(emb):
    """Convert rastermap embedding outputs to a numpy array, with columns embedding and user
    cluster.

    Args:
        emb (dict): rastermap embedding, loaded from 'Fc_zscore_embedding.npy'

    Returns:
        rmap_arr (np.array): (n_roi, 2) array

          Column 0 contains the user-defined clusters, where cluster labels start at 1,
          and 0 means not assigned to any cluster.

          Column 1 contains the embedding values.
    """
    n_rois = emb['embedding'].shape[0]
    rmap_arr = np.zeros((n_rois, 2))

    rmap_arr[:, 1] = emb['embedding'][:, 0]

    for iclust, clust in enumerate(emb['user_clusters']):
        rmap_arr[clust['ids'], 0] = iclust + 1

    return rmap_arr


def rmap_2_onehot_cluster(emb):
    n_rois = emb['embedding'].shape[0]
    n_clusters = len(emb['user_clusters'])

    onehot_clusters = np.zeros(n_rois, n_clusters)

    for iclust, clust in enumerate(emb['user_clusters']):
        onehot_clusters[clust['ids'], iclust] = 1

    return onehot_clusters


def rmap_file_2_rmap_arr(filename):
    emb = np.load(filename, allow_pickle=True).item()
    return rmap_2_rmap_arr(emb)
