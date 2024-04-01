import numpy as np


def rmap_compatible_with_iscell(emb, iscell):
    """Check if the rastermap output embedding, `emb` is compatible with `iscell`

    Args:
        emb (dict): rastermap outputs, loaded from "*_embedding.npy"
        iscell (np.ndarray): suite2p `iscell`, loaded from "suite2p/*/iscell.npy"

    Returns:
        bool: True if the # of rois in `emb` matches the # of good rois in 'iscell'

    """
    n_iscell_rois = int(iscell[:, 0].sum())
    n_rmap_rois = emb['embedding'].shape[0]

    return n_iscell_rois == n_rmap_rois


def expand_rmap_arr_against_iscell(rmap_arr, iscell):
    """Expands 'rmap_arr' against 'iscell', where rmap_arr is emb['embedding'].

    Args:
        rmap_arr (np.ndarray): rastermap embedding
        iscell (np.ndarray): suite2p `iscell.npy`

    Returns:
        rmap_arr_expanded (np.ndarray): 2-column array, where column 1 contains user clusters, \
          and row 2 contains the embeddings.

    """
    if rmap_arr.shape[0] != iscell[:, 0].sum():
        raise (ValueError('Rastermap embedding not compatible with iscell.'))

    n_rois = iscell.shape[0]

    rmap_arr_expanded = np.full((n_rois, 2), np.nan)
    rmap_arr_expanded[iscell[:, 0] == 1, :] = rmap_arr

    return rmap_arr_expanded

# def expand_rmap_file_against_iscell_file(rmap_file, iscell_file, rmap_save_name):
#     emb = np.load(rmap_file, allow_pickle=True).item()
#     rmap_arr = rmap_2_rmap_arr(emb)
#
#     iscell = np.load(iscell_file)
#     rmap_arr_expanded = expand_rmap_arr_against_iscell(rmap_arr, iscell)
#
#     save_file = iscell_file.with_name(rmap_save_name)
#     np.save(save_file, rmap_arr_expanded)
#     return save_file
#
# # def rmap_file_to_iscell_file(rmap_file, rmap_save_name):
