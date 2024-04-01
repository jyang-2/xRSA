import numpy as np
from pathlib import Path
from external.suite2p.iscells import get_iscell_suffix
from . import embedding
from scipy.stats import zscore


def make_Fc_zscore_file(stat_file):
    F = np.load(stat_file.with_name('F.npy'))
    Fneu = np.load(stat_file.with_name('Fneu.npy'))

    Fc = F - 0.7 * Fneu
    Fc_zscore = zscore(Fc, axis=1)

    np.save(stat_file.with_name('Fc_zscore.npy'), Fc_zscore)


def prep_fluorescence_for_rastermap(stat_file, data_var, iscell_filename):
    iscell_filepath = stat_file.with_name(iscell_filename)
    iscell_stem = iscell_filepath.stem
    suffix = get_iscell_suffix(iscell_filename)

    iscell = np.load(iscell_filepath)

    F_ori = np.load(stat_file.with_name(f'{data_var}.npy'))
    F = F_ori[iscell[:, 0] == 1, :]

    save_dir = stat_file.with_name('rmap').joinpath(iscell_stem)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save iscell*.npy
    np.save(save_dir.joinpath(iscell_filename), iscell)
    np.save(save_dir.joinpath(f'{data_var}_{suffix}.npy'), F)
    # save


def rmap_2_iscell_simple(rmap_file, iscell_file=None, save_name='iscell_rmap.npy'):
    emb = np.load(rmap_file, allow_pickle=True).item()
    if iscell_file is None:
        iscell_file = rmap_file.with_name('iscell.npy')

    iscell_rmap = embedding.rmap_2_rmap_arr(emb)
    file = rmap_file.with_name(save_name)

    np.save(file, iscell_rmap)

    return file
