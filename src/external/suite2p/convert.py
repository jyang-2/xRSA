from pathlib import Path
import numpy as np
import xarray as xr
from scipy.stats import zscore
from . import helpers


def outputs_2_xarray_base(stat_file):
    """Converts suite2p outputs into an xarray dataset, no extra metadata added.

    Args:
        stat_file (Path): path to stat.npy file in folder holding suite2p outputs.
    Returns:
        (xr.Dataset): ds_suite2p_outputs
    """

    F = np.load(stat_file.with_name('F.npy'), allow_pickle=True)
    Fneu = np.load(stat_file.with_name('Fneu.npy'), allow_pickle=True)
    spks = np.load(stat_file.with_name('spks.npy'), allow_pickle=True)

    Fc = F - 0.7 * Fneu
    n_cells, T = Fc.shape

    # if iscell_filename is None:
    #     iscell_filename = 'iscell.npy'
    #
    iscell, cellprob = np.load(stat_file.with_name('iscell.npy'), allow_pickle=True).T
    # iscell = iscell.astype('int').squeeze()
    cellprob = cellprob.squeeze()

    stat = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    # zscore F, Fc
    F_zscore = zscore(F, axis=1)
    Fc_zscore = zscore(Fc, axis=1)

    data_vars = {'Fc': (["cells", "time"], Fc),
                 'F': (["cells", "time"], F),
                 'Fneu': (["cells", "time"], Fneu),
                 'spks': (["cells", "time"], spks),
                 'F_zscore': (["cells", "time"], F_zscore),
                 'Fc_zscore': (["cells", "time"], Fc_zscore),
                 }

    ds_suite2p_outputs = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                    cells=range(n_cells),
                    # iscell=('cells', iscell),
                    cellprob=('cells', cellprob)
                    )
            )

    return ds_suite2p_outputs
