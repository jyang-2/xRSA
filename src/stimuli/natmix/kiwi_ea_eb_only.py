import pandas as pd
from itertools import product
from pathlib import Path
import json

stim_ord_folder = Path("/local/matrix/Remy-Data/projects/natural_mixtures/"
                       "analysis_outputs/stim_ord/kiwi_ea_eb_only")

conc_by_odor = {
    'ea': [-6.2, -5.2, -4.2],
    'eb': [-5.5, -4.5, -3.5]
    }

grouped_by_ea = [
    'pfo @ 0.0',
    'ea @ -6.2',
    'ea @ -5.2',
    'ea @ -4.2',
    'eb @ -5.5',
    'eb @ -4.5',
    'eb @ -3.5',
    'ea @ -6.2, eb @ -5.5',
    'ea @ -6.2, eb @ -4.5',
    'ea @ -6.2, eb @ -3.5',
    'ea @ -5.2, eb @ -5.5',
    'ea @ -5.2, eb @ -4.5',
    'ea @ -5.2, eb @ -3.5',
    'ea @ -4.2, eb @ -5.5',
    'ea @ -4.2, eb @ -4.5',
    'ea @ -4.2, eb @ -3.5'
    ]

grouped_by_eb = [
    'pfo @ 0.0',
    'ea @ -6.2',
    'ea @ -5.2',
    'ea @ -4.2',
    'eb @ -5.5',
    'eb @ -4.5',
    'eb @ -3.5',
    'ea @ -6.2, eb @ -5.5',
    'ea @ -5.2, eb @ -5.5',
    'ea @ -4.2, eb @ -5.5',
    'ea @ -6.2, eb @ -4.5',
    'ea @ -5.2, eb @ -4.5',
    'ea @ -4.2, eb @ -4.5',
    'ea @ -6.2, eb @ -3.5',
    'ea @ -5.2, eb @ -3.5',
    'ea @ -4.2, eb @ -3.5'
    ]

stim_ord_by = {'ea': grouped_by_ea,
               'eb': grouped_by_eb}

grid_ord = [
    'pfo @ 0.0', 'eb @ -5.5', 'eb @ -4.5', 'eb @ -3.5',
    'ea @ -6.2', 'ea @ -6.2, eb @ -5.5', 'ea @ -6.2, eb @ -4.5', 'ea @ -6.2, eb @ -3.5',
    'ea @ -5.2', 'ea @ -5.2, eb @ -5.5', 'ea @ -5.2, eb @ -4.5', 'ea @ -5.2, eb @ -3.5',
    'ea @ -4.2', 'ea @ -4.2, eb @ -5.5', 'ea @ -4.2, eb @ -4.5', 'ea @ -4.2, eb @ -3.5',
    ]

def _make_df_stim_ord():
    ea_stim = [None, 'ea @ -6.2', 'ea @ -5.2', 'ea @ -4.2']
    eb_stim = [None, 'eb @ -5.5', 'eb @ -4.5', 'eb @ -3.5']

    stim_list = list(product(ea_stim, eb_stim))

    df_stim_ord = pd.DataFrame(stim_list, columns=['component_a', 'component_b'])

    stim = []
    n_components = []

    for row in df_stim_ord.itertuples():
        if row.component_a is None and row.component_b is None:
            stim.append('pfo @ 0.0')
            n_components.append(0)
        elif row.component_a is None:
            stim.append(row.component_b)
            n_components.append(1)
        elif row.component_b is None:
            stim.append(row.component_a)
            n_components.append(1)
        else:
            stim.append(f"{row.component_a}, {row.component_b}")
            n_components.append(2)

    df_stim_ord['stim'] = stim
    df_stim_ord['n_components'] = n_components

    return df_stim_ord


def _save_df_stim_ord():
    df_stim_ord = _make_df_stim_ord()
    filename = stim_ord_folder.joinpath('df_stim_ord.pkl')
    df_stim_ord.to_pickle(filename)
    print(filename)
    return filename
