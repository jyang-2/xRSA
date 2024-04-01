import pandas as pd
from itertools import product
from pathlib import Path
import json

stim_ord_folder = Path("/local/matrix/Remy-Data/projects/natural_mixtures/"
                       "analysis_outputs/stim_ord/control1_top2_ramps")

conc_by_odor = {
    '1o3ol': [-5.0, -4.0, -3.0],
    '2h': [-7.0, -6.0, -5.0],
    }

grouped_by_1o3ol = [
    'pfo @ 0.0',
    '1o3ol @ -5.0',
    '1o3ol @ -4.0',
    '1o3ol @ -3.0',
    '2h @ -7.0',
    '2h @ -6.0',
    '2h @ -5.0',
    '1o3ol @ -5.0, 2h @ -7.0',
    '1o3ol @ -5.0, 2h @ -6.0',
    '1o3ol @ -5.0, 2h @ -5.0',
    '1o3ol @ -4.0, 2h @ -7.0',
    '1o3ol @ -4.0, 2h @ -6.0',
    '1o3ol @ -4.0, 2h @ -5.0',
    '1o3ol @ -3.0, 2h @ -7.0',
    '1o3ol @ -3.0, 2h @ -6.0',
    '1o3ol @ -3.0, 2h @ -5.0'
    ]

grouped_by_2h = [
    'pfo @ 0.0',
    '1o3ol @ -5.0',
    '1o3ol @ -4.0',
    '1o3ol @ -3.0',
    '2h @ -7.0',
    '2h @ -6.0',
    '2h @ -5.0',
    '1o3ol @ -5.0, 2h @ -7.0',
    '1o3ol @ -4.0, 2h @ -7.0',
    '1o3ol @ -3.0, 2h @ -7.0',
    '1o3ol @ -5.0, 2h @ -6.0',
    '1o3ol @ -4.0, 2h @ -6.0',
    '1o3ol @ -3.0, 2h @ -6.0',
    '1o3ol @ -5.0, 2h @ -5.0',
    '1o3ol @ -4.0, 2h @ -5.0',
    '1o3ol @ -3.0, 2h @ -5.0'
    ]

stim_ord_by = {'1o3ol': grouped_by_1o3ol,
               '2h': grouped_by_2h}

grid_ord = [
    'pfo @ 0.0', '2h @ -7.0', '2h @ -6.0', '2h @ -5.0',
    '1o3ol @ -5.0', '1o3ol @ -5.0, 2h @ -7.0', '1o3ol @ -5.0, 2h @ -6.0', '1o3ol @ -5.0, 2h @ -5.0',
    '1o3ol @ -4.0', '1o3ol @ -4.0, 2h @ -7.0', '1o3ol @ -4.0, 2h @ -6.0', '1o3ol @ -4.0, 2h @ -5.0',
    '1o3ol @ -3.0', '1o3ol @ -3.0, 2h @ -7.0', '1o3ol @ -3.0, 2h @ -6.0', '1o3ol @ -3.0, 2h @ -5.0',
    ]

def _make_df_stim_ord():
    stim_1o3ol = [None, '1o3ol @ -5.0', '1o3ol @ -4.0', '1o3ol @ -3.0']
    stim_2h = [None, '2h @ -7.0', '2h @ -6.0', '2h @ -5.0']

    stim_list = list(product(stim_1o3ol, stim_2h))

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
