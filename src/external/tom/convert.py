import copy
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import preprocessing
import ryeutils
import xrsa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def df_ori_2_dataarray(df_ori):
    # fixed_index = df_ori.index.to_frame().convert_dtypes()
    # mi_row = pd.MultiIndex.from_frame(fixed_index)
    #
    # fixed_columns = df_ori.columns.to_frame().convert_dtypes()
    # mi_col = pd.MultiIndex.from_frame(fixed_columns)

    da_ori = xr.DataArray(df_ori.to_numpy(),
                          dims=['row', 'col'],
                          coords=dict(
                                  row=df_ori.index,
                                  col=df_ori.columns
                                  ))
    da_ori = da_ori.reset_index('row')
    da_ori = da_ori.reset_index('col')
    return da_ori


def fix_da_ori(da_ori, single_odor_stim=True):
    if single_odor_stim:
        da_fixed = da_ori.drop_vars(['is_pair', 'odor2'])
    else:
        da_fixed = da_ori.copy(deep=True)

    # da_fixed = da_fixed.where(da_fixed.panel == panel_name, drop=True)
    for coord_name in ['panel', 'odor1', 'roi']:
        da_fixed[coord_name] = da_fixed[coord_name].astype(str)
    da_fixed['repeat'] = da_fixed['repeat'].astype('int')
    da_fixed['fly_num'] = da_fixed['fly_num'].astype('int')
    da_fixed['date'] = da_fixed['date'].dt.date
    da_fixed['date'] = da_fixed['date'].astype(str)

    return da_fixed


