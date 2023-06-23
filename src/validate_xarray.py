"""Data structure validation schema for xarray files, by pipeline step

Pipeline steps:
- convert_suite2p_outputs
- convert_suite2p_trials
-
"""
import numpy as np
import xarray_schema
from xarray_schema import DataArraySchema, DatasetSchema, CoordsSchema

coords_schema = CoordsSchema(
        coords={

            }
        )

iscell_schema = CoordsSchema(dtype=np.int, dims=['cells'], name='iscell')
cellprob_schema = DataArraySchema(dtype=np.int, dims=['cells'], name='cellprob')

da_schema_output_base = DataArraySchema(
        dtype=np.float32,
        dims=['cells', 'time'],
        coords={
            'iscell': iscell_schema,
            'cellprob': cellprob_schema
            }
        )

ds_schema_output_base = DatasetSchema({'F': da_schema_output_base,
                                       'Fc': da_schema_output_base,
                                       'Fneu': da_schema_output_base,
                                       'spks': da_schema_output_base
                                       })

suite2p_outputs_base = {'da_schema': da_schema_output_base,
                        'ds_schema': ds_schema_output_base}
