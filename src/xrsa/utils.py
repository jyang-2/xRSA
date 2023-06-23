import xarray as xr


def convert_stim_2_odor(ds, coord_name):
    stim_list = ds[coord_name].values
    odor_list = [item.split(' @ ')[0] for item in stim_list]
    return ds.assign_coords({coord_name: (ds[coord_name].dims[0], odor_list)})
