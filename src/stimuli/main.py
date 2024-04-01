import numpy as np
import ryeutils
import xarray as xr
import re


def index_stim_coord(ds, coord_name,
                     stimulus_index_keys=('stim', 'stim_occ', 'trial_idx'),
                     suffix=None, prefix=None):
    """

    Args:
        ds (xr.Dataset): has stimulus coordinates
        coord_name (str): name of stimulus coordinate to index (usually `stim`)
        stimulus_index_keys:
        suffix:
        prefix:

    Returns:

    """
    print(coord_name)
    if len(ds[coord_name].dims) != 1:
        raise ValueError(f"Coord `{coord_name}` must be 1-D.")

    if suffix is not None and prefix is not None:
        raise Exception("Only one of `suffix` of `prefix` can have a set value.")

    stim_dim, = ds[coord_name].dims
    stim_list = ds[coord_name].to_numpy()
    print(f"stim_dim: {stim_dim}")

    # index stimuli
    # -------------
    stim_idx = ryeutils.index_stimuli(stim_list, include_trial_idx=True)

    if stimulus_index_keys is not None:
        stim_idx = {k: stim_idx[k] for k in stimulus_index_keys}

    # rename stimulus index keys, if either suffix or prefix is set
    # --------------------------------------------------------------
    if suffix is not None:
        stim_idx = {f"{k}{suffix}": v for k, v in stim_idx.items()}
    elif prefix is not None:
        stim_idx = {f"{prefix}{k}": v for k, v in stim_idx.items()}

    added_coord_names = list(stim_idx.keys())

    new_coords = {k: (stim_dim, v) for k, v in stim_idx.items()}
    return ds.assign_coords(new_coords)


def stim2odor(stim_list):
    """Strips '@ {{conc}} from stimulus strings

    Args:
        stim_list (Union[List, np.array]): list of stimulus strings w/ "{odor} @ {concentration}"
          format

    Returns:
        odor_list (Union[List, np.array]): List of odor strings only (concentration dropped)

    """
    odor_list = [item.split(" @ ")[0] for item in stim_list]

    if isinstance(stim_list, np.ndarray):
        odor_list = np.array(odor_list)
    return odor_list


def split_stim_list(stim_list):
    """Splits list of stimulus strings like `"{odor} @ {concentration}"` to `odor_list` and
    `conc_list`.

    Args:
        stim_list (List[str]): ["1-6ol @ -3.0", "ep @ -3.0", ...]
        conc_type:

    Returns:
         odor_list (List[str]):
    """

    stim_info = [item.split(" @ ") for item in stim_list]
    # odor_list, conc_list = list(zip(*stim_info))
    odor_list, conc_list = zip(*stim_info)
    odor_list = list(odor_list)
    conc_list = [float(item) for item in conc_list]
    return odor_list, conc_list


def split_stim_coord(ds, stim_coords, new_coord_names=None, substr_to_replace=None, ):
    """Splits stimulus coordinates like `"{odor} @ {conc}"` to 'abbrev' and 'conc'.

    Args:
        ds (Union[xr.DataArray, xr.Dataset]): dataset w/ stimulus coords
        stim_coords (List[str]): must be 1-D coords
        substr_to_replace (str): substring in stim_coords names to replace with 'abbrev' and
        'conc'. Only used if
            `new_coord_names` is not provided
        new_coord_names (Dict): mapping {coord_name: (abbrev_coord_name, conc_coord_name)}

    Returns:
        ds_with_split_stim_coords (Union[xr.DataArray, xr.Dataset]): dataset with abbrev and
        float coordinates added
        along the dimensions corresponding to the coordinates in `stim_coords`
    """
    if new_coord_names is None:
        if substr_to_replace is None:
            raise ValueError("Either `new_coord_names` or `substr_to_replace` must be provided.")
        else:
            # generate new abbrev. and conc. coord names
            new_coord_names = {}

            for coord_name in stim_coords:
                new_coord_names[coord_name] = (
                    re.sub(substr_to_replace, "abbrev", coord_name),
                    re.sub(substr_to_replace, "conc", coord_name)
                    )

    new_coords = {}
    for coord_name in stim_coords:
        dimname, *_ = ds.coords[coord_name].dims
        abbrev_coord_name, conc_coord_name = new_coord_names[coord_name]

        abbrevs, concs = split_stim_list(ds[coord_name].to_numpy())
        # ori coord name : (dimname, abbrevs)

        new_coords[abbrev_coord_name] = (dimname, abbrevs)
        new_coords[conc_coord_name] = (dimname, concs)

    print(new_coords)

    return ds.assign_coords(new_coords)
