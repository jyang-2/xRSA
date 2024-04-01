"""Functions for parsing and fixing stimulus strings"""

import numpy as np
from typing import Union, List


def convert_conc_2_float(stim_list):
    """Extracts concentration values from a list of stimulus strings.

    Args:
        stim_list (List[str]): List of stimulus strings (ex: '1-6ol @ -3')

    Returns:
        stim_list_float (List[float]): List of stimuli w/ float concentrations

    Examples:

        >>> stim_list = ['1-6ol @ -3', '1-5ol @ -3', 'geos @ -4']
        >>> stimuli.fix.convert_conc_2_float(stim_list)
        Out[7]: ['1-6ol @ -3.0', '1-5ol @ -3.0', 'geos @ -4.0']
    """
    abbrev, conc = zip(*[item.split(" @ ") for item in stim_list])
    conc_arr = np.array(conc).astype(float)

    stim_list_float = [f"{a} @ {b}" for a, b in zip(abbrev, conc_arr)]

    if isinstance(stim_list, np.ndarray):
        stim_list_float = np.array(stim_list_float)

    return stim_list_float


def replace_abbrevs(stim_list, abbrev_to_replace):
    """
    Replace abbreviations in `stim_list` with abbreviations in `stim_to_replace`.

    Args:
        stim_list (Union[List[str], np.ndarray[str]]): List of stimuli, like '2-but @ -6.0'
        abbrev_to_replace (dict): Dictionary of abbreviations to replace

    Returns:
        fixed_stim_list (Union[List, np.ndarray]): stimuli w/ abbrevs replaced
    """
    abbrevs, concs = zip(*[item.split(" @ ") for item in stim_list])

    fixed_abbrevs = []
    for item in abbrevs:
        if item in abbrev_to_replace.keys():
            fixed_abbrevs.append(abbrev_to_replace[item])
        else:
            fixed_abbrevs.append(item)

    fixed_stim_list = [f"{a} @ {c}" for a, c in zip(fixed_abbrevs, concs)]

    if isinstance(stim_list, np.ndarray):
        fixed_stim_list = np.array(fixed_stim_list)

    return fixed_stim_list


def fix_stim(stim_list, abbrev_to_replace=None, conc_as_float=False):

    fixed_stim_list = stim_list.copy()

    if abbrev_to_replace is not None:
        fixed_stim_list = replace_abbrevs(fixed_stim_list, abbrev_to_replace)
    if conc_as_float:
        fixed_stim_list = convert_conc_2_float(fixed_stim_list)
    return fixed_stim_list
