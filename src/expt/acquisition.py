from pathlib import Path
import numpy as np
import json
import xarray as xr
from attrs import define, field
import attrs


def get_proc_dir(file):
    proc_idx = file.parts.index('processed_data')
    return Path("").joinpath(*file.parts[:proc_idx + 1])


def get_date_imaged_dir(file, proc_dir=None):
    if proc_dir is None:
        proc_dir = get_proc_dir(file)
    return proc_dir.joinpath(file.relative_to(proc_dir).parts[0])


def get_fly_dir(file, proc_dir=None):
    if proc_dir is None:
        proc_dir = get_proc_dir(file)
    return proc_dir.joinpath(*file.relative_to(proc_dir).parts[:2])


def get_mov_dir(file, proc_dir=None):
    if proc_dir is None:
        proc_dir = get_proc_dir(file)
    return proc_dir.joinpath(*file.relative_to(proc_dir).parts[:3])


def load_stim_list(stim_list_file):
    with open(stim_list_file, 'r') as f:
        stim_list = json.load(f)
    return stim_list['stim_list_flatstr']


# %%

@define
class Acquisition:
    """Keeps track of directories and metadata files for ThorImage expt and analysis
    directories"""
    date_imaged: str
    fly_num: int
    thorimage_name: str
    proc_dir: Path
    thorsync_name: str = field(init=False)
    mov_dir: Path = field(init=False)
    timestamps_file: Path = field(init=False)
    experiment_xml_file: Path = field(init=False)
    stim_list_file: Path = field(init=False)
    stat_file: Path = field(init=False)
    timestamps: dict = field(init=False)
    stim_list: list = field(init=False)

    # def __init__(self, date_imaged: str, fly_num: int, thorimage_name: str,
    #              proc_dir: Path,
    #              stat_file: Path = None):
    def __init__(self,
                 date_imaged: str,
                 fly_num: int,
                 thorimage_name: str,
                 proc_dir: Path,
                 stat_file: Path = None):
        self.__attrs_init__(date_imaged, fly_num, thorimage_name, proc_dir)
        self.stat_file = stat_file

    @classmethod
    def from_stat_file(cls, stat_file):
        proc_dir = get_proc_dir(stat_file)
        date_imaged = get_date_imaged_dir(stat_file).name
        fly_num = int(get_fly_dir(stat_file).name)
        thorimage_name = get_mov_dir(stat_file).name

        return cls(date_imaged=date_imaged,
                   fly_num=fly_num,
                   thorimage_name=thorimage_name,
                   proc_dir=proc_dir,
                   stat_file=stat_file)

    def __attrs_post_init__(self):
        self.mov_dir = self.proc_dir.joinpath(self.date_imaged, str(self.fly_num),
                                              self.thorimage_name)
        if self.mov_dir.joinpath('timestamps.npy').is_file():
            self.timestamps_file = self.mov_dir.joinpath('timestamps.npy')
        if self.mov_dir.joinpath('Experiment.xml').is_file():
            self.experiment_xml_file = self.mov_dir.joinpath('Experiment.xml')
        if self.mov_dir.joinpath('stim_list.json').is_file():
            self.stim_list_file = self.mov_dir.joinpath('stim_list.json')

    def title(self, style='filepath'):
        if style == 'filepath':
            tstr = f"{self.date_imaged}/{self.fly_num:d}/{self.thorimage_name}"
        elif style == 'human':
            tstr = "{} fly {:02d}: {}".format(self.date_imaged,
                                              self.fly_num,
                                              self.thorimage_name)
        return tstr

    def filename_base(self):
        return f"{self.date_imaged}__fly{self.fly_num:02d}__{self.thorimage_name}"

    def load_timestamps(self):
        self.timestamps = np.load(self.timestamps_file, allow_pickle=True).item()

    def load_stim_list(self):
        self.stim_list = load_stim_list(self.stim_list_file)
