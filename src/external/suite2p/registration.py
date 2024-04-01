from pathlib import Path
import re
import numpy as np
import tifffile
from typing import Union, List, Any, Dict
import xarray as xr
from scipy.ndimage import percentile_filter

