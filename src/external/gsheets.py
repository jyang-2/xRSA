from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Generator, Sequence, Union, Any
from urllib.error import URLError
import warnings
import re


def csv_link_from_sheet_id(sheet_id: str, gid: int):
    """Get csv export link from google sheet info.

    Args:
        sheet_id (str): google sheet ID
        gid (int): sheet id (default 0 for 1st sheet)

    Returns:
        str: .csv export url
    """
    # return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid:d}"
    gsheet_link = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid:d}"
    return gsheet_link


def parse_browser_url(url):
    """Get sheet_id and gid from browser url.

    Args:
        url (str): google sheets browser link

    Returns:
        gsheet_id (str): google sheet ID
        gid (int): sheet # id
    """
    pattern = "https://docs.google.com/spreadsheets/d/(\w+)/edit#gid=(\d+)"

    url_parts = re.search(pattern, url)
    gsheet_id, gid = url_parts.groups()
    gid = int(gid)

    return gsheet_id, gid


def browser2csv(url):
    """Converts browser url to csv export link.

    Args:
        url (str): browser url, copied when editing google spreadsheet in browser

    Returns:
        str: link to use when reading spreadsheet into pandas

    Examples:
        >>>
    """
    sheet_id, gid = parse_browser_url(url)
    return csv_link_from_sheet_id(sheet_id, gid)
