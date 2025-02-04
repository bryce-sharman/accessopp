from geopandas import GeoSeries
import numpy as np
from os import PathLike
import pandas as pd
from pathlib import Path
from shutil import rmtree
from typing import Tuple

from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN

    

def test_file_existence(path_to_test: PathLike, msg_on_fail: str) -> None:
    """ Test if a file exists. """
    path_to_test = Path(path_to_test)
    if not path_to_test.is_file():
        raise FileNotFoundError(msg_on_fail)


def test_dir_existence(path_to_test: PathLike, msg_on_fail: str) -> None:
    """ Test if a directory exists. """
    path_to_test = Path(path_to_test)
    if not path_to_test.is_dir():
        raise FileNotFoundError(msg_on_fail)
    
def empty_directory_recursive(dir_path: PathLike) -> None:
    """ Remove all files in a directory, including all subdirectories. """
    for dpath in Path(dir_path).glob("**/*"):
        if dpath.is_file():
            dpath.unlink()
        elif dpath.is_dir():
            rmtree(dpath)

def validate_origins_destinations(
        origins: GeoSeries, 
        destinations: GeoSeries
    ) -> Tuple[GeoSeries, GeoSeries]:
    if not isinstance(origins, GeoSeries):
        raise RuntimeError("origins are not a geopandas.GeoSeries")
    if destinations is None:
        destinations = origins.copy()
    elif not isinstance(destinations, GeoSeries):
            raise RuntimeError("destinations are not a geopandas.GeoSeries")
    return origins, destinations

def create_blank_ttmatrix(
        origins: GeoSeries, 
        destinations: GeoSeries
    ) -> pd.Series: 
    mi = pd.MultiIndex.from_product(
        [origins.index, destinations.index], names=INDEX_COLUMNS)
    return pd.Series(index=mi, name=COST_COLUMN, data=np.nan)