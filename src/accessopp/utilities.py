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

def weighted_median(
        df: pd.DataFrame, val_col: str, weight_col: str
    ) -> float:
    """ Calculate weighted median of a dataframe.

    Args:
        df: DataFrame containing the values and weights. 
        val_col: Name of the column containing the values. 
        weight_col: Name of the column containing the weights.

    Returns:
        Weighted median of the values. 
        
    """
    df = df.sort_values(val_col)

    # Take the weighted list and expand it to one row per trip
    es_list = []
    for _, row in df.iterrows():
        s = pd.Series([row[val_col]] * int(round(row[weight_col], 0)))
        es_list.append(s)
    es = pd.concat(es_list)
    # Calculate the median of this list using standard pandas.Series method
    return es.median()

