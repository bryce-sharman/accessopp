from os import PathLike
from pathlib import Path
from shutil import rmtree

from geopandas import GeoDataFrame
from .enumerations import ID_COLUMN

def test_od_input(gdf):
    """ Test origin/destination input to ensure that it meets expected format.
    
    Arguments:
        gdf: 

    Raises:
        RuntimeError:
            Invalid format

    """
    if not isinstance(gdf, GeoDataFrame):
        raise RuntimeError("Points list are not a geopandas.GeoDataFrame")
    if ID_COLUMN not in gdf.columns:
        raise RuntimeError("Points list must have columns 'ID'.")
    

def test_file_existence(path_to_test: PathLike, msg_on_fail: str):
    """ Test if a file exists. """
    path_to_test = Path(path_to_test)
    if not path_to_test.is_file():
        raise FileNotFoundError(msg_on_fail)


def test_dir_existence(path_to_test: PathLike, msg_on_fail: str):
    """ Test if a directory exists. """
    path_to_test = Path(path_to_test)
    if not path_to_test.is_dir():
        raise FileNotFoundError(msg_on_fail)
    
def empty_directory_recursive(dir_path: PathLike):
    """ Remove all files in a directory, including all subdirectories. """
    for dpath in Path(dir_path).glob("**/*"):
        if dpath.is_file():
            dpath.unlink()
        elif dpath.is_dir():
            rmtree(dpath)