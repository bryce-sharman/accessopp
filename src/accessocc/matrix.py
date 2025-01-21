""" Functions to help read and write matrices from/to files. """

from pathlib import Path
import pandas as pd

INDEX_NAMES = ["origin_id", "destination_id"]


def read_matrix(fp, name="matrix"):
    """ Reads a previously calculated matrix.

    The matrix format is expected to match that produced by the OTP2 and r5py 
    matrix computation classes provided in this package. This format is as follows:

    ```
    from_id,to_id,travel_time
    origin_id_1,destination_id_1,travel_time_1_1
    origin_id_2,destination_id_2,travel_time_1_2
    origin_id_3,destination_id_3,travel_time_1_3
    ```

    Arguments
    ---------
    fp: str or pathlib.Path
        Filepath to file containing the matrix
    name: str, optional
        Name to set on the pandas dataframe, defaults to "matrix"

    Returns
    -------
    pandas DataFrame

    """
    fp = Path(fp)
    if not fp.is_file():
        raise FileNotFoundError(f"File {fp} does not exist.")
    df = pd.read_csv(fp, usecols=['from_id', 'to_id', 'travel_time'], index_col=False)
    df.name = name
    return df


def write_matrix(df, fp):
    """ Writes a matrix to matrix file, using the following format.
    
    ```
    from_id,to_id,travel_time
    origin_id_1,destination_id_1,travel_time_1_1
    origin_id_2,destination_id_2,travel_time_1_2
    origin_id_3,destination_id_3,travel_time_1_3
    ```

    Arguments
    ---------
    df: pandas.DataFrame
        cost matrix in wide format
    fp: str or pathlib.Path
        Filepath to file containing the matrix

    Returns
    -------
    None

    """
    assert df.columns.equals(pd.Index(['from_id', 'to_id', 'travel_time']))
    df.to_csv(fp, index=False)
