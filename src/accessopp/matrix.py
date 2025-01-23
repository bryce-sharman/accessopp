""" Functions to help read and write matrices from/to files. """

from pathlib import Path
import pandas as pd

from accessopp.enumerations import INDEX_COLUMNS, ALL_COLUMNS

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
    matrix = pd.read_csv(fp, usecols=ALL_COLUMNS, index_col=INDEX_COLUMNS)
    matrix.squeeze()
    matrix.name = name
    return matrix


def write_matrix(matrix, fp):
    """ Writes a matrix to matrix file, using the following format.
    
    Arguments
    ---------
    df: pandas.DataFrame
        cost matrix in wide format
    fp: str or pathlib.Path
        Filepath to file containing the matrix

    """
    matrix.to_csv(fp, index=False)
