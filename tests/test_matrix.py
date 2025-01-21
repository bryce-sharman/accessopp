from importlib.resources import files
from os import environ
import numpy as np
import pandas as pd
import pandas.testing as tm
from pathlib import Path
import unittest

import accessocc
from accessocc.matrix import read_matrix, write_matrix


class TestMatrixHelperFunctions(unittest.TestCase):

    def setUp(self):
        # Find the path to to the test-directions
        # The src path is to the src/accessopp directory. 
        src_path = files(accessocc)
        root_path = src_path.parents[1]
        self.testdata_path = root_path / "tests" / "test_data" / "matrix"

    def test_read_matrix(self):
        """ Tests file read of a simple matrix in tall (stacked) format. """
        read_mat = read_matrix(self.testdata_path / 'small_matrix.csv')
        ref_mat = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                [1,100,1000.1],
                [1,101,1001.1],
                [1,102,1002.1],
                [2,100,2000.1],
                [2,101,2001.1],
                [2,102,2002.1],
                [3,100,3000.1],
                [3,101,3001.1],
                [3,102,3002.1]
            ]
        )
        self._test_matrix(read_mat, ref_mat)
        self.assertTrue(read_mat.name=="matrix")

    def test_read_matrix_named(self):
        """ Tests file read of a simple matrix in tall (stacked) format. """
        my_name = "new_name"
        read_mat = read_matrix(self.testdata_path / 'small_matrix.csv', my_name)
        ref_mat = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                [1,100,1000.1],
                [1,101,1001.1],
                [1,102,1002.1],
                [2,100,2000.1],
                [2,101,2001.1],
                [2,102,2002.1],
                [3,100,3000.1],
                [3,101,3001.1],
                [3,102,3002.1]
            ]
        )
        self._test_matrix(read_mat, ref_mat)
        self.assertTrue(read_mat.name==my_name)

    def test_write_read_roundtrip_intids(self):
        df = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                    [1, 100, 1000.1],
                    [1, 101, 1001.1],
                    [1, 102, 1002.1],
                    [2, 100, 2000.1],
                    [2, 101, 2001.1],
                    [2, 102, 2002.1],
                    [3, 100, 3000.1],
                    [3, 101, 3001.1],
                    [3, 102, 3002.1]
                ]
        )
        # Find the Windows temp directory, clear if it already exists
        temp_dir = Path(environ['TMP']) / 'OTP2' / 'testing'
        temp_dir.mkdir(parents=True, exist_ok=True)
        write_matrix(df, temp_dir / "test_round_trip.csv")

        df2 = read_matrix(temp_dir / "test_round_trip.csv")
        self._test_matrix(df2, df)

    def test_write_read_roundtrip_strids(self):
        df = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                    ['a', 'aaa', 1000.1],
                    ['a', 'bbb', 1001.1],
                    ['a', 'ccc', 1002.1],
                    ['b', 'aaa', 2000.1],
                    ['b', 'bbb', 2001.1],
                    ['b', 'ccc', 2002.1],
                    ['c', 'aaa', 3000.1],
                    ['c', 'bbb', 3001.1],
                    ['c', 'ccc', 3002.1]
                ]
        )
        # Find the Windows temp directory, clear if it already exists
        temp_dir = Path(environ['TMP']) / 'OTP2' / 'testing'
        temp_dir.mkdir(parents=True, exist_ok=True)
        write_matrix(df, temp_dir / "test_round_trip2.csv")

        df2 = read_matrix(temp_dir / "test_round_trip2.csv")
        self._test_matrix(df2, df)

    def _test_matrix(self, matrix, ref_matrix, rtol=1.0e-5, atol=1.0e-8):
        """ Test matrix helping function that checks two matrices. Raises Assertion error if test fails."""
        tm.assert_frame_equal(matrix, ref_matrix, rtol=rtol, atol=atol)
