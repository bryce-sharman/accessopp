from geopandas import GeoSeries
from importlib.resources import files
from os import environ
import numpy as np
import pandas as pd
import pandas.testing as tm
from pathlib import Path
import pytest
from shapely.geometry import Point

import accessopp
import accessopp.io as io
from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN

@pytest.fixture
def matrixdata_path():
    src_path = files(accessopp)
    root_path = src_path.parents[1]
    return root_path / "tests" / "test_data" / "matrix"

@pytest.fixture
def pointsdata_path():
    src_path = files(accessopp)
    root_path = src_path.parents[1]
    return root_path / "tests" / "test_data" / "points"

def test_read_pts_file_from_csv_no_other(pointsdata_path):
    fp = pointsdata_path / 'small_origins.csv'
    gs = io.read_points_from_csv(fp, 'id', 'longitude', 'latitude')
    idx = pd.Index([11, 12, 13], name='id')
    ref_gs = GeoSeries(
        index=idx,
        data=[
            Point(-79.52809, 43.61918), 
            Point(-79.52353, 43.62561), 
            Point(-79.51702, 43.62160)
        ],
        crs="EPSG:4326"
    )
    tm.assert_series_equal(gs, ref_gs)

def test_read_pts_file_from_csv_with_1other(pointsdata_path):
    fp = pointsdata_path / 'small_origins.csv'
    gs, df = io.read_points_from_csv(fp, 'id', 'longitude', 'latitude', 'orig_wt')
    idx = pd.Index([11, 12, 13], name='id')
    ref_gs = GeoSeries(
        index=idx,
        data=[
            Point(-79.52809, 43.61918), 
            Point(-79.52353, 43.62561), 
            Point(-79.51702, 43.62160)
        ],
        crs="EPSG:4326"
    )
    tm.assert_series_equal(gs, ref_gs)
    ref_df = pd.DataFrame(
        index=idx,
        columns=['orig_wt'],
        data = [100, 250, 500]
    )
    tm.assert_frame_equal(df, ref_df)

def test_read_pts_file_from_csv_with_2others(pointsdata_path):
    fp = pointsdata_path / 'small_origins.csv'
    gs, df = io.read_points_from_csv(
        fp, 'id', 'longitude', 'latitude', ['orig_wt', 'orig_wt2'])
    idx = pd.Index([11, 12, 13], name='id')
    ref_gs = GeoSeries(
        index=idx,
        data=[
            Point(-79.52809, 43.61918), 
            Point(-79.52353, 43.62561), 
            Point(-79.51702, 43.62160)
        ],
        crs="EPSG:4326"
    )
    tm.assert_series_equal(gs, ref_gs)
    ref_df = pd.DataFrame(
        index=idx,
        columns=['orig_wt', 'orig_wt2'],
        data = [[100, 55.5], [250, 66.6], [500, 77.7]]
    )
    tm.assert_frame_equal(df, ref_df)

def test_read_pts_file_from_csv_mutl_wts_myid(pointsdata_path):
    fp = pointsdata_path / 'small_origins_newid.csv'
    gs, df = io.read_points_from_csv(
        fp, 'my_id', 'longitude', 'latitude', ['orig_wt', 'orig_wt2'])
    idx = pd.Index([11, 12, 13], name='id')
    ref_gs = GeoSeries(
        index=idx,
        data=[
            Point(-79.52809, 43.61918), 
            Point(-79.52353, 43.62561), 
            Point(-79.51702, 43.62160)
        ],
        crs="EPSG:4326"
    )
    tm.assert_series_equal(gs, ref_gs)
    ref_df = pd.DataFrame(
        index=idx,
        columns=['orig_wt', 'orig_wt2'],
        data = [[100, 55.5], [250, 66.6], [500, 77.7]]
    )
    tm.assert_frame_equal(df, ref_df)

def test_read_matrix(matrixdata_path):
    """ Tests file read of a simple matrix in tall (stacked) format. """
    read_mat = io.read_matrix(matrixdata_path / 'small_matrix.csv')
    mi = pd.MultiIndex.from_product(
        [[1, 2, 3], [100, 101, 102]], names=INDEX_COLUMNS)
    ref_mat = pd.Series(
        index = mi,
        data=[1000.1, 1001.1, 1002.1, 
              2000.1, 2001.1, 2002.1, 
              3000.1, 3001.1, 3002.1
        ],
        name=COST_COLUMN
    )
    tm.assert_series_equal(read_mat, ref_mat)








