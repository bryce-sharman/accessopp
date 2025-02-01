from geopandas import GeoSeries
from importlib.resources import files
from math import sqrt
from os import environ
import numpy as np
import pandas as pd
import pandas.testing as tm
from pathlib import Path
import pytest
from shapely.geometry import Point, Polygon

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

def test_rect_grid_generation():
    # Create a grid in UTM zone 17, 
    # note that x=500000 is the midpoint of a UTM zone
    centroids, polygons = io.build_rectangle_grid(
        Point(500000, 0), 100, 100, 300, 200, crs="EPSG:32617")
    idx = pd.Index(range(6), name='id')
    ref_centroids = GeoSeries(
        index=idx,
        data = [
            Point(500050, 50),
            Point(500050, 150),
            Point(500150, 50),
            Point(500150, 150),
            Point(500250, 50),
            Point(500250, 150),
        ]
    )
    ref_polygons = GeoSeries(
        index=idx,
        data=[
            Polygon((Point(500000, 0), Point(500100, 0), Point(500100, 100), Point(500000, 100), Point(500000, 0))),
            Polygon((Point(500000, 100), Point(500100, 100), Point(500100, 200), Point(500000, 200), Point(500000, 100))),
            Polygon((Point(500100, 0), Point(500200, 0), Point(500200, 100), Point(500100, 100), Point(500100, 0))),
            Polygon((Point(500100, 100), Point(500200, 100), Point(500200, 200), Point(500100, 200), Point(500100, 100))),
            Polygon((Point(500200, 0), Point(500300, 0), Point(500300, 100), Point(500200, 100), Point(500200, 0))),
            Polygon((Point(500200, 100), Point(500300, 100), Point(500300, 200), Point(500200, 200), Point(500200, 100)))
        ]
    )
    tm.assert_series_equal(centroids, ref_centroids)
    tm.assert_series_equal(polygons, ref_polygons)

def test_hex_grid_generation():
    incr = 100
    centroids, polygons = io.build_hexagonal_grid(
        Point(500000, 0), incr, 210, 260, crs="EPSG:32617")

    dy = 0.5 * incr                 # two hex rows define each increment
    el = incr / sqrt(3)             # edge length
    half_el = 0.5 * el              # half of the edge length
    dx = 3.0 * half_el
    idx = pd.Index(range(4), name='id')
    ref_centroids = GeoSeries(
        index=idx,
        data = [
            Point(500000 + 2.0*half_el, dy),
            Point(500000 + 2.0*half_el, 3 * dy),
            Point(500000 + 5.0*half_el, 2 * dy),
            Point(500000 + 5.0*half_el, 4 * dy),
        ],
        crs="EPSG:32617"
    )
    tm.assert_series_equal(centroids, ref_centroids)

    # todo: add polygon test