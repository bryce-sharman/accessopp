from importlib.resources import files
import geopandas as gpd
import pandas.testing as tm
from shapely.geometry import Point
import unittest


import accessocc
from accessocc.points import read_points_from_csv

class TestPointsIO(unittest.TestCase):


    def setUp(self):
        # Find the path to to the test-directions
        # The src path is to the src/accessopp directory. 
        src_path = files(accessocc)
        root_path = src_path.parents[1]
        self.testdata_path = root_path / "tests" / "test_data" / "points"


    def test_read_pts_file_from_csv_nowt(self):
        fp = self.testdata_path / 'small_origins.csv'
        gdf = read_points_from_csv(fp, 'id', 'longitude', 'latitude')
        ref_gdf = gpd.GeoDataFrame({
                'id': [11, 12, 13],
                'geometry': [Point(-79.52809, 43.61918), Point(-79.52353, 43.62561), Point(-79.51702, 43.62160)]
            },
            crs="EPSG:4326"
        )
        tm.assert_frame_equal(gdf, ref_gdf)


    def test_read_pts_file_from_csv_wt(self):
        fp = self.testdata_path / 'small_origins.csv'
        gdf = read_points_from_csv(fp, 'id', 'longitude', 'latitude', 'orig_wt')
        ref_gdf = gpd.GeoDataFrame({
                'id': [11, 12, 13],
                'orig_wt': [100, 250, 500],
                'geometry': [Point(-79.52809, 43.61918), Point(-79.52353, 43.62561), Point(-79.51702, 43.62160)]

            },
            crs="EPSG:4326"
        )
        tm.assert_frame_equal(gdf, ref_gdf)

    def test_read_pts_file_from_csv_mutl_wts(self):
        fp = self.testdata_path / 'small_origins.csv'
        gdf = read_points_from_csv(fp, 'id', 'longitude', 'latitude', ['orig_wt', 'orig_wt2'])
        ref_gdf = gpd.GeoDataFrame({
                'id': [11, 12, 13],
                'orig_wt': [100, 250, 500],
                'orig_wt2': [55.5, 66.6, 77.7],
                'geometry': [Point(-79.52809, 43.61918), Point(-79.52353, 43.62561), Point(-79.51702, 43.62160)]
            },
            crs="EPSG:4326"
        )
        tm.assert_frame_equal(gdf, ref_gdf)

    def test_read_pts_file_from_csv_mutl_wts_myid(self):
        fp = self.testdata_path / 'small_origins_newid.csv'
        gdf = read_points_from_csv(fp, 'my_id', 'longitude', 'latitude', ['orig_wt', 'orig_wt2'])
        ref_gdf = gpd.GeoDataFrame({
                'id': [11, 12, 13],
                'orig_wt': [100, 250, 500],
                'orig_wt2': [55.5, 66.6, 77.7],
                'geometry': [Point(-79.52809, 43.61918), Point(-79.52353, 43.62561), Point(-79.51702, 43.62160)]
            },
            crs="EPSG:4326"
        )
        tm.assert_frame_equal(gdf, ref_gdf)