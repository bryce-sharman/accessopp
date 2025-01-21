from datetime import datetime, timedelta
import geopandas as gpd
import numpy as np
import pandas as pd
import pandas.testing as tm
from shapely import Point
import unittest

from accessocc.travel_time_computer.r5py import R5PYTravelTimeComputer  

class TestR5PYTravelTimeComputer(unittest.TestCase):
    """ 
    The r5py class in this package is really just a wrapper for r5py, hence these tests are more
    about making sure that nothing changes instead of code accuracy.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r5i = R5PYTravelTimeComputer()
        cls.r5i.build_network_from_dir(r"C:\Users\bsharma2\AccessOpportunities\Test Data\tests")

    def setUp(self):
        origin_df = pd.DataFrame(
            columns = ['id', 'geometry'], 
            data = [[11, Point(-79.52809, 43.61918)], [12, Point(-79.52353, 43.62561)], [13, Point(-79.51702, 43.62160)]])
        self.origins = gpd.GeoDataFrame(origin_df, crs="EPSG:4326", geometry='geometry')

        destination_df = pd.DataFrame(
            columns = ['id', 'geometry'], 
            data = [[21, Point(-79.5012, 43.62924)], [23, Point(-79.49908, 43.62977)]])
        self.destinations = gpd.GeoDataFrame(destination_df, crs="EPSG:4326", geometry='geometry')

        # This OD combo is the Queensway & Kipling, and the Queensway at Park Lawn
        self.origin = Point(-79.52664, 43.62091)
        self.destination = Point(-79.49030, 43.62901)

        self.departure_1trip = datetime(2024,1,15,9,15,0)
        self.departure_interval = datetime(2024,1,15,9,0,0)


    def test_walk_matrix_default_speed_walking(self):
        """ Test walk travel time matrix. """
        ttm = self.r5i.compute_walk_traveltime_matrix(self.origins, self.destinations)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [[11, 21, 34], [11, 23, 36], [12, 21, 28], [12, 23, 31], [13, 21, 22], [13, 23, 25]]
        )
        self._test_matrix(ttm, ref_matrix)

    def test_walk_matrix_speed_walking_2_0(self):
        """ Test walk travel time matrix. """
        ttm = self.r5i.compute_walk_traveltime_matrix(self.origins, self.destinations, speed_walking=2.0)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [[11, 21, 85], [11, 23, 89], [12, 21, 71], [12, 23, 76], [13, 21, 56], [13, 23, 60]]
        )
        self._test_matrix(ttm, ref_matrix)

    def test_transit_matrix_default_speed_walking(self):
        """ Test walk travel time matrix. """
        ttm = self.r5i.compute_transit_traveltime_matrix(self.origins, self.destinations, self.departure_interval)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [[11, 21, 27], [11, 23, 28], [12, 21, 27], [12, 23, 28], [13, 21, 22], [13, 23, 24]]
        )
        self._test_matrix(ttm, ref_matrix)

    def test_transit_matrix_speed_walking_2_0(self):
        """ Test walk travel time matrix. """
        ttm = self.r5i.compute_transit_traveltime_matrix(self.origins, self.destinations, self.departure_interval, speed_walking=2.0)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [[11, 21, 40], [11, 23, 42], [12, 21, 44], [12, 23, 46], [13, 21, 37], [13, 23, 39]]
        )
        self._test_matrix(ttm, ref_matrix)        


    def _test_matrix(self, matrix, ref_matrix, rtol=1.0e-5, atol=1.0e-8):
        """ Test matrix helping function that checks two matrices. Raises Assertion error if test fails."""
        tm.assert_frame_equal(matrix, ref_matrix, rtol=rtol, atol=atol)
