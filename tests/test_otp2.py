from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import testing as tm
from shapely import Point
import unittest


# from accessopp.matrix import Matrix
from accessocc.travel_time_computer.otp2 import OTP2TravelTimeComputer

class Test_OTP2TravelTimeComputer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # Build the OTP graph and launch server.
        cls.otpi = OTP2TravelTimeComputer()
        cls.otpi.java_path = r"C:\Program Files\Microsoft\jdk-17.0.9.8-hotspot\bin\java"
        cls.otpi.otp_jar_path = r"C:\MyPrograms\otp\2_4\otp-2.4.0-shaded.jar"
        cls.otpi.memory_str = "1G"
        cls.otpi.build_network_from_dir(
            r"C:\Users\bsharma2\AccessOpportunities\Test Data\tests", overwrite=True, launch_server=True)

    def setUp(self):
        # There do seem to be differences between OTP and r5py, especially for walk trips. I don't understand what's
        # going on with this so for now we unfortunately need a higher tolerance between the two.
        self.RTOL_COARSE = 0.2
        self.ATOL_COARSE = 5

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

        # Add tests for unconnected point
        self.unconnected_pt = Point(-77.52809, 43.61918)
        self.dests_with_unconnected_pt_df = pd.DataFrame(
            columns = ['id', 'geometry'], 
            data = [[21, Point(-79.5012, 43.62924)], [23, self.unconnected_pt]])
        self.dests_with_unconnected_pt = gpd.GeoDataFrame(self.dests_with_unconnected_pt_df, crs="EPSG:4326", geometry='geometry')

#region Departure time testing
    def test_departure_time_testing_okay(self):
        self.otpi.test_departure_within_service_time_range(datetime(2024, 1, 15, 9, 0, 0))

    def test_departure_time_testing_too_early(self):
        with self.assertRaises(ValueError):
            self.otpi.test_departure_within_service_time_range(datetime(2023, 1, 15, 9, 0, 0))

    def test_departure_time_testing_too_late(self):
        with self.assertRaises(ValueError):
            self.otpi.test_departure_within_service_time_range(datetime(2025, 1, 15, 9, 0, 0))
#endregion

#region single trip tests
    # walk trips
    def test_walk_trip_default_speed_walking(self):
        """ Compare walk trip request using default walk speed of 5 km/hr."""
        # r5py gives a travel time of 37 minutes, make sure we're within same ballpark
        tt = self.otpi.compute_walk_trip_traveltime(self.origin, self.destination)
        r5py_time = 37
        self._test_single_tt(tt, r5py_time, self.ATOL_COARSE, self.ATOL_COARSE)

           
    def test_walk_trip_speed_walking_2_0(self):
        """ Compare walk trip request using walk speed of 2 km/hr."""
        # r5py gives a travel time of 92 minutes, make sure we're within same ballpark
        tt = self.otpi.compute_walk_trip_traveltime(self.origin, self.destination, speed_walking=2.0)
        r5py_time = 92
        self._test_single_tt(tt, r5py_time, self.ATOL_COARSE, self.ATOL_COARSE)

    def test_walk_trip_same_orig_dest(self):
        tt = self.otpi.compute_walk_trip_traveltime(self.origin, self.origin)
        self.assertEqual(tt, 0.0)

    def test_walk_trip_unconnected(self):
        tt = self.otpi.compute_walk_trip_traveltime(self.origin, self.unconnected_pt)
        self.assertTrue(np.isnan(tt))

    # Single transit trips
    def test_single_transit_trip_default_speed_walking(self):
        """ Test transit time for a single trip, which has a long wait for the bus. """
        tt = self.otpi.compute_transit_traveltime(self.origin, self.destination, self.departure_1trip)
        r5py_time = 15
        self._test_single_tt(tt, r5py_time, self.ATOL_COARSE, self.ATOL_COARSE)

    def test_single_transit_trip_speed_walking_2_0(self):
        """ Test transit time for a single trip, which has a long wait for the bus. """
        tt = self.otpi.compute_transit_traveltime(self.origin, self.destination, self.departure_1trip, speed_walking=2.0)
        r5py_time = 17
        self._test_single_tt(tt, r5py_time, self.ATOL_COARSE, self.ATOL_COARSE)

    def test_single_transit_trip_same_orig_dest(self):
        tt = self.otpi.compute_transit_traveltime(self.origin, self.origin, self.departure_1trip)
        self.assertEqual(tt, 0.0)

    def test_single_transit_trip_unconnected(self):
        tt = self.otpi.compute_transit_traveltime(self.origin, self.unconnected_pt, self.departure_1trip)
        self.assertTrue(np.isnan(tt))

    # Transit trips over time interval
    def test_transit_trip_interval_default_speed_walking(self):
        """ Test transit time for a single trip, which has a long wait for the bus. """
        tt = self.otpi.compute_interval_transit_traveltime(self.origin, self.destination, self.departure_interval)
        r5py_time = 23
        self._test_single_tt(tt, r5py_time, self.ATOL_COARSE, self.ATOL_COARSE)

    def test_transit_interval_speed_walking_2_0(self):
        """ Test transit time for a single trip, which has a long wait for the bus. """
        tt = self.otpi.compute_interval_transit_traveltime(self.origin, self.destination, self.departure_interval, speed_walking=2.0)
        r5py_time = 26
        self._test_single_tt(tt, r5py_time, self.ATOL_COARSE, self.ATOL_COARSE)

    def test_transit_interval_trip_same_orig_dest(self):
        tt = self.otpi.compute_transit_traveltime(self.origin, self.origin, self.departure_interval)
        self.assertEqual(tt, 0.0)

    def test_transit_interval_trip_unconnected(self):
        tt = self.otpi.compute_transit_traveltime(self.origin, self.unconnected_pt, self.departure_interval)
        print(tt)
        self.assertTrue(np.isnan(tt))

#endregion            

#region matrix tests
    # walk trips
    def test_walk_matrix_default_speed_walking(self):
        """ Test walk travel time matrix. """
        ttm = self.otpi.compute_walk_traveltime_matrix(self.origins, self.destinations)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[[11, 21, 34.0], [11, 23, 36.0], [12, 21, 28.0], [12, 23, 31.0], [13, 21, 22.0], [13, 23, 25.0]]
        )
        self._test_matrix(ttm, ref_matrix, self.RTOL_COARSE, self.ATOL_COARSE)

    def test_walk_matrix_speed_walking_2_0(self):
        """ Test walk travel time matrix. """
        ttm = self.otpi.compute_walk_traveltime_matrix(self.origins, self.destinations, speed_walking=2.0)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[[11, 21, 85.0], [11, 23, 89.0], [12, 21, 71.0], [12, 23, 76.0], [13, 21, 56.0], [13, 23, 60.0]]
        )
        self._test_matrix(ttm, ref_matrix, self.RTOL_COARSE, self.ATOL_COARSE)

    def test_walk_matrix_same_ods_1(self):
        ttm = self.otpi.compute_walk_traveltime_matrix(self.origins, self.origins, speed_walking=2.0)
        # This reference matrix is an OTP travel time and not from r5, hence we can do a tight check
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                [11, 11, 0.0], [11, 12, 36.0], [11, 13, 39.0], 
                [12, 11, 35.0], [12, 12, 0.0], [12, 13, 30.0], 
                [13, 11, 39.0], [13, 12, 30.0], [13, 13, 0.0]
            ] 
        )
        self._test_matrix(ttm, ref_matrix)

    def test_walk_matrix_same_ods_2(self):
        ttm = self.otpi.compute_walk_traveltime_matrix(self.origins, None, speed_walking=2.0)
        # This reference matrix is an OTP travel time and not from r5, hence we can do a tight check
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                [11, 11, 0.0], [11, 12, 36.0], [11, 13, 39.0], 
                [12, 11, 35.0], [12, 12, 0.0], [12, 13, 30.0], 
                [13, 11, 39.0], [13, 12, 30.0], [13, 13, 0.0]
            ] 
        )
        self._test_matrix(ttm, ref_matrix)

    def test_walk_matrix_with_unconnected_pt(self):
        ttm = self.otpi.compute_walk_traveltime_matrix(self.origins, self.dests_with_unconnected_pt)
        # This reference matrix is from r5py, with NaNs for second column, hence need coarse check.
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[[11, 21, 34.0], [11, 23, np.NaN], [12, 21, 28.0], [12, 23, np.NaN], [13, 21, 22.0], [13, 23, np.NaN]]
        )
        self._test_matrix(ttm, ref_matrix, self.RTOL_COARSE, self.ATOL_COARSE)

    def test_transit_matrix_default_speed_walking(self):
        """ Test walk travel time matrix. """
        ttm = self.otpi.compute_transit_traveltime_matrix(self.origins, self.destinations, self.departure_interval)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [[11, 21, 27.0], [11, 23, 28.0], [12, 21, 27.0], [12, 23, 28.0], [13, 21, 22.0], [13, 23, 24.0]]
        )
        self._test_matrix(ttm, ref_matrix, self.RTOL_COARSE, self.ATOL_COARSE)

    def test_transit_matrix_speed_walking_2_0(self):
        """ Test walk travel time matrix. """
        ttm = self.otpi.compute_transit_traveltime_matrix(self.origins, self.destinations, self.departure_interval, speed_walking=2.0)
        ref_matrix = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [[11, 21, 40.0], [11, 23, 42.0], [12, 21, 44.0], [12, 23, 46.0], [13, 21, 37.0], [13, 23, 39.0]]
        )
        self._test_matrix(ttm, ref_matrix, self.RTOL_COARSE, self.ATOL_COARSE)      


#endregion

    def _test_matrix(self, matrix, ref_matrix, rtol=1.0e-5, atol=1.0e-8):
        """ Test matrix helping function that checks two matrices. Raises Assertion error if test fails."""
        tm.assert_frame_equal(matrix, ref_matrix, rtol=rtol, atol=atol)

    def _test_single_tt(self, tt, ref_tt, rtol=1.0e-5, atol=1.0e-8):
        self.assertTrue(ref_tt / (1.0 + rtol) < tt < ref_tt * (1.0 + rtol))
        self.assertTrue(ref_tt - atol < tt < ref_tt + atol)