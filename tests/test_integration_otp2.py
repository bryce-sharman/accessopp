from datetime import datetime, timedelta
from importlib.resources import files
import pandas as pd
import pandas.testing as tm
import unittest

import accessopp
from accessopp.travel_time_computer.otp2 import OTP2TravelTimeComputer
from accessopp.access_opportunities import calc_spatial_access, closest_opportunity, within_threshold
from accessopp.points import read_points_from_csv


class TestPointsIO(unittest.TestCase):

    def setUp(self):
        # Find the path to to the test-directions
        # The src path is to the src/accessopp directory. 
        src_path = files(accessopp)
        root_path = src_path.parents[1]
        testdata_path = root_path / "tests" / "test_data" 
        self.points_path = testdata_path / "points"
        self.network_path = testdata_path / "network"

    def test_integration_test_1(self):
        """ Integration test 1: read origins and destinations; calc travel times; find closest opportunity. """
        
        # Read in points data
        origins = read_points_from_csv(self.points_path / "small_origins.csv", 'id', 'longitude', 'latitude', 'orig_wt')
        destinations = read_points_from_csv(self.points_path / "small_destinations.csv", 'id', 'longitude', 'latitude', 'dest_wt')

        departure = datetime(2024,1,15,9,0,0)
        departure_time_window = timedelta(minutes=180)
        dep_time_interval = timedelta(minutes=5)

        # Calculate travel times
        otp_comp = OTP2TravelTimeComputer()
        otp_comp.java_path = r"C:\Program Files\Microsoft\jdk-17.0.9.8-hotspot\bin\java"
        otp_comp.otp_jar_path = r"C:\MyPrograms\otp\2_4\otp-2.4.0-shaded.jar"
        otp_comp.memory_str = "1G"

        otp_comp.build_network_from_dir(self.network_path, overwrite=True)
        ttm = otp_comp.compute_transit_traveltime_matrix(
            origins, destinations, departure, departure_time_window, dep_time_interval)

        # Calculate closest opportunity dual access measure
        cl_opp = closest_opportunity(ttm)
        ref_cl_opp = pd.Series(
            index=[11, 12, 13],
            data=[28, 28, 24]
        )
        ref_cl_opp.index.name = "from_id"
        tm.assert_series_equal(cl_opp, ref_cl_opp, check_dtype=False)

    def test_integration_test_2(self):
        """ Integration test 1: read origins and destinations; calc travel times; calculate weighted access to destinations. """
        
        # Read in points data
        origins = read_points_from_csv(self.points_path / "small_origins.csv", 'id', 'longitude', 'latitude', 'orig_wt')
        destinations = read_points_from_csv(self.points_path / "small_destinations.csv", 'id', 'longitude', 'latitude', 'dest_wt')

        departure = datetime(2024,1,15,9,0,0)
        departure_time_window = timedelta(minutes=180)
        dep_time_interval = timedelta(minutes=5)

        # Calculate travel times
        otp_comp = OTP2TravelTimeComputer()
        otp_comp.java_path = r"C:\Program Files\Microsoft\jdk-17.0.9.8-hotspot\bin\java"
        otp_comp.otp_jar_path = r"C:\MyPrograms\otp\2_4\otp-2.4.0-shaded.jar"
        otp_comp.memory_str = "1G"

        otp_comp.build_network_from_dir(self.network_path, overwrite=True)
        ttm = otp_comp.compute_transit_traveltime_matrix(
            origins, destinations, departure, departure_time_window, dep_time_interval)

        # Calculate closest opportunity dual access measure
        destination_weights = destinations.set_index('id')['dest_wt']
        access = calc_spatial_access(ttm, within_threshold, threshold=28.5, o_j=destination_weights)
        ref_access = pd.Series(
            index=[11, 12, 13],
            data=[2500, 2500, 10000]
        )
        ref_access.index.name = "from_id"
        tm.assert_series_equal(access, ref_access, check_dtype=False)
