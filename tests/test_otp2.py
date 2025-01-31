from datetime import datetime, timedelta
from geopandas import GeoSeries
import numpy as np
import pandas as pd
from pandas import testing as tm
import pytest
from shapely import Point

# from accessopp.matrix import Matrix
from accessopp.travel_time_computer.otp2 import OTP2TravelTimeComputer
from accessopp.enumerations import DEFAULT_SPEED_WALKING, DEFAULT_SPEED_CYCLING
from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN

RTOL_COARSE = 0.01    # 1%
ATOL_COARSE = 30      # seconds


@pytest.fixture
def otpi():
    otpi = OTP2TravelTimeComputer()
    otpi.java_path = r"C:\Program Files\Microsoft\jdk-17.0.9.8-hotspot\bin\java"
    otpi.otp_jar_path = r"C:\MyPrograms\otp\2_4\otp-2.4.0-shaded.jar"
    otpi.memory_str = "1G"
    otpi.build_network_from_dir(
        r"C:\Users\bsharma2\AccessOpportunities\Test Data\tests", overwrite=True, launch_server=True)
    yield otpi

@pytest.fixture
def origins():
    yield GeoSeries(
        index=[11, 12, 13],
        data=[
            Point(-79.52809, 43.61918), 
            Point(-79.52353, 43.62561), 
            Point(-79.51702, 43.62160)
        ],
        crs="EPSG:4326"
    )

@pytest.fixture
def destinations():
    yield  GeoSeries(
        index=[21, 22],
        data=[
            Point(-79.5012, 43.62924), 
            Point(-79.49908, 43.62977)
        ],
        crs="EPSG:4326"
    )

@pytest.fixture
def pt_1():   # Queensway & Kipling
    yield Point(-79.52664, 43.62091)

@pytest.fixture
def pt_2():    # Queensway at Park Lawn
    yield Point(-79.49030, 43.62901)

@pytest.fixture
def pt_unconnected():
    yield Point(-77.52809, 43.61918)

@pytest.fixture
def destinations_unconnected_pt():
        yield GeoSeries(
            index=[21, 23],
            data= [
                Point(-79.5012, 43.62924), 
                Point(-77.52809, 43.61918)
            ],
            crs="EPSG:4326"
        )

@pytest.fixture
def departuretime_9_15():
    yield datetime(2024,2,9,9,15,0)

@pytest.fixture
def departuretime_9_00():
    yield datetime(2024,2,9,9,0,0)

#region Departure time testing
def test_departure_time_testing_okay(otpi):
    otpi._test_departure_within_service_time_range(
        datetime(2024, 2, 9, 9, 0, 0))
         
def test_departure_time_testing_too_early(otpi):
    with pytest.raises(ValueError):
        otpi._test_departure_within_service_time_range(
            datetime(2023, 1, 15, 9, 0, 0))

def test_departure_time_testing_too_late(otpi):
    with pytest.raises(ValueError):
        otpi._test_departure_within_service_time_range(
            datetime(2025, 1, 15, 9, 0, 0))
#endregion

#region single trip tests
def test_walk_trip_default_speed_walking(otpi, pt_1, pt_2):
    """ Compare walk trip request using default walk speed of 5 km/hr."""
    tt = otpi._compute_walk_trip_traveltime(pt_1, pt_2, DEFAULT_SPEED_WALKING)
    _test_single_tt(tt, 2451, ATOL_COARSE, ATOL_COARSE)
      
def test_walk_trip_speed_walking_2_0(otpi, pt_1, pt_2):
    """ Compare walk trip request using walk speed of 2 km/hr."""
    tt = otpi._compute_walk_trip_traveltime(pt_1, pt_2, speed_walking=2.0)
    _test_single_tt(tt, 5789, ATOL_COARSE, ATOL_COARSE)

def test_walk_trip_same_orig_dest(otpi, pt_1):
    """ Test that walk travel time to/from same point is 0. """
    tt = otpi._compute_walk_trip_traveltime(pt_1, pt_1, DEFAULT_SPEED_WALKING)
    assert tt == 0.0

def test_walk_trip_unconnected(otpi, pt_1, pt_unconnected):
    """ Test that walk travel time to unconnected point is NaN. """
    tt = otpi._compute_walk_trip_traveltime(
        pt_1, pt_unconnected, DEFAULT_SPEED_WALKING)
    assert np.isnan(tt)

def test_bike_trip_default_speed_cycling(otpi, pt_1, pt_2):
    """ Compare bike trip request using default bike speed of 18 km/hr."""
    tt = otpi._compute_bike_trip_traveltime(pt_1, pt_2, DEFAULT_SPEED_CYCLING)
    _test_single_tt(tt, 1068, ATOL_COARSE, ATOL_COARSE)
 
def test_bike_trip_speed_cycling_12(otpi, pt_1, pt_2):
    """ Compare bike trip request using cycling speed of 12 km/hr."""
    tt = otpi._compute_bike_trip_traveltime(pt_1, pt_2, speed_biking=12.0)
    _test_single_tt(tt, 1523, ATOL_COARSE, ATOL_COARSE)

def test_bike_trip_same_orig_dest(otpi, pt_1):
    """ Test that bike travel time to/from same point is 0. """
    tt = otpi._compute_bike_trip_traveltime(pt_1, pt_1, DEFAULT_SPEED_CYCLING)
    _test_single_tt(tt, 0, ATOL_COARSE, ATOL_COARSE)

def test_bike_trip_unconnected(otpi, pt_1, pt_unconnected):
    """ Test that bike travel time to unconnected point is NaN. """
    tt = otpi._compute_bike_trip_traveltime(
        pt_1, pt_unconnected, DEFAULT_SPEED_CYCLING)
    assert np.isnan(tt)

def test_single_transit_trip_default_speed_walking(
        otpi, pt_1, pt_2, departuretime_9_15):
    """ Single transit trip with long wait for the bus. """
    tt = otpi._compute_transit_trip_traveltime(
        pt_1, pt_2, departuretime_9_15, speed_walking=DEFAULT_SPEED_WALKING)
    _test_single_tt(tt, 1015, ATOL_COARSE, ATOL_COARSE)

def test_single_transit_trip_speed_walking_2_0(
        otpi, pt_1, pt_2, departuretime_9_15):
    """ Single transit trip with long wait for the bus, slow walk speed. """
    tt = otpi._compute_transit_trip_traveltime(
        pt_1, pt_2, departuretime_9_15, speed_walking=2.0)
    _test_single_tt(tt, 1121, ATOL_COARSE, ATOL_COARSE)

def test_single_transit_trip_same_orig_dest(otpi, pt_1, departuretime_9_15):
    """ Single transit trip to/from same point. """
    tt = otpi._compute_transit_trip_traveltime(
        pt_1, pt_1, departuretime_9_15, speed_walking=DEFAULT_SPEED_WALKING)
    assert tt == 0.0

def test_single_transit_trip_unconnected(
        otpi, pt_1, pt_unconnected, departuretime_9_15):
    """ Single transit trip to unconnected point"""
    tt = otpi._compute_transit_trip_traveltime(
        pt_1, pt_unconnected, departuretime_9_15, 
        speed_walking=DEFAULT_SPEED_WALKING
    )
    assert np.isnan(tt)
#endregion

#region Transit trips over time interval
def test_transit_trip_interval_default_speed_walking(
        otpi, pt_1, pt_2, departuretime_9_00):
    """ Transit time interval over 1 hour with default walk speed.  """
    tt = otpi._compute_interval_transit_traveltime(
        pt_1, 
        pt_2, 
        departure=departuretime_9_00, 
        departure_time_window=timedelta(hours=1),
        time_increment = timedelta(minutes=1),
        speed_walking=DEFAULT_SPEED_WALKING
    )
    _test_single_tt(tt, 1555, ATOL_COARSE, ATOL_COARSE)

def test_transit_trip_interval_default_speed_walking(
        otpi, pt_1, pt_2, departuretime_9_00):
    """ Test transit time interval over 1 hour with slow walk speed. """
    tt = otpi._compute_interval_transit_traveltime(
        pt_1, 
        pt_2, 
        departure=departuretime_9_00, 
        departure_time_window=timedelta(hours=1),
        time_increment = timedelta(minutes=1),
        speed_walking=DEFAULT_SPEED_WALKING
    )
    _test_single_tt(tt, 1465, ATOL_COARSE, ATOL_COARSE)

def test_transit_interval_trip_same_orig_dest(
        otpi, pt_1, departuretime_9_00):
    """ Test transit time interval over 1 hour, same O-D """
    tt = otpi._compute_interval_transit_traveltime(
        pt_1, 
        pt_1, 
        departure=departuretime_9_00, 
        departure_time_window=timedelta(hours=1),
        time_increment = timedelta(minutes=1),
        speed_walking=DEFAULT_SPEED_WALKING
    )
    assert tt == 0.0

def test_transit_interval_unconnected_points(
        otpi, pt_1, pt_unconnected, departuretime_9_00):
    """ Test transit time interval over 1 hour, unconnected points """
    tt = otpi._compute_interval_transit_traveltime(
        pt_1, 
        pt_unconnected, 
        departure=departuretime_9_00, 
        departure_time_window=timedelta(hours=1),
        time_increment = timedelta(minutes=1),
        speed_walking=DEFAULT_SPEED_WALKING
    )
    assert np.isnan(tt)

def test_transit_trip_interval_short_timewindow(
        otpi, pt_1, pt_2, departuretime_9_00):
    """ Test transit time interval over 5 minutes, unconnected points. 
    
        Given the schedule of the test service, this should be 
        much below the average over the whole hour interval.
    """
    tt = otpi._compute_interval_transit_traveltime(
        pt_1, 
        pt_2, 
        departure=departuretime_9_00, 
        departure_time_window=timedelta(minutes=5),
        time_increment = timedelta(minutes=1),
        speed_walking=DEFAULT_SPEED_WALKING
    )
    _test_single_tt(tt, 1795, ATOL_COARSE, ATOL_COARSE)

#endregion            

#region matrix tests
def test_walk_matrix_default_speed_walking(otpi, origins, destinations):
    """ Test walk travel time matrix with default walk speed. """
    ttm = otpi.compute_walk_traveltime_matrix(origins, destinations)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[2230, 2351, 1809, 1907, 1479, 1600]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_walk_matrix_slow_speed_walking(otpi, origins, destinations):
    """ Test walk travel time matrix with walk speed of 2 km/hr. """
    ttm = otpi.compute_walk_traveltime_matrix(
        origins, destinations, speed_walking=2.0)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[5311, 5605, 4393, 4635, 3519, 3802]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_walk_matrix_same_ods_1(otpi, origins):
    """ 
    Test walk matrix where destinations are not defined, 
    should produce a square matrix.
    """
    ttm = otpi.compute_walk_traveltime_matrix(origins)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[11, 12, 13]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[0, 902, 1006, 890, 0, 760, 1006, 760, 0]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_walk_matrix_same_ods_2(otpi, origins):
    """ 
    Test walk matrix where destinations are not defined, 
    should produce a square matrix. Slow walking speed
    """
    ttm = otpi.compute_walk_traveltime_matrix(origins, speed_walking=2.0)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[11, 12, 13]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[0, 2167, 2380, 2135, 0, 1827, 2379, 1826, 0]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)


def test_walk_matrix_with_unconnected_pt(
        otpi, origins, destinations_unconnected_pt):
    """
    Test walk matrix when there's an unconnected point.
    """
    ttm = otpi.compute_walk_traveltime_matrix(
        origins, destinations_unconnected_pt)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 23]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[2230, np.NaN, 1809, np.NaN, 1479, np.NaN]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)


def test_bike_matrix_default_speed_cycling(otpi, origins, destinations):
    """ Test bike travel time matrix with default cycling speed. """
    ttm = otpi.compute_bike_traveltime_matrix(origins, destinations)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[940, 964, 614, 643, 648, 686]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_bike_matrix_slow_speed_cycling(otpi, origins, destinations):
    """ Test bike travel time matrix with slower cycling speed. """
    ttm = otpi.compute_bike_traveltime_matrix(
        origins, destinations, speed_cycling=12.0)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[1331, 1372, 911, 954, 878, 957]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_transit_matrix_default_speed_walking(
        otpi, origins, destinations, departuretime_9_00):
    """ Test transit travel time matrix. """
    ttm = otpi.compute_transit_traveltime_matrix(
        origins, destinations, departuretime_9_00)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[1638, 1673, 1638, 1673, 1428, 1463]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_transit_matrix_slow_walking_short_interval(
        otpi, origins, destinations, departuretime_9_00):
    """ Test transit travel time matrix. """
    ttm = otpi.compute_transit_traveltime_matrix(
        origins, destinations, departuretime_9_00,
        departure_time_window=timedelta(minutes=5),
        speed_walking=2.0
    )
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[2143, 2230, 2143, 2230, 2143, 2230]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

#endregion

def _test_matrix(matrix, ref_matrix, rtol=1.0e-5, atol=1.0e-8):
    """ 
    Test matrix helping function that checks two matrices. 
    Raises Assertion error if test fails.
    """
    tm.assert_series_equal(matrix, ref_matrix, rtol=rtol, atol=atol)

def _test_single_tt(tt, ref_tt, rtol=1.0e-5, atol=1.0e-8):
    assert ref_tt / (1.0 + rtol) <= tt <= ref_tt * (1.0 + rtol)
    assert ref_tt - atol <= tt <= ref_tt + atol
