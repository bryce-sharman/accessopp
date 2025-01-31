from datetime import datetime, timedelta
from geopandas import GeoSeries
import numpy as np
import pandas as pd
from pathlib import Path
from pandas import testing as tm
import pytest
from shapely import Point

# from accessopp.matrix import Matrix
from accessopp.travel_time_computer.valhalla import ValhallaTravelTimeComputer
from accessopp.enumerations import DEFAULT_SPEED_WALKING, DEFAULT_SPEED_CYCLING
from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN

RTOL_COARSE = 0.01    # 1%
ATOL_COARSE = 30      # seconds


@pytest.fixture
def vttc():
    custom_files_path = Path(r'C:\Users\bsharma2\Valhalla\custom_files')

    queensway_dir = Path(r"C:\Users\bsharma2\Valhalla\testing\queensway")
    osm_path = queensway_dir / "queensway_test.osm.pbf"
    gtfs_path= queensway_dir / "GTFS_rt80.zip"
    server_threads=1
    vttc = ValhallaTravelTimeComputer(custom_files_path=custom_files_path)
    vttc.build_network(osm_path, gtfs_path, force_rebuild=True, test_mode=True)
    yield vttc


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
def gs_pt_1():
    yield  GeoSeries(
        index=[31],
        data=[
            Point(-79.52664, 43.62091)
        ],
        crs="EPSG:4326"
    )

@pytest.fixture
def gs_pt_2():
    yield  GeoSeries(
        index=[41],
        data=[
            Point(-79.49030, 43.62901) 
        ],
        crs="EPSG:4326"
    )


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


#region walk matrix tests
def test_walk_matrix_default_speed_walking(vttc, origins, destinations):
    """ Test walk travel time matrix with default walk speed. """
    ttm = vttc.compute_walk_traveltime_matrix(origins, destinations)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[2046, 2159, 1806, 1925, 1357, 1470]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)


def test_walk_matrix_slow_speed_walking(vttc, origins, destinations):
    """ Test walk travel time matrix with walk speed of 2 km/hr. """
    ttm = vttc.compute_walk_traveltime_matrix(
        origins, destinations, speed_walking=2.0)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[5095, 5382, 4493, 4779, 3370, 3657]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)


def test_walk_matrix_use_hills_0(vttc, gs_pt_1, gs_pt_2):
    """ Test walk travel time matrix with default walk speed. 
    
    Not seeing a distance, maybe we can change the locations later.
    """
    ttm = vttc.compute_walk_traveltime_matrix(
        gs_pt_1, gs_pt_2, use_hills=0)
    mi = pd.MultiIndex.from_tuples(
        [(31, 41)], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[2209]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)


def test_walk_matrix_same_ods(vttc, origins):
    """ 
    Test walk matrix where destinations are not defined, 
    should produce a square matrix.
    """
    ttm = vttc.compute_walk_traveltime_matrix(origins)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[11, 12, 13]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[0, 828, 925, 828, 0, 708, 925, 708, 0]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

#region transit matrix tests


def test_single_transit_trip_9_00(
        vttc, pt_1, pt_2, departuretime_9_00):
    """ Single transit trip with long wait for the bus. """
    tt = vttc._compute_transit_trip_traveltime(
        pt_1, pt_2, departuretime_9_00)
    _test_single_tt(tt, 1870, ATOL_COARSE, ATOL_COARSE)

def test_single_transit_trip_9_15g(
        vttc, pt_1, pt_2, departuretime_9_15):
    """ Single transit trip with long wait for the bus. """
    tt = vttc._compute_transit_trip_traveltime(
        pt_1, pt_2, departuretime_9_15)
    _test_single_tt(tt, 970, ATOL_COARSE, ATOL_COARSE)

def test_single_transit_trip_same_orig_dest(vttc, pt_1, departuretime_9_15):
    """ Single transit trip to/from same point. """
    tt = vttc._compute_transit_trip_traveltime(
        pt_1, pt_1, departuretime_9_15, speed_walking=DEFAULT_SPEED_WALKING)
    assert tt == 0.0

def test_single_transit_trip_speed_walking_2_0(
        vttc, pt_1, pt_2, departuretime_9_15):
    """ Single transit trip with long wait for the bus, slow walk speed. """
    tt = vttc._compute_transit_trip_traveltime(
        pt_1, pt_2, departuretime_9_15, speed_walking=2.0)
    _test_single_tt(tt, 1080, ATOL_COARSE, ATOL_COARSE)

def test_walk_matrix_pt1_pt2(vttc, gs_pt_1, gs_pt_2):
    tt = vttc.compute_walk_traveltime_matrix(gs_pt_1, gs_pt_2)
    mi = pd.MultiIndex.from_tuples(
        [(31, 41)], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[2209]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(tt, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_transit_matrix_pt1_pt2_0900(
        vttc, gs_pt_1, gs_pt_2, departuretime_9_00):
    """ Test transit travel time matrix. """
    tt = vttc.compute_transit_traveltime_matrix(
        gs_pt_1, gs_pt_2, departuretime_9_00)
    mi = pd.MultiIndex.from_tuples(
        [(31, 41)], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[1420]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(tt, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_transit_matrix_pt1_pt2_0915(
        vttc, gs_pt_1, gs_pt_2, departuretime_9_15):
    """ Test transit travel time matrix. """
    tt = vttc.compute_transit_traveltime_matrix(
        gs_pt_1, 
        gs_pt_2, 
        departuretime_9_15
    )
    mi = pd.MultiIndex.from_tuples(
        [(31, 41)], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[1509]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(tt, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_transit_short_interval(
        vttc, gs_pt_1, gs_pt_2, departuretime_9_00):
    """ I tested this manually, but calling the individual trips
    and taking the median would be a good improvement on this test. """
    tt = vttc.compute_transit_traveltime_matrix(
        gs_pt_1, 
        gs_pt_2, 
        departuretime_9_00,
        departure_time_window=timedelta(minutes=7),
        time_increment=timedelta(minutes=2)
    )
    mi = pd.MultiIndex.from_tuples(
        [(31, 41)], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi, 
        dtype=np.float64,
        data=[1690]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(tt, ref_matrix, RTOL_COARSE, ATOL_COARSE)
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
