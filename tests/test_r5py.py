from datetime import datetime, timedelta
from geopandas import GeoSeries
import numpy as np
import pandas as pd
import pandas.testing as tm
from pathlib import Path
from shapely import Point
import pytest

from accessopp.travel_time_computer.r5py import R5PYTravelTimeComputer  
from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN

RTOL_COARSE = 0.01    # 1%
ATOL_COARSE = 30      # seconds


@pytest.fixture
def r5i():
    r5i = R5PYTravelTimeComputer()
    root_dir = Path(r'C:\accessopp_examples')
    osm_path = root_dir / "network" / "osm" / "queensway_test.osm.pbf"
    gtfs_path = root_dir / "network" / "gtfs" / "GTFS_rt80.zip"
    r5i.build_network(
        osm_pbf=osm_path,
        gtfs=gtfs_path
    )
    yield r5i

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


def test_walk_matrix_default_speed_walking(r5i, origins, destinations):
    """ Test walk travel time matrix. """
    ttm = r5i.compute_walk_traveltime_matrix(origins, destinations)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[2040, 2160, 1680, 1860, 1320, 1500]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)


def test_walk_matrix_speed_walking_2_0(r5i, origins, destinations):
    """ Test walk travel time matrix with slow walking speed. """
    ttm = r5i.compute_walk_traveltime_matrix(
          origins, destinations, speed_walking=2.0)
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[5100, 5340, 4260, 4560, 3360, 3600]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)

def test_bike_matrix_speed_cycling_16(r5i, origins, destinations):
    """ Test bike matrix"""
    ttm = r5i.compute_bike_traveltime_matrix(
        origins=origins,
        destinations=destinations,
        speed_cycling = 16.0, # kilometers per hour
        max_bicycle_traffic_stress=3
    )
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[960, 1020, 540, 660, 600, 660]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)



def test_transit_matrix_speed_walking_2_0(
        r5i, origins, destinations, departuretime_9_00):
    """ 
    Test walk travel time matrix with slower walk speed and small window. """
    ttm = r5i.compute_transit_traveltime_matrix(
        origins, 
        destinations, 
        departuretime_9_00, 
        departure_time_window=timedelta(minutes=10), 
        speed_walking=2.0
    )
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[2040, 2160, 2220, 2340, 1980, 2100]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)    


def test_transit_matrix_speed_walking_2_0_later_departure(
        r5i, origins, destinations, departuretime_9_15):
    """ 
    Test walk travel time matrix with slower walk speed, 
    and a different departure time with small window. 
    """
    ttm = r5i.compute_transit_traveltime_matrix(
        origins, 
        destinations, 
        departuretime_9_15, 
        departure_time_window=timedelta(minutes=10), 
        speed_walking=2.0
    )
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[2280, 2400, 2280, 2400, 2280, 2400]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)   


def test_transit_matrix_speed_walking_2_0_unconnected_pt(
        r5i, origins, destinations_unconnected_pt, departuretime_9_00):
    """ 
    Test walk travel time matrix with slower walk speed and small window. """
    ttm = r5i.compute_transit_traveltime_matrix(
        origins, 
        destinations_unconnected_pt, 
        departuretime_9_00, 
        departure_time_window=timedelta(minutes=10), 
        speed_walking=2.0
    )
    mi = pd.MultiIndex.from_product(
        [[11, 12, 13],[21, 22]], names=INDEX_COLUMNS)
    ref_matrix = pd.Series(
        index=mi,         
        dtype=np.float64,
        data=[2040, np.NaN, 2220, np.NaN, 1980, np.NaN]
    )
    ref_matrix.name = COST_COLUMN
    _test_matrix(ttm, ref_matrix, RTOL_COARSE, ATOL_COARSE)  

def _test_matrix(matrix, ref_matrix, rtol=1.0e-5, atol=1.0e-8):
    tm.assert_series_equal(matrix, ref_matrix, rtol=rtol, atol=atol)
