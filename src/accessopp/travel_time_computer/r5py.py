from datetime import datetime, timedelta
from geopandas import GeoSeries, points_from_xy
from os import PathLike
import pandas as pd
from pathlib import Path
from typing import List, Optional
import zipfile

import r5py

from accessopp.enumerations import DEFAULT_SPEED_WALKING, DEFAULT_SPEED_CYCLING
from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN
from accessopp.utilities import validate_origins_destinations

class R5PYTravelTimeComputer():
        
    def __init__(self):
        self._transport_network = None

    def build_network(
            self, 
            osm_pbf: PathLike, 
            gtfs: PathLike | List[PathLike]
        ) -> None:
        """ 
        Build a transport network from specified OSM and GTFS files, saving 
        to self.transport_network.

        Args:
            osm_pbf : file path of an OpenStreetMap extract in PBF format
            gtfs : path(s) to GTFS public transport schedule(s)

        """
        self._transport_network = r5py.TransportNetwork(osm_pbf, gtfs)


    def build_network_from_dir(self, path: PathLike):
        """ 
        Builds a transport network given a directory containing OSM and GTFS 
        files, saving to self._transport_network.

        Args:
            path : directory path in which to search for GTFS and .osm.pbf files

        """
        self._transport_network = r5py.TransportNetwork.from_directory(path)

    def compute_walk_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            speed_walking: float = DEFAULT_SPEED_WALKING,
            **kwargs
        ) -> pd.Series:
        """
        Requests walk-only trip matrix from r5py, returing trip duration 
        in seconds.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            speed_walking: Walking speed in kilometres per hour. If None then
                this is set to the default walk speed; currently 5 km/hr.
            **kwargs:
                Additional parameters to be passed into r5py.TravelTimeMatrixComputer
                https://r5py.readthedocs.io/en/stable/reference/reference.html#r5py.TravelTimeMatrixComputer.compute_travel_times
                Commonly used kwargs include:
                    max_time:  maximum trip duration 

        Returns:
            Travel times matrix in stacked (tall) format. The calculated 
                travel time is the median calculated travel time between 
                origin and destinations, if it exists, and numpy.nan if no 
                connection with the given parameters was found using
                search parameters.
        """
        origins, destinations = validate_origins_destinations(
            origins, destinations)  
        # r5py wants the columns to be in a column called 'id', so make this
        origins.index.name = 'id'
        destinations.index.name = 'id'
        ttm = r5py.TravelTimeMatrixComputer(
           self._transport_network,
           origins=origins.reset_index(),
           destinations=destinations.reset_index(),
           transport_modes=[r5py.TransportMode.WALK],
           speed_walking=speed_walking,
           **kwargs
        )
        df = ttm.compute_travel_times()
        return self._convert_tt_matrix(df)


    def compute_bike_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            speed_cycling: float = DEFAULT_SPEED_CYCLING,
            **kwargs
        ) -> pd.Series:
        """
        Requests walk-only trip matrix from r5py, returing trip duration 
        in seconds.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            speed_cycling: Cycling speed in kilometres per hour. If None then
                this is set to the default walk speed; currently 18 km/hr.
            **kwargs:
                Additional parameters to be passed into r5py.TravelTimeMatrixComputer
                https://r5py.readthedocs.io/en/stable/reference/reference.html#r5py.TravelTimeMatrixComputer.compute_travel_times
                Commonly used kwargs include:
                    - max_time:  maximum trip duration 
                    - max_bicycle_traffic_stress (int) Maximum stress level for 
                      cyclist routing, ranges from 1-4 see 
                      https://docs.conveyal.com/learn-more/traffic-stress 
                      Default: 3

        Returns:
            Travel times matrix in stacked (tall) format. The calculated 
                travel time is the median calculated travel time between 
                origin and destinations, if it exists, and numpy.nan if no 
                connection with the given parameters was found using
                search parameters.
        """
        origins, destinations = validate_origins_destinations(
            origins, destinations)  
        # r5py wants the columns to be in a column called 'id', so make this
        origins.index.name = 'id'
        destinations.index.name = 'id'
        ttm = r5py.TravelTimeMatrixComputer(
           self._transport_network,
           origins=origins.reset_index(),
           destinations=destinations.reset_index(),
           transport_modes=[r5py.TransportMode.BICYCLE],
           speed_cycling=speed_cycling,
           **kwargs
        )
        df = ttm.compute_travel_times()
        return self._convert_tt_matrix(df)


    def compute_transit_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            departure: datetime, 
            departure_time_window: timedelta, 
            speed_walking: float=DEFAULT_SPEED_WALKING,
            **kwargs
        ) ->pd.Series:
        """ 
        Requests walk/transit trip matrix from OTP, returing either trip duration in minutes.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            departure: Date and time of the start of the departure window
                r5py will find public transport connections leaving every minute 
                within `departure_time_window` after `departure`. 
            departure_time_window: Length of departure time window. All trips
                are averaged in this window to produce an average transit 
                travel time.
            speed_walking: Walking speed in kilometres per hour. If None, 
                this is set to the default walk speed; currently 5 km/hr.
            **kwargs:
                Additional parameters to be passed into r5py.TravelTimeMatrixComputer
                https://r5py.readthedocs.io/en/stable/reference/reference.html#r5py.TravelTimeMatrixComputer.compute_travel_times
                Commonly used kwargs include:
                    - max_time:  maximum trip duration 
                    - max_public_transport_rides (int) 
                        Use at most max_public_transport_rides consecutive 
                        public transport connections. Default: 8
        Returns:
            Travel times, in seconds, in stacked (tall) format.

        """
        origins, destinations = validate_origins_destinations(
            origins, destinations)  

        origins.index.name = 'id'
        destinations.index.name = 'id'
        ttm = r5py.TravelTimeMatrixComputer(
           self._transport_network,
           origins=origins.reset_index(),
           destinations=destinations.reset_index(),
           departure=departure,
           departure_time_window=departure_time_window,
           transport_modes=[r5py.TransportMode.WALK, r5py.TransportMode.TRANSIT],
           speed_walking=speed_walking,
           **kwargs
        )
        df = ttm.compute_travel_times()
        return self._convert_tt_matrix(df)

    
    def compute_walk_traveltime_matrix_to_transit_stops(
            self, 
            origins: GeoSeries, 
            gtfs_path: PathLike, 
            speed_walking: float = DEFAULT_SPEED_WALKING,
            **kwargs
        ) -> pd.Series:
        """
        Requests walk-only trip matrix from r5py between origins and all
        stops defined inthe GTFS stops file.

        Args:
            - origins: Origin points.  Index is the point ids.
            - gtfs_path: Path to the GTFS file containing transit schedule.
            = speed_walking: Walking speed in kilometres per hour. If None then
                this is set to the default walk speed; currently 5 km/hr.
            - **kwargs:
                 Additional parameters to be passed into r5py.TravelTimeMatrixComputer
                 https://r5py.readthedocs.io/en/stable/reference/reference.html#r5py.TravelTimeMatrixComputer.compute_travel_times
                 Commonly used kwargs include:
                     max_time:  maximum trip duration 

        Returns:
            Travel times matrix in stacked (tall) format between origins
                and all stops defined in the GTFS stops file. 

        """
        with zipfile.ZipFile(gtfs_path) as zf:
            stops = zf.open("stops.txt")
            df = pd.read_csv(stops, usecols=['stop_id', 'stop_lat', 'stop_lon'])
            destinations = GeoSeries(
                index=df['stop_id'],
                data = points_from_xy(
                    df['stop_lon'], df['stop_lat'], crs="EPSG:4326")
            )
        return self.compute_walk_traveltime_matrix(
            origins, destinations, speed_walking=speed_walking, **kwargs)


    @staticmethod
    def _convert_tt_matrix(tt):
        tt = tt.set_index(['from_id', 'to_id']).squeeze()
        tt.index.names = INDEX_COLUMNS
        tt.name = COST_COLUMN
        return tt * 60.0   # convert from minutes to seconds
    