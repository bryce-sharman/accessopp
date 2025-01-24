from datetime import datetime, timedelta
from geopandas import GeoSeries
from os import PathLike
import pandas as pd
from pathlib import Path
from typing import List, Optional

import r5py

from accessopp.enumerations import DEFAULT_SPEED_WALKING, DEFAULT_SPEED_CYCLING
from accessopp.enumerations import DEFAULT_DEPARTURE_WINDOW
from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN

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
        ) -> pd.DataFrame:
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
        origins, destinations = self._validate_origins_destinations(
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
        ) -> pd.DataFrame:
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
        origins, destinations = self._validate_origins_destinations(
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
        origins, destinations = self._validate_origins_destinations(
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


    def _validate_origins_destinations(self, origins, destinations):
        if not isinstance(origins, GeoSeries):
            raise RuntimeError("origins are not a geopandas.GeoSeries")
        if destinations is None:
            destinations = origins.copy()
        elif not isinstance(destinations, GeoSeries):
                raise RuntimeError("destinations are not a geopandas.GeoSeries")
        return origins, destinations
    
    @staticmethod
    def _convert_tt_matrix(tt):
        tt = tt.set_index(['from_id', 'to_id']).squeeze()
        tt.index.names = INDEX_COLUMNS
        tt.name = COST_COLUMN
        return tt * 60.0   # convert from minutes to seconds
    