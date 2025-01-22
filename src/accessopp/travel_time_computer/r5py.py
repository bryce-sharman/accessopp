import datetime
from pathlib import Path
import r5py

from ..enumerations import DEFAULT_SPEED_WALKING, DEFAULT_DEPARTURE_WINDOW
from ..utilities import test_od_input

class R5PYTravelTimeComputer():
        
    def __init__(self):
        self._transport_network = None

    def build_network(self, osm_pbf, gtfs):
        """ Build a transport network from specified OSM and GTFS files, saving to self.transport_network.

        Arguments
        ---------
        osm_pbf : str | pathlib.Path
            file path of an OpenStreetMap extract in PBF format
        gtfs : str | pathlib.Path | list[str] | list[pathlib.Path]
            path(s) to public transport schedule information in GTFS format

        Returns
        -------
        None

        """
        self._transport_network = r5py.TransportNetwork(osm_pbf, gtfs)

    def build_network_from_dir(self, path):
        """ Builds a transport network given a directory containing OSM and GTFS files, saving to self._transport_network.

            Arguments
            ---------
            path : str
                directory path in which to search for GTFS and .osm.pbf files

        """
        self._transport_network = r5py.TransportNetwork.from_directory(path)

    def compute_walk_traveltime_matrix(self, origins, destinations=None, speed_walking=DEFAULT_SPEED_WALKING, **kwargs):
        """ Requests walk-only trip travel time matrix from r5py.

            Parameters
            ----------
            origins: geopandas.GeoDataFrame
                Origin points.  Has to have at least an ``id`` column and a geometry
            destinations: geopandas.GeoDataFrame or None, optional
                Destination points. If None, will use the origin points. Default is None
                If not None, has to have at least an ``id`` column and a geometry.
            speed_walking: float or None, optional
                Walking speed in kilometres per hour.
                If None, set this is set to the default walk speed; currently 5 km/hr.
            **kwargs:
                Additional parameters to be passed into r5py.TravelTimeMatrixComputer
                https://r5py.readthedocs.io/en/stable/reference/reference.html#r5py.TravelTimeMatrixComputer.compute_travel_times

            Returns
            -------
            pd.DataFrame
                Table with travel costs in tall format. Columns are 'from_id', 'to_id', 'travel_time', where travel_time 
                is the median calculated travel time between from_id and to_id or numpy.nan if no connection with the 
                given parameters was found.
        """

        test_od_input(origins)
        if destinations is not None:
           test_od_input(destinations)        
        ttm = r5py.TravelTimeMatrixComputer(
           self._transport_network,
           origins=origins,
           destinations=destinations,
           transport_modes=[r5py.TransportMode.WALK],
           speed_walking=speed_walking,
           **kwargs
        )
        return ttm.compute_travel_times()


    def compute_transit_traveltime_matrix(self, origins, destinations=None, departure=datetime.datetime.now(), 
                                          departure_time_window=DEFAULT_DEPARTURE_WINDOW, speed_walking=DEFAULT_SPEED_WALKING, **kwargs):
        """ Requests transit-only trip matrix from OTP, returing either duration, trip distance or OTP's generalized cost.

            Parameters
            ----------
            origins: geopandas.GeoDataFrame
                Origin points.  Has to have at least an ``id`` column and a geometry
            destinations: geopandas.GeoDataFrame or None, optional
                Destination points. If None, r5py will use the origin points also as destinations. Default is None
                If not None, has to have at least an ``id`` column and a geometry.
            departure : datetime.datetime
                r5py will find public transport connections leaving every minute within
                ``departure_time_window`` after ``departure``. Default: current date and time
            departure_time_window : datetime.timedelta
                (see ``departure``) Default: 60 minutes
            speed_walking: float or None, optional
                Walking speed in kilometres per hour.
                If None, set this is set to the default walk speed; currently 5 km/hr.
            **kwargs:
                Additional parameters to be passed into r5py.TravelTimeMatrixComputer
                https://r5py.readthedocs.io/en/stable/reference/reference.html#r5py.TravelTimeMatrixComputer.compute_travel_times

            Returns
            -------
            pd.DataFrame
                Table with travel costs in tall format. Columns are 'from_id', 'to_id', 'travel_time', where travel_time 
                is the median calculated travel time between from_id and to_id or numpy.nan if no connection with the 
                given parameters was found. Note that from_id and to_id are both strings, as r5py performs this conversion.
        """

        test_od_input(origins)
        if destinations is not None:
           test_od_input(destinations)

        ttm = r5py.TravelTimeMatrixComputer(
           self._transport_network,
           origins=origins,
           destinations=destinations,
           departure=departure,
           departure_time_window=departure_time_window,
           transport_modes=[r5py.TransportMode.WALK, r5py.TransportMode.TRANSIT],
           speed_walking=speed_walking,
           **kwargs
        )
        return ttm.compute_travel_times()


