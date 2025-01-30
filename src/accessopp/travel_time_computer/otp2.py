from datetime import datetime, timedelta
from geopandas import GeoSeries
from  os import environ, PathLike
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from shapely import Point
from shutil import rmtree, copy2
from subprocess import Popen, run
from time import sleep
from typing import List, Optional

from accessopp.enumerations import DEFAULT_SPEED_WALKING, DEFAULT_SPEED_CYCLING
from accessopp.enumerations import DEFAULT_DEPARTURE_WINDOW, DEFAULT_TIME_INCREMENT
from accessopp.enumerations import N_DECIMALS
from accessopp.utilities import validate_origins_destinations, create_blank_ttmatrix

class OTP2TravelTimeComputer():
    """ Class to interface with OpenTripPlanner to calculate run times.

    This class is currently designed to operate with OpenTripPlanner version 2.4. The calls to OTP are conducted
    through their newer GTFS `*`GraphQL API`, as their previous `Restful API` will be discontinued.
    
    Parameters
    ----------
    None

    Methods
    -------
    Network and server methods:
    build_network: 
        Build a transport network from specified OSM and GTFS files.
    build_network_from_dir:
        Builds a transport network given a directory containing OSM and GTFS files.
    launch_local_otp2_server: 
        Launches OTP 2 server on the local machine using previously built network. 
    test_otp_server:
        Tests if can successfully connect to OTP 2 server. 

    Walk trip traveltime computations
    compute_walk_traveltime_matrix: 
        Requests walk-only trip matrix from OTP, returing either duration, trip distance or OTP's generalized cost.
    compute_walk_trip_traveltime:
        Requests a walk-only trip from OTP, returing walk time and distance.

    Transit trip traveltime computations
    compute_transit_traveltime_matrix: 
        Requests walk/transit trip matrix from OTP, returing either trip duration in minutes.
    compute_interval_transit_traveltime:
        Requests median travel time over interval, inclusive at interval start, exclusive at interval end.
    compute_transit_traveltime:
        Requests a transit/walk trip from OTP, returing total time and walk distance.
    test_departure_within_service_time_range;
        Test if provided date is within the graph service time range. 


    Attributes
    ----------
    java_path : pathlib.Path
        Path to the java executable used to run OTP. This must be set before any operations are done.
    otp_jar_path : pathlib.Path
        Path to OPT 2.4 jar file. This must be set before any operations are done.
    memory_str: str
        Code describing the memory to allocate. e.g. '2G' = 2 GB. Default is "1G"
    max_server_sleep_time: int or float
        Maximum number of seconds to wait after launching server. Default is 30 seconds.
    request_host_url: pathlib.Path
        Root URL used for requests to OTP server. Defaults to "http://localhost:8080", which is 
        a server running on a local machine.
    """

    # Class variables
    JAVA_PATH_ERROR = "Must set Java path before running Open Trip Planner."
    OTP_JAR_PATH_ERROR = \
        "Must define OTP 2.4 jar file path before running Open Trip Planner."

    HEADERS = {
        'Content-Type': 'application/json',
        'OTPTimeout': '180000',
    }

    def __init__(self):
        
        # These attributes will need to be defined before starting OTP instance
        self._java_path = None
        self._otp_jar_path = None
        self._memory_str = "1G"

        # These default URLs assume that the server is hosted locally. 
        # These can be changed by a user if desired
        self._request_host_url = "http://localhost:8080"
        self._request_api = \
            self._request_host_url + "/otp/routers/default/index/graphql"
        
        # The maximum allowed time for a server to fully load before the 
        # launch server method returns.
        self._max_server_sleep_time = 180

#region Graph and server
    def build_network(
            self, 
            osm_pbf: PathLike, 
            gtfs: PathLike | List[PathLike],
            launch_server: bool=True
        ):
        """ Build a transport network from specified OSM and GTFS files.

        Args:
            osm_pbf : file path of an OpenStreetMap extract in PBF format
            gtfs : path(s) to GTFS public transport schedule(s)
            launch_server: if True, launch the OTP server
                The server must be launched before computing travel
                time matrices.

        """
        # OTP needs to build from a directory, so this function creates 
        # a temp directory to which is copies all required files before using 
        # the build_graph_from_dir method
        osm_pbf = Path(osm_pbf).absolute()
        if isinstance(gtfs, (str, Path)):
            gtfs = [gtfs]
        gtfs = [Path(gtfs_file).absolute() for gtfs_file in gtfs]

        # Find the Windows temp directory, clear if it already exists
        temp_dir = Path(environ['TMP']) / 'OTP2'
        if temp_dir.exists():
            rmtree(temp_dir)
        temp_dir.mkdir()

        copy2(osm_pbf, temp_dir)
        for gtfs_file in gtfs:
            copy2(gtfs_file, temp_dir)

        # Create a network in this temporary directory, we can specify to overwrite as it is 
        # a temporary directiory.
        self.build_network_from_dir(temp_dir, True, launch_server)

    def build_network_from_dir(
            self, 
            path: PathLike, 
            overwrite: bool=False, 
            launch_server=True
        ) -> None:
        """ 
        Builds a transport network given a directory containing OSM and GTFS 
        files.

        Args:
            path : directory path in which to search for GTFS and .osm.pbf files
            overwrite: If True, overwrite any existing built OTP2 network.
                Will raise if set to False and there is an existing network.
            launch_server: If True, launch the OTP server. The server must be 
                launched before computing travel time matrices.

        """
        # Test to see if there is an existing network, delete if overwrite is True, otherwise exit.
        path = Path(path)
        network_file_path = path / "graph.obj"
        if network_file_path.is_file():
            if overwrite:
                network_file_path.unlink()
            else:
                raise FileExistsError("`overwrite` argument is False and network exists in the directory.")
        
        # Check that the JAVA and JAR paths have been set
        if self._java_path is None:
            raise FileNotFoundError(self.JAVA_PATH_ERROR)
        if self._otp_jar_path is None:
            raise FileNotFoundError(self.OTP_JAR_PATH_ERROR)
        
        full_memory_str = f"-Xmx{self._memory_str}"
        subproc_return = run([
            self._java_path.absolute(), full_memory_str, "-jar", self._otp_jar_path.absolute(), 
            "--build", "--save", path.absolute(),
        ], check=True)

        # Ensure that the graph has been created in the run directory
        if not network_file_path.is_file():
            raise FileExistsError("Network file was not created")

        if launch_server:
            self.launch_local_otp2_server(path)

    def launch_local_otp2_server(
            self, 
            path: PathLike
        ) -> None:
        """ Launches OTP 2 server on the local machine using previously built network. 
        
        Args
            path : Directory containg OTP graph file (network)

        Notes:
            This method will wait until a connection is made to the server  
            before finishing, up until a number of seconds defined in the class  
            attribute `max_server_sleep_time`. 
            
        """
        # Ensure there is an existing network
        path = Path(path)
        network_file_path = path / "graph.obj"
        if not network_file_path.is_file():
            raise FileExistsError(f"Need to build OTP network before launching server.")
        
        # Check that the JAVA and JAR paths have been set
        if self._java_path is None:
            raise FileNotFoundError(self.JAVA_PATH_ERROR)
        if self._otp_jar_path is None:
            raise FileNotFoundError(self.OTP_JAR_PATH_ERROR)

        full_memory_str = f"-Xmx{self._memory_str}"
        # Note that this server runs in the background, so we'll use a subprocess.popen constructor instead
        subproc_return = Popen([
            self._java_path.absolute(), full_memory_str, "-jar", self._otp_jar_path.absolute(), 
            "--load", path.absolute()
        ])

        # Sleep to ensure that the server is launched before exiting out of the method
        t = 0
        increment = 1
        while True:
            try:
                self.test_otp_server()
                break
            except requests.ConnectionError:
                # Not ready yet, try again
                pass
            except RuntimeError:
                # Not ready yet, try again
                pass
            sleep(increment)
            if t >= self._max_server_sleep_time:
                raise RuntimeError(
                    f"Connection to server could not be found within{self._max_server_sleep_time} seconds")
            t += increment

    def test_otp_server(self) -> None:
        """ Tests if can successfully connect to OTP 2 server. 
        
        Raises:
            requests.ConnectionError
                Raised if cannot connect 
            RuntimeError
                Raised if null values are returned from the request
        """

        host = requests.get(self._request_host_url)
        if host is None:
            raise RuntimeError("Null values returned from OTP request.")
#endregion
        
#region public walk travel-time matrix methods

    def compute_walk_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            speed_walking: Optional[float]=None
        ) -> pd.Series:
        """ 
        Requests walk-only trip matrix from OTP, returing trip duration 
        in seconds.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            speed_walking: Walking speed in kilometres per hour. If None then
                this is set to the default walk speed; currently 5 km/hr.

        Returns:
            Travel times matrix in stacked (tall) format.

        """
        origins, destinations = validate_origins_destinations(
            origins, destinations)        
        ttm = create_blank_ttmatrix(origins, destinations)
        for o_id, o_pt in origins.items():
            for d_id, d_pt in destinations.items():
                ttm.at[(o_id, d_id)] = self._compute_walk_trip_traveltime(
                    o_pt, d_pt, speed_walking, False)
        return ttm.round(N_DECIMALS)
    
    def compute_bike_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            speed_cycling: float = DEFAULT_SPEED_CYCLING, 
            triangle_time_factor: float = 0.5, 
            triangle_slope_factor: float = 0.5, 
            triangle_safety_factor: float= 0.5,
        ) -> pd.Series:
        """ 
        Requests bike-only trip matrix from OTP, returing trip duration 
        in seconds.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            speed_cycling: Cycling speed in kilometres per hour. If None then
                this is set to the default cycling speed; currently 18 km/hr.
            triangle_time_factor: Time(speed) optimization parameter
                for OTP2's 'triangle' optimization. (range 0-1), default is 0.5.
            triangle_slope_factor: Slope optimization parameter
                for OTP2's 'triangle' optimization. (range 0-1), default is 0.5.
            triangle_safety_factor: Safety optimization parameter
                for OTP2's 'triangle' optimization. (range 0-1), default is 0.5.

        Returns:
            Travel times matrix in stacked (tall) format.

        """
        origins, destinations = validate_origins_destinations(
            origins, destinations)        
        ttm = create_blank_ttmatrix(origins, destinations)
        for o_id, o_pt in origins.items():
            for d_id, d_pt in destinations.items():
                ttm.at[(o_id, d_id)] = self._compute_bike_trip_traveltime(
                    o_pt, d_pt, speed_cycling, triangle_time_factor, 
                    triangle_slope_factor, triangle_safety_factor, 
                    test_mode=False
                )
        return ttm.round(N_DECIMALS)

    def compute_transit_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            departure: datetime, 
            departure_time_window: timedelta = DEFAULT_DEPARTURE_WINDOW, 
            time_increment: timedelta = DEFAULT_TIME_INCREMENT, 
            speed_walking: float=DEFAULT_SPEED_WALKING
        ) ->pd.Series:
        """ 
        Requests walk/transit trip matrix from OTP, returing either trip duration in minutes.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            departure: Date and time of the start of the departure window
            departure_time_window: Length of departure time window. All trips
                are averaged in this window to produce an average transit 
                travel time.
            time_increment: Increment between different trip runs in minutes.
                If None, this is set to the default interval; currently 1 minute.
            speed_walking: Walking speed in kilometres per hour. If None, 
                this is set to the default walk speed; currently 5 km/hr.
                
        Returns:
            Travel times, in seconds, in stacked (tall) format.

        Notes:
            For transit, returning trip duration appears to be the only useful 
            measure returned from OTP2. The generalized cost output does not 
            appear to include waiting for the first bus.

        """
        self._test_departure_within_service_time_range(departure)
        origins, destinations = validate_origins_destinations(
            origins, destinations)
        ttm = create_blank_ttmatrix(origins, destinations)
        for o_id, o_pt in origins.items():
            for d_id, d_pt in destinations.items():
                tt = self._compute_interval_transit_traveltime(
                    o_pt, d_pt, departure, departure_time_window, 
                    time_increment, speed_walking, skip_test_trip_date=True)
                ttm.at[(o_id, d_id)] = tt
        return ttm.round(N_DECIMALS)
#endregion 


#region private helper functions


# CONTINUE TO SIMPLFY THIS CODE!!!

    def _compute_walk_trip_traveltime(
            self, origin, destination, speed_walking, test_mode=False):
        """ Requests walk-only trip from OTP returing walk time. """
        if origin == destination:
            return 0.0
        origin_str = self._set_pt_str("from", origin)
        destination_str = self._set_pt_str("to", destination)
        modes_str = "transportModes: [{mode: WALK}]"
        # Convert walk speed to metres per second
        walk_speed_str = f"walkSpeed: {speed_walking / 3.6}"  
        if test_mode:
            itineraries_str = \
                "{itineraries {startTime endTime walkDistance generalizedCost legs{mode duration distance}}}"
        else:
            itineraries_str = "{itineraries {startTime endTime}}"
        qry_str = "{plan(%s %s %s %s)%s}" % (
            origin_str, destination_str, modes_str, walk_speed_str, itineraries_str)
        request = requests.post(
            self._request_api, headers=self.HEADERS, json={'query': qry_str})
        response = request.json()
        return self._parse_json_response(
            response, False, None, test_mode=test_mode)

    def _compute_bike_trip_traveltime(
            self, 
            origin, 
            destination, 
            speed_biking, 
            triangle_time_factor=0.5, 
            triangle_slope_factor=0.5, 
            triangle_safety_factor=0.5,
            test_mode=False
        ):
        """ Requests bike-only trip from OTP returing bike time. """
        if origin == destination:
            return 0.0
        origin_str = self._set_pt_str("from", origin)
        destination_str = self._set_pt_str("to", destination)
        modes_str = "transportModes: [{mode: BICYCLE}]"
        # Convert bike speed to metres per second

        bike_speed_str = f"bikeSpeed: {speed_biking / 3.6}"  
        optimize_str = "optimize: TRIANGLE"
        triangle_str = \
            "triangle: {timeFactor: %s, slopeFactor: %s, safetyFactor: %s}" % (
            triangle_time_factor, triangle_slope_factor, triangle_safety_factor)

        if test_mode:
            itineraries_str = \
                "{itineraries {startTime endTime legs{mode duration distance}}}"
        else:
            itineraries_str = "{itineraries {startTime endTime}}"
        qry_str = "{plan(%s %s %s %s %s %s)%s}" % (
            origin_str, destination_str, modes_str, bike_speed_str, optimize_str,
            triangle_str, itineraries_str
        )
        request = requests.post(
            self._request_api, headers=self.HEADERS, json={'query': qry_str})
        response = request.json()
        return self._parse_json_response(
            response, False, triptime=None, test_mode=test_mode)

    def _compute_transit_trip_traveltime(
            self, origin, destination, triptime, speed_walking, 
            skip_test_trip_date=False, test_mode=False
        ):
        """ 
        Requests a transit/walk trip from OTP, returing total time and walk distance.
        """
        if origin == destination:
            return 0.0
        if not skip_test_trip_date:
            self._test_departure_within_service_time_range(triptime)
        date_str = f"{triptime.year}-{triptime.month:02d}-{triptime.day:02d}"
        time_str = f"{triptime.hour:02d}:{triptime.minute:02d}:{triptime.second:02d}"
        origin_str = self._set_pt_str("from", origin)
        destination_str = self._set_pt_str("to", destination)
        modes_str = "[{mode: WALK}, {mode: TRANSIT}]"
        walk_speed_str = f"{speed_walking / 3.6}" 
        if test_mode:
            itineraries_str = \
                "{itineraries {startTime endTime walkDistance generalizedCost legs{mode duration distance}}}"
        else:
            itineraries_str = "{itineraries {endTime}}" 

        qry_str = '{' + f'plan({origin_str} {destination_str} transportModes: {modes_str} date: "{date_str}" time: "{time_str}" ' + \
                  f'arriveBy: {str(False).lower()} walkSpeed: {walk_speed_str}){itineraries_str}' + '}' 
        qry = {
            'query': qry_str
        }
        request = requests.post(
            self._request_api, headers=self.HEADERS, json=qry)
        response = request.json()
        return self._parse_json_response(
            response, True, triptime, test_mode=False)
        
    def _parse_json_response(
            self, response, is_transit, triptime, *, test_mode=False):
        """ Parse the JSON response to a OTP2 request. """
        try:
            itineraries = response['data']['plan']['itineraries']
        except KeyError:
            raise RuntimeError("Invalid response returned." )
        
        if len(itineraries) == 0:
            return np.NaN
        if test_mode:
            # In test mode, return full itinerary for additional testing
            return itineraries
        else:
            if not is_transit:   # There should only be one itinerary
                it = itineraries[0]
                return (it['endTime'] - it['startTime']) // 1000
            else:   
                # transit, hence need to find the fastest itinerary
                # use the desired trip start time (from the request) and not the 
                # actual trip start time.
                min_duration = 9.999e15
                for it in response['data']['plan']['itineraries']:
                    duration = (it['endTime'] // 1000 - triptime.timestamp())
                    if duration < min_duration:
                        min_duration = duration
                return min_duration


    def _compute_interval_transit_traveltime(
            self, origin, destination, departure, departure_time_window, 
            time_increment, speed_walking, skip_test_trip_date=False):
        """ 
        Requests median travel time over interval, inclusive at interval start, 
        exclusive at interval end. """
        if not skip_test_trip_date:
            self._test_departure_within_service_time_range(departure)

        elapsed_time = timedelta(0)
        travel_times = []
        trip_departure = departure
        while True:
            trip_time = self._compute_transit_trip_traveltime(
                origin, destination, trip_departure, speed_walking=speed_walking, 
                skip_test_trip_date=True, test_mode=False)
            travel_times.append(trip_time)
            trip_departure = trip_departure + time_increment
            elapsed_time += time_increment
            if elapsed_time >= departure_time_window:  # Exclusive at trip end
                break
        return np.median(travel_times)

    def _test_departure_within_service_time_range(self, date_time):
        """ Test if provided date is within the graph service time range. """
        test_posix = date_time.timestamp()

        # Test that the OTP server is up and running
        self.test_otp_server()

        # Find the POSIX representation of the graph start/end times 
        json_data = {
            'query': '{serviceTimeRange {start end} }'
        }
        
        request = requests.post(
            'http://localhost:8080/otp/routers/default/index/graphql', 
            headers=self.HEADERS, 
            json=json_data
        )
        result = request.json()
        
        start_posix = result['data']['serviceTimeRange']['start']
        end_posix = result['data']['serviceTimeRange']['end']

        if not start_posix <= test_posix <= end_posix:
            start_end_dates = np.array(
                [start_posix, end_posix], dtype='datetime64[s]')
            raise ValueError(
                f"Trip start {date_time} is not within graph start/end dates: {start_end_dates}")

    @staticmethod
    def _set_pt_str(from_to: str, pt: Point):
        return "%s: {lat: %f, lon: %f}" % (from_to, pt.y, pt.x)

    @staticmethod
    def _set_modes_str(modes: list):
        modes_str_int = ""
        for mode in modes:
            modes_str_int = modes_str_int + "{mode: %s}, " % mode
        modes_str_int = modes_str_int[:-2]    # Take out the final comma and space
        return "transportModes: [%s]" % modes_str_int
 
#endregion

#region property methods
    @property
    def java_path(self):
        return self._java_path

    @java_path.setter
    def java_path(self, new_path):
        new_path = Path(new_path)
        if not new_path.is_file:
            raise FileNotFoundError("Invalid java path.")
        self._java_path = new_path

    @property
    def otp_jar_path(self):
        return self._otp_jar_path
    
    @otp_jar_path.setter
    def otp_jar_path(self, new_path):
        new_path = Path(new_path)
        if not new_path.is_file:
            raise FileNotFoundError("Invalid OTP jar path.")
        self._otp_jar_path = new_path

    @property
    def graph_dir(self):
        return self._graph_dir

    @graph_dir.setter
    def graph_dir(self, new_path):
        new_path = Path(new_path)
        if not new_path.is_dir:
            raise FileNotFoundError("Invalid path to graph directory.")
        self._graph_dir = new_path

    @property
    def memory_str(self):
        return self._memory_str
    
    @memory_str.setter
    def memory_str(self, new_memory_str):
        self._memory_str = new_memory_str

    @property
    def request_host_url(self):
        return self._request_host_url

    @request_host_url.setter
    def request_host_url(self, new_url):
        self._request_host_url = new_url
        self._request_api = self._request_host_url + "/otp/routers/default/index/graphql"

    @property
    def max_server_sleep_time(self):
        return self._max_server_sleep_time
    
    @max_server_sleep_time.setter
    def max_server_sleep_time(self, new_sleep_time):
        if not (isinstance(new_sleep_time, float) or isinstance(new_sleep_time, int)):
            raise TypeError("max_server_sleep_time must be a float or int value.")
        if new_sleep_time < 0.0:
            raise ValueError("max_server_sleep_time must be a number >= 0.")
        self._max_server_sleep_time = new_sleep_time

#endregion