from datetime import datetime, timedelta
from geopandas import GeoDataFrame
from  os import environ
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from shapely import Point
from shutil import rmtree, copy2
from subprocess import Popen, run
from time import sleep

from ..enumerations import DEFAULT_SPEED_WALKING, DEFAULT_DEPARTURE_WINDOW, OTP_DEPARTURE_INCREMENT
from ..utilities import test_od_input


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
    OTP_JAR_PATH_ERROR = "Must defined the path to the OTP 2.4 jar file before running Open Trip Planner."

    HEADERS = {
        'Content-Type': 'application/json',
        'OTPTimeout': '180000',
    }

    def __init__(self):
        
        # These attributes will need to be defined before starting an OTP instance
        self._java_path = None
        self._otp_jar_path = None
        self._memory_str = "1G"

        # These default URLs assume that the server is hosted locally. These can be changed by a user if desired
        self._request_host_url = "http://localhost:8080"
        self._request_api = self._request_host_url + "/otp/routers/default/index/graphql"
        
        # This allows time for a server to fully load before running requests
        # Setting this too low will cause problems as the server won't be available when subsequent actions are run.
        self._max_server_sleep_time = 30

#region Graph and server
    def build_network(self, osm_pbf, gtfs, launch_server=True):
        """ Build a transport network from specified OSM and GTFS files.

        Arguments
        ---------
        osm_pbf : str | pathlib.Path
            file path of an OpenStreetMap extract in PBF format
        gtfs : str | pathlib.Path | list[str] | list[pathlib.Path]
            path(s) to public transport schedule information in GTFS format
        launch_server: bool, optional
            if True, launch the OTP server

        Returns
        -------
        None

        """
        # We can do this by copying all files to a temp directory, and then building the graph from that directory
        # using the build_graph_from_dir method
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


    def build_network_from_dir(self, path, overwrite=False, launch_server=True):
        """ Builds a transport network given a directory containing OSM and GTFS files.

        Arguments
        ---------
        path : str or pathlib.Path
            directory path in which to search for GTFS and .osm.pbf files
        overwrite:
            If True, overwrite any existing network
            Will raise if set to False and there is an existing network.
        launch_server: bool, optional
            if True, launch the OTP server

        Returns
        -------
        None

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

    def launch_local_otp2_server(self, path) -> None:
        """ Launches OTP 2 server on the local machine using previously built network. 

        
        Arguments
        ---------
        path : str or pathlib.Path
            Directory containg OTP network

        Notes
        -----
        This method will wait until a connection is made to the server before finishing, up until 
        a number of seconds defined in the attribute `max_server_sleep_time`
        
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

        Tests server at the address defined in `request_host_url` attribute. 
        
        Raises
        ------
        requests.ConnectionError
            Raised if cannot connect 
        RuntimeError
            Raised if null values are returned from the request
        """

        host = requests.get(self._request_host_url)
        if host is None:
            raise RuntimeError("Null values returned from OTP request.")
#endregion
        
#region Walk Trips

    def compute_walk_traveltime_matrix(self, origins, destinations, speed_walking=None):
        """ Requests walk-only trip matrix from OTP, returing either duration, trip distance or OTP's generalized cost.

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

            Returns
            -------
            pd.DataFrame
                Travel time matrix in the same format as returned by r5py:
                    - tall format
                    - columns are 'from_id', 'to_id', 'travel_time'
                    - travel time is median calculated travel time for trips in departure interval
                    - np.nan if not connection found with given parameters.
        """

        test_od_input(origins)
        if destinations is not None:
           test_od_input(destinations)
        else:
            destinations = origins.copy()
        
        mi = pd.MultiIndex.from_product([origins['id'], destinations['id']], names=['from_id', 'to_id'])
        df = pd.DataFrame(index=mi, columns=['travel_time'], data=np.nan)
        for _, origin in origins.iterrows():
            origin_id = origin['id']
            origin_pt = origin['geometry']
            for _, destination in destinations.iterrows():
                destination_id = destination['id']
                destination_pt = destination['geometry']
                r = self.compute_walk_trip_traveltime(origin_pt, destination_pt, speed_walking, False)
                df.loc[(origin_id, destination_id), 'travel_time'] = r
        df.name = "walk_cost_matrix_from_otp2"
        return self._reformat_df_to_r5py_style(df)
    
    def compute_walk_trip_traveltime(self, origin, destination, speed_walking=None, test_mode=False):
        """ Requests a walk-only trip from OTP, returing walk time and distance.

            Parameters
            ----------
            origin: shapely.Point
                Point containing x,y location of trip start
            destination: shapely.Point
                Point containing x,y location of trip end
            speed_walking: float or None
                Walk speed in kilometres per hour.
                If None, set this is set to the default walk speed; currently 5 km/hr.
                
            Returns
            -------
            float
                Trip duration in minutes

            Other Parameters
            ----------------
            test_mode: bool, optional
                if True, returns request itineraries instead of usual return dictionary. 
                This is intended only to be run from test scripts. 

            Notes
            -----
            For a walk trip in OTP, we do not need to specify a date and time. 

        """
        if origin == destination:
            return 0.0
        if speed_walking is None:
            speed_walking = DEFAULT_SPEED_WALKING
        origin_str = self._set_pt_str("from", origin)
        destination_str = self._set_pt_str("to", destination)
        modes_str = self._set_modes_str(["WALK"])
        walk_speed_str = f"walkSpeed: {speed_walking / 3.6}"  # Convert walk speed to metres per second
        itineraries_str = self._set_itineraries_str()
        qry_str = "{plan(%s %s %s %s)%s}" % (origin_str, destination_str, modes_str, walk_speed_str, itineraries_str)
        qry = {
            'query': qry_str
        }
        request = requests.post(self._request_api, headers=self.HEADERS, json=qry)
        result = request.json()

        try:
            itineraries = result['data']['plan']['itineraries']
        except KeyError:
            raise RuntimeError("Invalid query string:  '%s'" % qry_str)

        if len(itineraries) == 0:
            return np.NaN
        if len(itineraries) > 1:
            raise RuntimeError("More than one itinerary found for walk trip. This is unexpected to look into this.")

        if not test_mode:
            # Default mode, returns trip duration and walk distance
            return (itineraries[0]['endTime'] - itineraries[0]['startTime']) // 1000 // 60
        else:
            # In test mode, return full itinerary for additional testing
            return itineraries
#endregion
        
#region bikes

    def compute_bike_traveltime_matrix(self, origin, destination, speed_biking=None, test_mode=False):
        """ Requests a bike-only trip from OTP, returing walk time and distance. """
        # todo: need to explore more for bike trips, including possible changes to the router-config file
        raise NotImplementedError("Further testing is required for this library to support bike travel. ")

#endregion
    
#region transit
    def compute_transit_traveltime_matrix(
            self, origins, destinations=None, departure=datetime.now(), 
            departure_time_window=DEFAULT_DEPARTURE_WINDOW, time_increment=OTP_DEPARTURE_INCREMENT, 
            speed_walking=DEFAULT_SPEED_WALKING):
        """ Requests walk/transit trip matrix from OTP, returing either trip duration in minutes.

            Parameters
            ----------
            origins: geopandas.GeoDataFrame
                Origin points.  Has to have at least an ``id`` column and a geometry
            destinations: geopandas.GeoDataFrame or None, optional
                Destination points. If None,use the origin points also as destinations, 
                mimicking r5py's behaviour. Default is None.
                If not None, has to have at least an ``id`` column and a geometry.
            departure : datetime.datetime
                r5py will find public transport connections leaving every minute within
                ``departure_time_window`` after ``departure``. Default: current date and time
            departure_time_window : datetime.timedelta
                (see ``departure``) Default: 60 minutes
            time_increment: int or None, optional
                Increment between different trip runs in minutes.
                If None, set this is set to the default interval; currently 1 minute.
            speed_walking: float or None, optional
                Walking speed in kilometres per hour.
                If None, set this is set to the default walk speed; currently 5 km/hr.
                
            Returns
            -------
            pd.DataFrame
                Travel time matrix in the same format as returned by r5py:
                    - tall format
                    - columns are 'from_id', 'to_id', 'travel_time'
                    - travel time is median calculated travel time for trips in departure interval
                    - np.nan if not connection found with given parameters.

            Notes
            -----
            For transit, returning trip duration appears to be the only useful measure. The generalized cost
            does not appear to include waiting for the first bus.

        """
        test_od_input(origins)
        if destinations is not None:
           test_od_input(destinations)
        else:
            destinations = origins.copy()

        # Check that we're within the start/end time of the graph
        self.test_departure_within_service_time_range(departure)

        mi = pd.MultiIndex.from_product([origins['id'], destinations['id']], names=['from_id', 'to_id'])
        df = pd.DataFrame(index=mi, columns=['travel_time'], data=np.nan)

        for _, origin in origins.iterrows():
            origin_id = origin['id']
            origin_pt = origin['geometry']
            for _, destination in destinations.iterrows():
                destination_id = destination['id']
                destination_pt = destination['geometry']
                r = self.compute_interval_transit_traveltime(
                    origin_pt, destination_pt, departure, departure_time_window, 
                    time_increment, speed_walking, skip_test_trip_date=True)
                df.loc[(origin_id, destination_id), 'travel_time'] = r
        df.name = "walk_cost_matrix_from_otp2"
        return self._reformat_df_to_r5py_style(df)

            

    def compute_interval_transit_traveltime(
            self, origin, destination, departure=datetime.now(), departure_time_window=DEFAULT_DEPARTURE_WINDOW, 
            time_increment=OTP_DEPARTURE_INCREMENT, speed_walking=DEFAULT_SPEED_WALKING, skip_test_trip_date=False):
        """ Requests median travel time over interval, inclusive at interval start, exclusive at interval end.

            Parameters
            ----------
            origin: shapely.Point
                Point containing x,y location of trip start
            destination: shapely.Point
                Point containing x,y location of trip end
            departure : datetime.datetime
                r5py will find public transport connections leaving every minute within
                ``departure_time_window`` after ``departure``. Default: current date and time
            departure_time_window : datetime.timedelta
                (see ``departure``) Default: 60 minutes
            time_increment: int or None, optional
                Increment between different trip runs in minutes.
                If None, set this is set to the default interval; currently 1 minute.
            speed_walking: float or None, optional
                Walking speed in kilometres per hour.
                If None, set this is set to the default walk speed; currently 5 km/hr.


            Returns
            -------
            float:
                Median of trip durations over the time interval in minutes.
        
            
            Other Parameters
            ----------------
            skip_test_trip_date: bool, optional
                if true, then do not test the trip start time. This is meant as an efficiency parameter when requesting
                multiple trip costs, such as when calculating a travel time matrix.

        """
        if not skip_test_trip_date:
            self.test_departure_within_service_time_range(departure)

        elapsed_time = timedelta(0)
        travel_times = []
        trip_departure = departure
        while True:
            trip_time = self.compute_transit_traveltime(
                origin, destination, trip_departure, arrive_by=False, speed_walking=speed_walking, 
                skip_test_trip_date=True, test_mode=False)
            travel_times.append(trip_time)

            trip_departure = trip_departure + time_increment
            elapsed_time += time_increment
            if elapsed_time >= departure_time_window:  # Exclusive at trip end
                break
        return np.median(travel_times)


    def compute_transit_traveltime(self, origin, destination, triptime=datetime.now(), arrive_by=False, speed_walking=DEFAULT_SPEED_WALKING, 
                                    skip_test_trip_date=False, test_mode=False):
        """ Requests a transit/walk trip from OTP, returing total time and walk distance.

            Parameters
            ----------
            origin: shapely.Point
                Point containing x,y location of trip start
            destination: shapely.Point
                Point containing x,y location of trip end
            triptime : datetime.datetime
                Trip departure or arrive-by time.
            arrive_by: bool, optional
                Flag if 'triptime' reflects latest arrival time (if True) or departure time (if False).
                Defaults to False
            speed_walking: float or None, optional
                Walking speed in kilometres per hour.
                If None, set this is set to the default walk speed; currently 5 km/hr.

            Returns
            -------
            float:
                Trip travel time in minutes.
                
            Other Parameters
            ----------------
            skip_test_trip_date: bool, optional
                if true, then do not test the trip start time. This is meant as an efficiency parameter when requesting
                multiple trip costs, such as when calculating a travel time matrix.

            test_mode: bool, optional
                if True, returns request itineraries instead of usual return dictionary. 
                This is intended only to be run from test scripts. 

        """
        if origin == destination:
            return 0.0
        if not skip_test_trip_date:
            self.test_departure_within_service_time_range(triptime)

        date_str = f"{triptime.year}-{triptime.month:02d}-{triptime.day:02d}"
        time_str = f"{triptime.hour:02d}:{triptime.minute:02d}:{triptime.second:02d}"
        origin_str = self._set_pt_str("from", origin)
        destination_str = self._set_pt_str("to", destination)
        modes_str = self._set_modes_str(["WALK", "TRANSIT"])
        itineraries_str = self._set_itineraries_str()

        qry_str = '{' + f'plan({origin_str} {destination_str} {modes_str} date: "{date_str}" time: "{time_str}" ' + \
                  f'arriveBy: {str(arrive_by).lower()} walkSpeed: {speed_walking / 3600 * 1000} ){itineraries_str}' + '}' 
        qry = {
            'query': qry_str
        }
        request = requests.post(self._request_api, headers=self.HEADERS, json=qry)
        result = request.json()

        try:
            itineraries = result['data']['plan']['itineraries']
        except KeyError:
            raise RuntimeError("Invalid query string:  '%s'" % qry_str)

        if len(itineraries) == 0:
            return np.NaN

        if not test_mode:
            # Default mode, returns trip duration and walk distance
            # Find and return the details from the itinerary with the shortest travel time
            # Note that the trip startTime and endTime are in posix milliseconds, hence / 1000
            # to convert to posix seconds, which is what Python's datetime.timestamp uses.
            min_duration = 9.999e15
            for it in result['data']['plan']['itineraries']:
                if arrive_by:
                    duration = (triptime.timestamp() - it['startTime'] // 1000) // 60
                else:
                    duration = (it['endTime'] // 1000 - triptime.timestamp()) // 60
                if duration < min_duration:
                    min_duration = duration
            return min_duration
        else:
            # In test mode, return full itinerary for additional testing
            return itineraries
                   
    def test_departure_within_service_time_range(self, date_time):
        """ Test if provided date is within the graph service time range. 

        Parameters
        ----------
        date_time: datetime.datetime


        Raises
        ------
        ValueError: 
            if date is not within service time range
            OTP server is not running

        """
        test_posix = date_time.timestamp()

        # Test that the OTP server is up and running
        self.test_otp_server()

        # Find the POSIX representation of the graph start/end times 
        json_data = {
            'query': '{serviceTimeRange {start end} }'
        }
        
        request = requests.post('http://localhost:8080/otp/routers/default/index/graphql', headers=self.HEADERS, json=json_data)
        result = request.json()
        
        start_posix = result['data']['serviceTimeRange']['start']
        end_posix = result['data']['serviceTimeRange']['end']

        if not start_posix <= test_posix <= end_posix:
            start_end_dates = np.array([start_posix, end_posix], dtype='datetime64[s]')
            raise ValueError(f"Trip start {date_time} is not within graph start/end dates: {start_end_dates}")
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

#region Hidden methods   
    @staticmethod
    def _reformat_df_to_r5py_style(df):
        """ Converts the output matrix to a style matching that from r5py. """
        df = df.reset_index()
        # My original approach was to cast to integer. This caused issues if NaNs are returned,
        # hence I'm now rounding the values but keeping as float.
        df['travel_time'] = df['travel_time'].round()
        return df

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

    @staticmethod
    def _set_itineraries_str():
        return "{itineraries {startTime endTime walkDistance generalizedCost legs{mode duration}}}"
  
#endregion
