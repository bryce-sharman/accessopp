from datetime import datetime, timedelta
from geopandas import GeoSeries
from multiprocessing import cpu_count
import numpy as np
from os import PathLike
import pandas as pd
from pathlib import Path
import requests
from shapely import Point
import shutil
import subprocess
from typing import Dict, Optional
import yaml
import zipfile

from time import sleep

from accessopp.enumerations import DEFAULT_SPEED_WALKING, DEFAULT_SPEED_CYCLING
from accessopp.enumerations import DEFAULT_DEPARTURE_WINDOW, DEFAULT_TIME_INCREMENT
from accessopp.enumerations import N_DECIMALS

from accessopp.utilities import test_file_existence
from accessopp.utilities import test_dir_existence, empty_directory_recursive
from accessopp.utilities import validate_origins_destinations, create_blank_ttmatrix


NEW_BUILD_MSG = """
This method will create a Valhalla network in a new Docker container. Before 
creation, the following will be removed from the Docker volume mount before 
being replaced with the updated versions:
    - all OSM network files (.osm.pbf)
    - all existing network tiles in the 'valhalla_tiles' subfolder
    - all elevation data
    - all GTFS data, if available
Please enter ('Y' or 'y' to continue, or any other key to exit.)
"""
NO_DOCKER_ERR_MSG = "Error running Docker. Docker Desktop must be installed " \
                    "and the Docker Engine started to run Valhalla routing."
NO_EXISTING_OSM_TILES_ERR_MSG = \
    "OSM file not defined, and no existing tiles have been built."
VALHALLA_DOCKER_COMPOSE_TEMPLATE = {
    'version': '3.4',
    'services': {
        'valhalla': {
            'image': 'ghcr.io/gis-ops/docker-valhalla/valhalla:latest',
            'container_name': 'valhalla_latest',
            'ports': ['8002:8002'],
            'volumes': [],
            'environment': []
        }
    }
}


class ValhallaTravelTimeComputer():
    """ Class to calculate travel times using Valhalla routing engine.

    From the Valhalla documentation:
    Valhalla is an open source routing engine and accompanying libraries for 
    use with OpenStreetMap data. Valhalla also includes tools like time+distance 
    matrix computation, isochrones, elevation sampling, map matching and tour 
    optimization (Travelling Salesman).

    The Valhalla documentation recommends running Valhalla using Docker (see 
    https://github.com/valhalla/valhalla?tab=readme-ov-file#installation). 
    Hence Docker must be installed on your computer to run Valhalla. 
    In this package, we are using the GIS-OPS Valhalla Docker container.

    The elevation data is accessed automatically when building the network.
    If running this software behind a firewall or other internet security
    (such as ZScaler) you will need the certificate to the website 
    containing the elevation data, which is 
    https://elevation-tiles-prod.s3.us-east-1.amazonaws.com/ 
    See the wiki to learn how to create this certificate.


    Parameters
    ---------
    custom_files_path: 
        Directory that is shared with Valhalla.

    Methods
    -------
    build_network:
    
    Attributes
    ----------
    container_id: str
        Id of the running Docker container used for Valhalla routing.
    custom_files_path: Path
        Working directory, also the volume_mount for the Docker container
        through which files are transferred into and out of the Docker
        container.
    request_host_url: str
        URL to which request calls are made (done in the Docker container)
        Defaults to "http://localhost:8002", where the server is on
        the local directory.

    """
        
    def __init__(self, custom_files_path: PathLike):     

        self.container_id:str = None
        self.custom_files_path = custom_files_path
        # This default URL assumes that the server is hosted locally. 
        # This can be changed by a user if desired
        self.request_host_url = "http://localhost:8002"
        # Request header, should not need to be changed
        self._headers = {
            'Content-Type': 'application/json'
        }
        # Ensure that Docker is installed and can be run
        self._test_docker()

    def build_network(
            self, 
            osm_paths: PathLike | list[PathLike] | None = None,
            gtfs_paths: PathLike | list[PathLike] | None = None,
            * , 
            force_rebuild=False,
            server_threads: int | None = None
        ) -> None: 
        """ Build a transport network from specified network files.

        Arguments
        ---------
        osm_paths: 
            Absolute path files to OSM networks, stored in binary (.pbf) format.
            Can be a single path or a list of multiple paths.
            If None then will start a container using previously built network.
        gtfs_paths: 
            Path to  public transport schedule information in GTFS format.
            Can be a single or list of GTFS files in zipped format. 
            If None then transit network will not be built.
            Previously-built transit network will still be used if osm_paths 
            is also None.
        force_rebuild:
            If True, then forces a bebuild of routing tiles. Defaults is False.
        server_threads: 
            Number of threads to use by Docker.
            Defaults to number of threads in the computer - 1.       

        """
        # Be sure that user knows they will overwrite custom_files
        response = input(NEW_BUILD_MSG)
        if response not in ('Y', 'y'):
            return

        # Vahalla can accept multiple OSM and GTFS files
        # Ensure that osm_paths and gtfs_paths are list so we know their type
        osm_paths = osm_paths if isinstance(
            osm_paths, list) else [osm_paths]
        gtfs_paths = gtfs_paths if isinstance(
            gtfs_paths, list) else [gtfs_paths]

        # Remove OSM files from the custom_folders to avoid confusion
        print("Removing existing .osm.pbf  and GTFS files from custom files "
              "directory.")
        for osm_file in self._custom_files_path.glob("*.osm.pbf"):
            (self._custom_files_path / osm_file).unlink()
        empty_directory_recursive(self._gtfs_dir)

        # Prepare OSM files
        if osm_paths is None:
            use_tiles_ignore_pbf = True
            # Check that valhalla_tiles directory and valhalla_tiles.tar exist
            test_file_existence(self._custom_files_path / 'valhalla.tar',
                                NO_EXISTING_OSM_TILES_ERR_MSG)
            test_dir_existence(self._custom_files_path / 'valhalla_tiles',
                               NO_EXISTING_OSM_TILES_ERR_MSG)
        else:
            use_tiles_ignore_pbf = False
            print("Copying OSM files to custom_files directory.")
            for osm_path in osm_paths:
                osm_path = Path(osm_path)
                shutil.copyfile(
                    osm_path, self._custom_files_path / osm_path.name)

        if gtfs_paths is None:
            build_transit = False
        else:
            build_transit=True
            print("Extracting GTFS files to custom_files directory.")
            for gtfs_path in gtfs_paths:
                gtfs_path = Path(gtfs_path)
                with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
                    zip_ref.extractall(self._gtfs_dir / gtfs_path.name)      

        self._start_container(
            use_tiles_ignore_pbf=use_tiles_ignore_pbf, 
            build_transit=build_transit, 
            force_rebuild=force_rebuild, 
            server_threads=server_threads
        )

    def build_network_from_dir(
            self, 
            path: PathLike,
            * , 
            force_rebuild: bool=False,
            server_threads: int | None = None) -> None:
        """ Builds transport network from directory containing OSM and GTFS data.

        Arguments
        ---------
        path: 
            Absolute path to directory containing OSM and, optionally, 
            GTFS files
        force_rebuild:
            If True, then forces a bebuild of routing tiles. Default is False.
        server_threads: 
            Number of threads to use by Docker.
            If None, Valhalla uses the threads in the computer.       
            Default is None.

        """
        # Be sure that user knows they will overwrite custom_files
        # response = input(new_build_msg)
        # if response not in ('Y', 'y'):
        #     return

        path = Path(path)
        test_dir_existence(path, f'Directory not found: {path.as_posix()}')

        # Remove OSM files from the custom_folders to avoid confusion
        print("Removing existing .osm.pbf  and GTFS files from custom files "
              "directory.")
        for osm_file in self._custom_files_path.glob("*.osm.pbf"):
            (self._custom_files_path / osm_file).unlink()
        empty_directory_recursive(self._gtfs_dir)

        print("Copying OSM files to custom_files directory.")
        use_tiles_ignore_pbf = False
        osm_paths = list(path.glob('*.osm.pbf'))
        for osm_path in osm_paths:
            osm_path = Path(osm_path)
            shutil.copyfile(
                osm_path, self._custom_files_path / osm_path.name)

        gtfs_paths = list(path.glob('*gtfs*.zip'))
        if gtfs_paths is None:
            build_transit = False
        else:
            build_transit=True
            print("Extracting GTFS files to custom_files directory.")
            for gtfs_path in gtfs_paths:
                gtfs_path = Path(gtfs_path)
                with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
                    zip_ref.extractall(self._gtfs_dir / gtfs_path.name)      

        self._start_container(
            use_tiles_ignore_pbf=use_tiles_ignore_pbf, 
            build_transit=build_transit, 
            force_rebuild=force_rebuild, 
            server_threads=server_threads
        )

    def compute_walk_traveltime_matrix(
            self,
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            speed_walking: float = DEFAULT_SPEED_WALKING,
            **kwargs
        ) -> pd.Series:
        """ 
        Requests walk-only trip matrix from Valhalla, returing trip durations 
        in seconds.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            speed_walking: Walking speed in kilometres per hour. If None then
                this is set to the default walk speed; currently 5 km/hr.
            **kwargs:
                Additional Valhalla cost parameters (costing options), 
                as defined in 
                https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#pedestrian-costing-options
                Anticipated parameters expected to be of particular interest 
                for walk-only trips include:
                - max_distance: Maximum total walking distance of a route. 
                - use_hills: Range of values from 0 to 1, where 0 attempts to  
                    avoid hills and steep grades even if it means a longer (time 
                    and distance) path, while 1 indicates the pedestrian does 
                    not fear hills and steeper grades. Valahlla default is 0.5.

        Returns:
            Travel times matrix in stacked (tall) format. 

        """
        origins, destinations = validate_origins_destinations(
            origins, destinations) 
        sources, targets = self._create_sources_and_targets_json_calls(
            origins, destinations)
        # set the costing options JSON calls
        walking_cost_options = {'walking_speed': str(speed_walking)}
        for k, v in kwargs.items():
            walking_cost_options[k] = v
        costing_options = {'pedestrian': walking_cost_options}
        full_call = {
            'sources': sources,
            'targets': targets,
            'costing': 'pedestrian',
            'costing_options': costing_options
        }
        r = requests.get(
            self._s2t_request_api, headers=self._headers, json=full_call)
        return _process_valhalla_matrix_result(r.json(), origins, destinations)


    def compute_bike_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            speed_cycling: float = DEFAULT_SPEED_CYCLING, 
            **kwargs
        ) -> pd.Series:
        """ 
        Requests bike-only trip matrix from OTP, returing trip durations 
        in seconds.

        Args:
            origins: Origin points.  Index is the point ids.
            destinations: Destination points. If None, then will use the origin 
                points. Default is None
            speed_cycling: Cycling speed in kilometres per hour. If None then
                this is set to the default cycling speed; currently 18 km/hr.
            kwargs:
                Additional Valhalla cost parameters (costing options), 
                as defined in 
                https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#bicycle-costing-options
                Anticipated parameters expected to be of particular interest 
                for bike-only trips include:
                - bicycle_type: one of Road, Hybrid, City, Cross, Mountain
                    sets the default cycling speed
                - use_roads: A cyclist's propensity to use roads alongside other 
                    vehicles. This is a range of values from 0 to 1, where 0 
                    attempts to avoid roads and stay on cycleways and paths, and 1 
                    indicates the rider is more comfortable riding on roads. 
                    Valhalla default is 0.5, but we will likely want lower values.
                - use_hills: A cyclist's propensity to tackle hills in their routes. 
                    This is a range of values from 0 to 1, where 0 
                    where 0 attempts to avoid hills and steep grades even if it 
                    means a longer (time and distance) path, while 1 indicates the 
                    rider does not fear hills and steeper grades. 
                    Valhalla default is 0.5, but we will likely want lower values.

        Returns:
            Travel times matrix in stacked (tall) format.

        """
        raise NotImplementedError(
            "Cannot currently compute bike travel time matrices. "
            "See Valhalla issue: \n"
            "https://github.com/valhalla/valhalla/issues/4861"
        )
        origins, destinations = validate_origins_destinations(
            origins, destinations) 
        sources, targets = self._create_sources_and_targets_json_calls(
            origins, destinations)
        # set the costing options JSON calls
        cycling_cost_options = {'cycling_speed': str(speed_cycling)}
        for k, v in kwargs.items():
            cycling_cost_options[k] = v
        full_call = {
            'sources': sources,
            'targets': targets,
            'costing': 'bicycle',
            'costing_options': {'bicycle': cycling_cost_options}
        }
        r = requests.get(
            self._s2t_request_api, headers=self._headers, json=full_call)
        return _process_valhalla_matrix_result(r.json(), origins, destinations)
    

    def compute_transit_traveltime_matrix(
            self, 
            origins: GeoSeries, 
            destinations: Optional[GeoSeries], 
            departure: datetime, 
            departure_time_window: timedelta=DEFAULT_DEPARTURE_WINDOW, 
            time_increment: timedelta=DEFAULT_TIME_INCREMENT, 
            speed_walking: float=DEFAULT_SPEED_WALKING,
            **kwargs
        ) ->pd.Series:
        """ 
        Requests multimodal transit trip matrix from Valhalla, returing 
        trip durations in seconds.

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
            speed_walking: Walking speed in kilometres per hour. If None then
                this is set to the default walk speed; currently 5 km/hr.
            **kwargs:
                Additional Valhalla cost parameters (costing options), 
                as defined in 
                https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#pedestrian-costing-options
                and
                https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#transit-costing-options
                Anticipated parameters expected to be of particular interest 
                for multi-modal transit trips include:
                Pedestrian costing options:
                - max_distance: Maximum total walking distance of a route. 
                - use_hills: Range of values from 0 to 1, where 0 attempts to  
                    avoid hills and steep grades even if it means a longer (time 
                    and distance) path, while 1 indicates the pedestrian does 
                    not fear hills and steeper grades. Valahlla default is 0.5.
                Transit costing options
                - use_bus: User's desire to use buses. Range of values from 0 
                  (try to avoid buses) to 1 (strong preference for riding buses)
                - use_rail:	User's desire to use rail/subway/metro. Range of 
                  values from 0 (try to avoid rail) to 1 (strong preference for 
                  riding rail).
                - use_transfers: User's desire to favor transfers. Range of 
                  values from 0 (try to avoid transfers) to 1 (totally 
                  comfortable with transfers).

        Returns:
            Travel times matrix in stacked (tall) format. 

        """
        origins, destinations = validate_origins_destinations(
            origins, destinations)
        ttm = create_blank_ttmatrix(origins, destinations)
        for o_id, o_pt in origins.items():
            for d_id, d_pt in destinations.items():
                tt = self._compute_interval_transit_traveltime(
                    o_pt, d_pt, departure, departure_time_window, 
                    time_increment, speed_walking, **kwargs)
                ttm.at[(o_id, d_id)] = tt
        return ttm.round(N_DECIMALS)


    def _compute_transit_trip_traveltime(
            self,
            origin: Point, 
            destination: Point, 
            triptime: datetime,
            speed_walking: Optional[float] = None, # keeping generic at this level
            **kwargs
        ) -> float:
        """ 
        Requests a transit/walk trip from OTP, returing trip time in seconds.

        Notes:
            Valhalla does not appear to test if the start date is within 
            GTFS intervals. Keeping this behaviour, at least for now.
        """
        if origin == destination:
            return 0.0
        costing_options = _create_transit_costing_options_call(
            speed_walking, **kwargs)
        full_call = {
            'locations': [
                {'lat': origin.y, 'lon': origin.x}, 
                {'lat': destination.y, 'lon': destination.x}
            ],
            'date_time': {
                'type': 1, # 1 means: specified departure time
                'value': triptime.isoformat(sep='T', timespec='minutes')
            },
            'costing': 'multimodal',
            'directions_type': 'none',
        }
        if costing_options:
            full_call['costing_options'] = costing_options
        r = requests.get(
            self._route_request_api, headers=self._headers, json=full_call)
        r_json = r.json()
        trip_response = r_json['trip']
        if trip_response['status_message'] != 'Found route between points':
            raise RuntimeError(
                f"Error computing trip, here's the entire response: \n{r_json()}")
        return trip_response['summary']['time']


    def _compute_interval_transit_traveltime(
            self, 
            origin: Point, 
            destination: Point, 
            time_window_start: datetime, 
            time_window_duration: timedelta, 
            time_increment: timedelta, 
            speed_walking: Optional[float] = None,  # keeping generic at this level
            **kwargs
        ) -> float:
        """ 
        Calculates median transit travel time over interval, inclusive at 
        interval start, exclusive at interval end. 
        """
        elapsed_time = timedelta(0)
        travel_times = []
        trip_time = time_window_start
        while True:
            travel_time = self._compute_transit_trip_traveltime(
                origin, destination, trip_time, speed_walking=speed_walking, 
                **kwargs)
            travel_times.append(travel_time)
            trip_time += time_increment
            elapsed_time += time_increment

            if elapsed_time >= time_window_duration:  # Exclusive at trip end
                break
        return np.median(travel_times)

    def _test_valhalla_status(
            self, 
            max_wait: timedelta=timedelta(seconds=120)
        ) -> None:
        """ Test that we can access Valhalla service from local computer.

        This function will test for a connection each second up until
        a maximum wait time, defined by `max_wait`.
        
        Args:
            max_wait: Maximum time to wait for connection.
                Default is 120 seconds.
        
        """
        # I sometimes find that this returns an error, but it works when I call
        # it again. Trying a sleep to give process time to catch up.
        # sleep(5)
        n_seconds = int(max_wait.total_seconds())
        for i in range(n_seconds):
            try:
                r = requests.get(self._status_api, headers=self._headers)
                # got past the request call without raising, worked successfully
                break
            except ConnectionError:
                print(f"Waiting for connection -- total wait: {i} seconds")
                sleep(1)
            except Exception as e:
                raise e
        return_json = r.json()
        if r.status_code == 400 or 'version' not in return_json:
            raise RuntimeError(
                "Connected to Valhalla server but there was "
                f"another error.\n{return_json}")
        print("Successfully connected to Valhalla server.")

#region Helper functions
    def _ensure_custom_files_write_permissions(self):
        subprocess.run(
            f'docker exec -t {self.container_id} bash -c '
            f'sudo chmod -R ugo+w /custom_files', shell=True
        )

    def _test_docker(self):
        """ Raises RuntimeError if Docker cannot be run. """
        result = subprocess.run("docker", shell=True) 
        if not isinstance(result, subprocess.CompletedProcess):
            raise RuntimeError(NO_DOCKER_ERR_MSG)

    def _start_container(
            self, use_tiles_ignore_pbf, build_transit, 
            force_rebuild, server_threads) -> None:
        """ Starts a Docker container and builds network, if specified.
        
        Network elements such as OSM, GTFS and elevation data must be in place.
        Container ID is saved in self.container_id.       

        The prepared compose file is a simplified version of the 
        docker-compose.yml template presented in 
        https://github.com/gis-ops/docker-valhalla repository
        
        """
        # Build the compose file
        if not isinstance(server_threads, int):
            server_threads = max(cpu_count() - 1, 1)
            print(f'server_threads not defined, setting to {server_threads}')

        ct = VALHALLA_DOCKER_COMPOSE_TEMPLATE.copy()
        ct['services']['valhalla']['volumes'] = [
            f'{self._custom_files_path.as_posix()}:/custom_files']
        if build_transit:
            ct['services']['valhalla']['volumes'].append(
                f'{self._gtfs_dir.as_posix()}:/gtfs_feeds')
        ct['services']['valhalla']['environment'] = [
            f'use_tiles_ignore_pbf={use_tiles_ignore_pbf}',
            f'force_rebuild={force_rebuild}',
            f'build_transit={build_transit}',
            f'server_threads={server_threads}',
            # we don't want to download from here due to SSL issues
            # has to be done beforehand
            'build_elevation=False',    
            'serve_tiles=True',
            'build_tar=True',
            'traffic_name=traffic.tar',
            'build_admins=True',
            # curl failed downloading time zones file (again)
            'build_time_zones=False',
            # needs more testing, but maybe a download issue
            'use_default_speeds_config=False' 
        ]
        # save the compose file to the root in the custom_files path
        with open(self._compose_fp , 'w') as outfile:
            yaml.dump(ct, outfile, default_flow_style=False)
        
        # Start the docker container, then get the container id
        proc = subprocess.run(
            ['docker', 'compose', '-f', self._compose_fp.as_posix(), 'up', '-d'], 
            shell=True
        )
        if proc.returncode != 0:
             raise RuntimeError("Error starting docker container")
        self.container_id = subprocess.check_output(
            ["docker", "ps", "-lq"], shell=True).decode('UTF-8').strip()
        self._ensure_custom_files_write_permissions()
        self._test_valhalla_status(max_wait=timedelta(seconds=120))
        print("Successfully built network.")

    @staticmethod
    def _create_sources_and_targets_json_calls(origins, destinations):
        """ Convert origins/destinations from gpd.GeoDataFrames to pandas. """
        def create_json_call(gdf) -> list:
            out = []
            for _, geometry in gdf.items():
                out.append(
                    {
                        'lat': geometry.y,
                        'lon': geometry.x
                    }
                )
            return out  
        sources = create_json_call(origins) 
        targets = create_json_call(destinations)
        return sources, targets

    @property
    def custom_files_path(self):
        return self._custom_files_path
    
    @custom_files_path.setter
    def custom_files_path(self, new_custom_files_path):
        self._custom_files_path = Path(new_custom_files_path)
        self._custom_files_path.mkdir(exist_ok=True, parents=True)

        # Update paths within the custom_files volume mount
        self._compose_fp = self._custom_files_path / 'compose.yml'
        self._gtfs_dir = self._custom_files_path / 'gtfs_feeds'
        self._gtfs_dir.mkdir(exist_ok=True)
        self._elev_dir = self._custom_files_path / 'elevation_data'

    @property
    def request_host_url(self):
        return self._request_host_url

    @request_host_url.setter
    def request_host_url(self, new_request_host_url):
        self._request_host_url = new_request_host_url
        self._status_api = self._request_host_url + "/status"
        self._s2t_request_api = self._request_host_url + "/sources_to_targets"
        self._route_request_api = self._request_host_url + "/route"


def _create_transit_costing_options_call(
        speed_walking: Optional[float]=None, **kwargs) -> Dict | None:
    """ 
    Separate out the walking and transit cost options. I don't know
    yet how Valhalla handles empty dictionaries, so making sure
    to not create any tags if they are not needed.

    See docstring for compute_transit_traveltime_matrix for a desciption
    of the arguments.
    
    """
    transit_cost_options = {}
    walking_cost_options = {}
    if speed_walking:
        walking_cost_options['walking_speed'] = str(speed_walking)
    if kwargs:
        for k, v in kwargs.items():
            if k in ['use_bus', 'use_rail', 'use_transfers', 'filters']:
                transit_cost_options[k] = str(v)
            else:
                walking_cost_options[k] = str[v]
    if not walking_cost_options and not transit_cost_options:
        return None
    elif walking_cost_options and not transit_cost_options:
        return {'pedestrian': walking_cost_options}
    elif not walking_cost_options and transit_cost_options:
        return {'transit': transit_cost_options}
    else:
        return {
            'pedestrian': walking_cost_options,
            'transit': transit_cost_options
        }

def _process_valhalla_matrix_result(
        result: Dict, 
        origins: GeoSeries, 
        destinations: GeoSeries
    ) -> pd.Series:
    """ Parse the sources-to-target result coming from Valhalla. """ 
    try:
        s2t = result['sources_to_targets']
    except KeyError:
        raise RuntimeError(
            "Incorrect format from Valhalla's matrix request. "
            f"The Full return JSON is: \n{result}"
        )
    ttm = create_blank_ttmatrix(origins, destinations)
    origin_ids = origins.index
    destination_ids = destinations.index
    for i in range(len(origin_ids)):
        for j in range(len(destination_ids)):
            s2t_ij = s2t[i][j]
            ttm.loc[(
                origin_ids[s2t_ij['from_index']],
                destination_ids[s2t_ij['to_index']]
            )] = s2t_ij['time']
    return ttm.round(N_DECIMALS)