import datetime
import geopandas as gpd
import json
from multiprocessing import cpu_count
from os import PathLike
import pandas as pd
from pathlib import Path
import requests
from shapely import Point
import shutil
import subprocess
from typing import Dict, List
import yaml
import zipfile

from time import sleep

from ..enumerations import DEFAULT_SPEED_WALKING, DEFAULT_DEPARTURE_WINDOW
from accessto.utilities import test_od_input, test_file_existence
from accessto.utilities import test_dir_existence, empty_directory_recursive


NEW_BUILD_MSG = """
This method will create a Valhalla network in a new
Docker container. Before creation, the following will be
removed from the Docker volume mount before being replaced
with the updated versions:
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

    """
        
    def __init__(self, custom_files_path: str | PathLike):       
        self.custom_files_path = Path(custom_files_path)
        self.custom_files_path.mkdir(exist_ok=True, parents=True)

        # Internal objects
        self.container_id:str = None
        self._compose_fp = self.custom_files_path / 'compose.yml'
        self.gtfs_dir = self.custom_files_path / 'gtfs_feeds'
        self.gtfs_dir.mkdir(exist_ok=True)
        self.elev_dir = self.custom_files_path / 'elevation_data'
        
        # Ensure that Docker is installed and can be run
        self._test_docker()
        
        # These default URLs assume that the server is hosted locally. These can be changed by a user if desired
        self._request_host_url = "http://localhost:8002"
        self._status_api = self._request_host_url + "/status"
        self._request_api = self._request_host_url + "/sources_to_targets"
        self._headers = {
            'Content-Type': 'application/json'
        }

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
        # response = input(new_build_msg)
        # if response not in ('Y', 'y'):
        #     return

        # Ensure that osm_paths and gtfs_paths are list so we know their type
        osm_paths = osm_paths if isinstance(
            osm_paths, list) else [osm_paths]
        gtfs_paths = gtfs_paths if isinstance(
            gtfs_paths, list) else [gtfs_paths]

        # Remove OSM files from the custom_folders to avoid confusion
        print("Removing existing .osm.pbf  and GTFS files from custom files "
              "directory.")
        for osm_file in self.custom_files_path.glob("*.osm.pbf"):
            (self.custom_files_path / osm_file).unlink()
        empty_directory_recursive(self.gtfs_dir)

        # Prepare OSM files
        if osm_paths is None:
            use_tiles_ignore_pbf = True
            # Check that valhalla_tiles directory and valhalla_tiles.tar exist
            test_file_existence(self.custom_files_path / 'valhalla.tar',
                                NO_EXISTING_OSM_TILES_ERR_MSG)
            test_dir_existence(self.custom_files_path / 'valhalla_tiles',
                               NO_EXISTING_OSM_TILES_ERR_MSG)
        else:
            use_tiles_ignore_pbf = False
            print("Copying OSM files to custom_files directory.")
            for osm_path in osm_paths:
                osm_path = Path(osm_path)
                shutil.copyfile(
                    osm_path, self.custom_files_path / osm_path.name)

        if gtfs_paths is None:
            build_transit = False
        else:
            build_transit=True
            print("Extracting GTFS files to custom_files directory.")
            for gtfs_path in gtfs_paths:
                gtfs_path = Path(gtfs_path)
                with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
                    zip_ref.extractall(self.gtfs_dir / gtfs_path.name)      

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
        for osm_file in self.custom_files_path.glob("*.osm.pbf"):
            (self.custom_files_path / osm_file).unlink()
        empty_directory_recursive(self.gtfs_dir)

        print("Copying OSM files to custom_files directory.")
        use_tiles_ignore_pbf = False
        osm_paths = list(path.glob('*.osm.pbf'))
        for osm_path in osm_paths:
            osm_path = Path(osm_path)
            shutil.copyfile(
                osm_path, self.custom_files_path / osm_path.name)

        gtfs_paths = list(path.glob('*gtfs*.zip'))
        if gtfs_paths is None:
            build_transit = False
        else:
            build_transit=True
            print("Extracting GTFS files to custom_files directory.")
            for gtfs_path in gtfs_paths:
                gtfs_path = Path(gtfs_path)
                with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
                    zip_ref.extractall(self.gtfs_dir / gtfs_path.name)      

        self._start_container(
            use_tiles_ignore_pbf=use_tiles_ignore_pbf, 
            build_transit=build_transit, 
            force_rebuild=force_rebuild, 
            server_threads=server_threads
        )

    def compute_walk_traveltime_matrix(
            self, 
            origins: gpd.GeoDataFrame, 
            destinations: gpd.GeoDataFrame | None, 
            walking_cost_options: Dict | None
        ):
        """ Requests walk-only trip travel time matrix from Valhalla.
            Arguments
        ---------
        origins: geopandas.GeoDataFrame
            Origin points.  Has to have at least an ``id`` column and a geometry. 
        destinations: geopandas.GeoDataFrame or None, optional
            Destination points. If None, will use the origin points. Default is None
            If not None, has to have at least an ``id`` column and a geometry.
        walking_cost_options:
            Valhalla cost parameters, as defined in 
            https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#pedestrian-costing-options
            If `walking_speed` option is not defined then it will default
            to the accessto default, currently 5.0 km/hr.
            
        Notes
        -----
        Anticipated parameters expected to be of particular interest 
        for walk-only trips include:
            - walking_speed: Walking speed in kilometers per hour.
            - max_distance: Sets the maximum total walking distance of a route. 
            - use_hills: Range of values from 0 to 1, where 0 attempts to avoid 
                hills and steep grades even if it means a longer (time and
                distance) path, while 1 indicates the pedestrian does not fear 
                hills and steeper grades. Default is 0.5.

        """
        # Test origins and destination points JSON calls
        sources, targets = self._create_sources_and_targets_json_calls(
            origins, destinations)

        # set the costing options JSON calls
        if walking_cost_options is None:
            walking_cost_options = {}
        if 'walking_speed' not in walking_cost_options:
            walking_cost_options['walking_speed'] = str(DEFAULT_SPEED_WALKING)
        costing_options = {
            'pedestrian': walking_cost_options
        }

        # Note that because it's using a web protocol that I can call the 
        # sources_to_targets request from the local computer even though it's being
        # hosted on the Docker container. 
        full_call = {
            'sources': sources,
            'targets': targets,
            'costing': 'pedestrian',
            'costing_options': costing_options
        }
        print(full_call)
        r = requests.get(
            self._request_api, headers=self._headers, json=full_call)
        return self._process_valhalla_matrix_result(
            r.json(), origins, destinations)


    def compute_bicycle_traveltime_matrix(
            self, 
            origins: gpd.GeoDataFrame, 
            destinations: gpd.GeoDataFrame | None, 
            bicycle_cost_options: Dict
        ):
        """ Requests bike-only trip travel time matrix from Valhalla.
            
        Arguments
        ---------
        origins: geopandas.GeoDataFrame
            Origin points.  Has to have at least an ``id`` column and a geometry. 
        destinations: geopandas.GeoDataFrame or None, optional
            Destination points. If None, will use the origin points. Default is None
            If not None, has to have at least an ``id`` column and a geometry.
        bicycle_cost_options:
            Valhalla cost parameters, as defined in 
            https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#bicycle-costing-options
            If `walking_speed` option is not defined then it will default
            to the accessto default, currently 5.0 km/hr.
            
        Notes
        -----
        Anticipated parameters expected to be of particular interest 
        for bicycle-only trips include:
            - bicycle_type: one of Road, Hybrid, City, Cross, Mountain
                sets the default cycling speed
            - cycling_speed: Cycling speed in kilometers per hour.
                Will default to that defined by bicycle_type if not provided.
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
        """
        # Test origins and destination points JSON calls
        sources, targets = self._create_sources_and_targets_json_calls(
            origins, destinations)

        # Note that there is no default cycling speed (yet) in accessto
        # so must be defined in the call.
        full_call = {
            'sources': sources,
            'targets': targets,
            'costing': 'bicycle',
            'costing_options': {
                'bicycle': bicycle_cost_options
            }
        }
        print(full_call)
        r = requests.get(
            self._request_api, headers=self._headers, json=full_call)
        print(r)
        return self._process_valhalla_matrix_result(
            r.json(), origins, destinations)


    @staticmethod
    def _process_valhalla_matrix_result(
            result: Dict, 
            origins: gpd.GeoDataFrame, 
            destinations: gpd.GeoDataFrame
        ) -> pd.DataFrame:
        """ Parse the sources-to-target result coming from Valhalla. """ 
        # we only care about the matrix results portion
        s2t = result['sources_to_targets']

        # setup a dataframe to hold the results
        # Valhalla appears to output results by origin, then by destination, 
        # hence the following index should match its output order.
        ttm_index = pd.MultiIndex.from_product(
            [origins['id'], destinations['id']], names=['from_id', 'to_id'])
        ttm = pd.Series(index=ttm_index, name='travel_time')

        if len(origins) != len(s2t):
            raise RuntimeError(
                "Unexpected number of sources_to_targets, look into this.")
        # copy the data over to the dataframe, one by one
        # it first loops over the origins
        i = 0
        for origin_results in s2t:
            # then the results loop over the destinations
            for od_result in origin_results:
                ttm.iloc[i] = od_result['time'] / 60.0
                i += 1

        # return this in the r5py form, which we use throughout accessto
        return ttm.reset_index()

    def test_valhalla_status(self):
        """ Test that we can access Valhalla service from local computer."""
        # I sometimes find that this returns an error, but it works when I call
        # it again. Trying a sleep to give process time to catch up.
        # sleep(5)
        try:
            r = requests.get(
                self._status_api, headers=self._headers)
        except Exception as e:
            print("Could not connect to network. Check in the Docker "
                  "logs if the build is complete")
            raise e
        return_json = r.json()
        if r.status_code == 400 or 'version' not in return_json:
            raise RuntimeError(
                "Connected to Valhalla server but there was "
                f"another error.\n{return_json}")
        print("Successfully connected to Valhalla server.")

#region Helper functions
    def _test_docker(self):
        """ Raises RuntimeError if Docker cannot be run. """
        result = subprocess.run("docker", shell=True) 
        if not isinstance(result, subprocess.CompletedProcess):
            raise RuntimeError(NO_DOCKER_ERR_MSG)

    def _ensure_custom_files_write_permissions(self):
        subprocess.run(
            f'docker exec -t {self.container_id} bash -c '
            f'sudo chmod -R ugo+w /custom_files', shell=True
        )

    def _retrieve_valhalla_result_json(self, result):
        result = result.stdout.decode('UTF-8')
        first_bracket = result.find('{')
        result = result[first_bracket:]
        return json.loads(result)

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
            f'{self.custom_files_path.as_posix()}:/custom_files']
        if build_transit:
            ct['services']['valhalla']['volumes'].append(
                f'{self.gtfs_dir.as_posix()}:/gtfs_feeds')
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
        self.test_valhalla_status()
        print("Successfully built network.")

    @staticmethod
    def _create_sources_and_targets_json_calls(origins, destinations):
        """ 
        Convert origins/destinations from gpd.GeoDataFrames to pandas
        """
        def _create_sources_or_targets_json_call(gdf) -> list:
            """ Create list of locations based on input GeoDataFrame """
            out = []
            for _, row in gdf.iterrows():
                geometry = row['geometry']
                out.append(
                    {
                        'lat': geometry.y,
                        'lon': geometry.x
                    }
                )
            return out

        test_od_input(origins)
        sources = _create_sources_or_targets_json_call(origins)
        if destinations is not None:
           test_od_input(destinations)    
           targets = _create_sources_or_targets_json_call(destinations)
        else:
            targets = sources
        return sources, targets
