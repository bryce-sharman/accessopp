import datetime
import geopandas as gpd
from glob import glob
import json
from multiprocessing import cpu_count
from os import PathLike
import pandas as pd
from pathlib import Path
from shapely import Point
import shutil
import subprocess
from typing import List
import yaml
import zipfile

from time import sleep

from ..enumerations import DEFAULT_SPEED_WALKING, DEFAULT_DEPARTURE_WINDOW
from accessto.utilities import test_od_input


new_build_msg = """
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

VALHALLA_DOCKER_COMPOSE_TEMPLATE = {
    'services': {
        'valhalla': {
            'image': 'ghcr.io/gis-ops/docker-valhalla/valhalla:latest',
            'container_name': 'valhalla_latest',
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
        
    def __init__(self, custom_files_path: str | Path):       
        self.custom_files_path = Path(custom_files_path)
        self.custom_files_path.mkdir(exist_ok=True, parents=True)

        # Internal objects
        self.container_id:str = None
        self._compose_fp = self.custom_files_path / 'compose.yml'
        self.gtfs_dir = self.custom_files_path / 'gtfs_feeds'
        self.gtfs_dir.mkdir(exist_ok=True)
        self.elev_dir = self.custom_files_path / 'elevation_data'
        
        # Ensure that Docker is installed
        test_docker()

    def build_network(
            self, 
            osm_paths: PathLike | list[PathLike] | None = None,
            gtfs_paths: PathLike | list[PathLike] | None = None,
            * , 
            build_elevation: bool=True,
            force_rebuild=False,
            server_threads: int | None = None
        ) -> None: 
        """ Build a transport network from specified network files.

        Starts a Docker container including built network that is ready
        for subsequent Valhalla routing commands. Container ID is saved in 
        self.container_id.

        Arguments
        ---------
        osm_paths: 
            Absolute path files to OSM networks, stored in binary (.pbf) format.
            Can be a single path or a list of multiple paths.
            If None then will start a container using previously built network.
            Default is None.
        gtfs_paths: 
            Path to  public transport schedule information in GTFS format.
            Can be a single or list of GTFS files in zipped format. 
            If None then transit network will not be built.
            Default is None.
        build_elevation:
            Flag to include elevation data in the network build
            If True, then elevation data must be placed in the 'elevation_data'
            subfolder of the custom_files directory.
            Default is True.
        force_rebuild:
            If True, then forces a bebuild of routing tiles. Default is False.
        server_threads: 
            Number of threads to use by Docker.
            If None, Valhalla uses the threads in the computer.       
            Default is None.

        """
        response = input(new_build_msg)
        if response not in ('Y', 'y'):
            return

        # Prepare OSM files
        if osm_paths is None:
            # Check that valhalla_tiles directory and valhalla_tiles.tar exist
            test_file_existence(
                self.custom_files_path / 'valhalla.tar', 
                'OSM fle not defined, and no existing tiles have been built.')
            test_dir_existence(
                self.custom_files_path / 'valhalla_tiles',
                'OSM fle not defined, and no existing tiles have been built.')
            # Remove OSM files from the custom_folders to avoid confusion
            for osm_file in self.custom_files_path.glob("*.osm.pbf"):
                (self.custom_files_path / osm_file).unlink()
            use_tiles_ignore_pbf = True
        else:
            # Remove OSM files from the custom_folders
            for osm_file in self.custom_files_path.glob("*.osm.pbf"):
                (self.custom_files_path / osm_file).unlink()
            # Copy OSM files over
            if isinstance(osm_paths, list):
                for osm_path in osm_paths:
                    osm_path = Path(osm_path)
                    print(self.custom_files_path.as_posix())
                    shutil.copyfile(osm_path, self.custom_files_path)
            else:
                osm_path = Path(osm_paths)
                shutil.copyfile(
                    osm_path, self.custom_files_path / osm_path.name)
            use_tiles_ignore_pbf = False

        if gtfs_paths is None:
            build_transit = False
            # remove all subdirectories in GTFS folder to remove confusion
            empty_directory_recursive(self.gtfs_dir)
        else:
            build_transit=True
            # remove all subdirectories in GTFS folder to start with clear slate
            empty_directory_recursive(self.gtfs_dir)
            # copy unzipped GTFS file to subfolders
            if isinstance(osm_paths, list):
                for gtfs_path in gtfs_paths:
                    gtfs_path = Path(gtfs_path)
                    with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
                        zip_ref.extractall(self.gtfs_dir / gtfs_path.name)    
            else:
                gtfs_path = Path(gtfs_paths)
                with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
                    zip_ref.extractall(self.gtfs_dir / gtfs_path.stem)     

        if build_elevation:
            test_dir_existence(
                self.elev_dir, 
                "build_elevation requires elevation data to be inplace in "
                "custom_files/elevation_data directory.")

        self._start_container(
            use_tiles_ignore_pbf=use_tiles_ignore_pbf, 
            build_elevation=build_elevation, 
            build_transit=build_transit, 
            force_rebuild=force_rebuild, 
            server_threads=server_threads
        )


    def build_network_from_dir(
            self, 
            path: PathLike,
            * , 
            build_elevation: bool=True,
            force_rebuild: bool=False,
            server_threads: int | None = None) -> None:
        """ Builds transport network from directory containing OSM and GTFS data.

        Starts a Docker container including built network that is ready
        for subsequent Valhalla routing commands. Container ID is saved in 
        self.container_id.

        Arguments
        ---------
        path: Path to directory containing OSM and, optionally, GTFS files
        build_elevation:
            Flag to include elevation data in the network build
            If True, then elevation data must be placed in the 'elevation_data'
            subfolder of the custom_files directory.
            Default is True.
        force_rebuild:
            If True, then forces a bebuild of routing tiles. Default is False.
        server_threads: 
            Number of threads to use by Docker.
            If None, Valhalla uses the threads in the computer.       
            Default is None.

        """
        path = Path(path)
        test_dir_existence(path, f'Directory not found: {path.as_posix()}')

        osm_files = list(path.glob('*.osm.pbf'))
        gtfs_files = list(path.glob('*gtfs*.zip', case_sensitive=None))

        if len(osm_files) == 0:
            print("No OSM files exist in specified directory, "
                  "trying to start with pre-built network.")
            self.build_network(
                osm_paths=None,
                gtfs_paths=None,
                build_elevation=False,
                force_rebuild=False
            )
        else:
            if len(gtfs_files) == 0:
                gtfs_files = None
            self.build_network(
                osm_paths=osm_files,
                gtfs_paths=gtfs_files,
                build_elevation=build_elevation,
                force_rebuild=force_rebuild,
                server_threads=server_threads
            )
       
    def compute_walk_traveltime_matrix(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame | None=None, 
        walking_cost_options=None
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
            https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/#costing-options
            Anticipated parameters expected to be of particular interest include:
            - walking_speed: float
              Walking speed in kilometers per hour. Must be between 0.5 and 25.
            - max_distance: float
              Sets the maximum total walking distance of a route. 
            - use_hills: float
              Range of values from 0 to 1, where 0 attempts to avoid 
              hills and steep grades even if it means a longer (time and
              distance) path, while 1 indicates the pedestrian does not fear 
              hills and steeper grades. Default is 0.5.
            - use_lit:  float
              Range of values from 0 to 1, where 0 indicates indifference 
              towards lit streets, and 1 indicates that unlit streets should be 
              avoided. The default value is 0.
              (Testing for shift-work equity analyses)

        """
        if walking_cost_options is None:
            walking_cost_options = {}
        # Set default walking speed, if required
        if 'walking_speed' not in walking_cost_options:
            walking_cost_options['walking_speed'] = str(DEFAULT_SPEED_WALKING)
        # Test origins and destination points
        test_od_input(origins)
        if destinations is not None:
           test_od_input(destinations)    

        sources = create_sources_or_targets(origins)
        targets = create_sources_or_targets(destinations)
        costing_options = {
            'pedestrian': walking_cost_options
        }

        full_call = {
            'sources': sources,
            'targets': targets,
            'costing': 'pedestrian',
            'costing_options': costing_options
        }
        # The valhalla reader appears to be sensitive to spacing,
        # trying to put everything on one line instead of a pretty printing
        full_call_str = json.dumps(full_call)
        with open(self.custom_files_path / "run_routing.sh", 'w', 
                  encoding='utf-8') as f:
            f.write(full_call_str + '\n')
        sleep(1)
        result = subprocess.check_output([
                'docker', 
                'exec', 
                '-t', 
                self._container_id, 
                'bash', 
                '-c', 
                'valhalla_service',
                'valhalla.json',
                'sources_to_targets',
                 './run_routing.sh'
            ], shell=True
        )
        print(result)
        return self._process_valhalla_matrix_result(result, origins, destinations)




    def _start_container(
            self, 
            use_tiles_ignore_pbf, 
            build_elevation, 
            build_transit, 
            force_rebuild, 
            server_threads
        ) -> None:
        """ Starts a Docker container including built network.
        
        Network elements such as OSM, GTFS and elevation data must be in place.
        Container ID is saved in self.container_id.       
        
        """
        self._create_docker_compose_file(
            use_tiles_ignore_pbf=use_tiles_ignore_pbf, 
            build_elevation=build_elevation, 
            build_transit=build_transit, 
            force_rebuild=force_rebuild, 
            server_threads=server_threads
        )
        
        # Start the docker container, then get the container id
        proc = subprocess.run(
            ['docker', 'compose', '-f', self._compose_fp.as_posix(), 'up', '-d'], 
            shell=True
        )
        if proc.returncode != 0:
             raise RuntimeError("Error starting docker container")
        self.container_id = subprocess.check_output(
            ["docker", "ps", "-lq"], shell=True).decode('UTF-8').strip()
 
        # Set write and read permissions to the custom_files mount 
        # This is mainly so that I can write the file containing the 
        # valhalla_service routing call definition scripts. 
        subprocess.run(f'docker exec -t {self.container_id} bash -c '
                        '"sudo chmod a+rw /custom_files"', shell=True)


    def _create_docker_compose_file(
            self, 
            use_tiles_ignore_pbf, 
            build_elevation, 
            build_transit, 
            force_rebuild, 
            server_threads
        ):
        """ Creates a Docker compose file to set container running settings.
        
        This compose file is a simplified version of the docker-compose.yml 
        template presented in 
        # https://github.com/gis-ops/docker-valhalla repository, modified
        for the expected use case.
            - always want to build elevation
            - we have pre-prepared OSM network files and elevation data
            - will be using the 'one-shot' valhalla_service command to call the
              calculation for routes, hence we don't need the port as we won't 
              be making calls through the Restful API.
            
        """
        ct = VALHALLA_DOCKER_COMPOSE_TEMPLATE.copy()
        # add the volumes
        ct['services']['valhalla']['volumes'] = [
            f'{self.custom_files_path.as_posix()}:/custom_files']
        if build_transit:
            ct['services']['valhalla']['volumes'].append(
                f'{self.gtfs_dir.as_posix()}:/gtfs_feeds')
        if build_elevation is not None:
            ct['services']['valhalla']['volumes'].append(
                f'{self.elev_dir.as_posix()}:/custom_files/elevation_data')

        # add the environment variables
        # This set of variables is fixed:
        # we want to use pre-prepared OSM files
        ct['services']['valhalla']['environment']= ['tile_urls=False']
        # always build this ... why not
        ct['services']['valhalla']['environment'].append('build_tar=True')   
        # I don't know if we really need this ... start with True
        ct['services']['valhalla']['environment'].append('build_admins=True')     
        # we'll need this for turn penalties and transit routing
        ct['services']['valhalla']['environment'].append('build_time_zones=True') 
        # keep container open to run following routing commands
        ct['services']['valhalla']['environment'].append('serve_tiles=True')  
        # don't build a traffic archive as we're not (yet) using for traffic
        ct['services']['valhalla']['environment'].append('traffic_name=""')      
        # I don't know what to set this to, starting with the default, False
        ct['services']['valhalla']['environment'].append(
            'update_existing_config=False')   
        # Likely only need this for traffic routing
        ct['services']['valhalla']['environment'].append(
            'use_default_speeds_config=False')  
        # This set of environment variables is from class instantiation
        ct['services']['valhalla']['environment'].append(
            f'use_tiles_ignore_pbf={use_tiles_ignore_pbf}')
        ct['services']['valhalla']['environment'].append(
            f'force_rebuild={force_rebuild}')
        ct['services']['valhalla']['environment'].append(
            f'build_elevation={build_elevation}')
        ct['services']['valhalla']['environment'].append(
            f'build_transit={build_transit}')
        if isinstance(server_threads, int):
            ct['services']['valhalla']['environment'].append(
                f'server_threads={server_threads}')
        else:
            ct['services']['valhalla']['environment'].append(
                f'server_threads={max(cpu_count() - 1, 1)}')

        # save the compose file to the root in the custom_files path
        print('docker compose file in dictionary form')
        print(ct)
        with open(self._compose_fp , 'w') as outfile:
            yaml.dump(ct, outfile, default_flow_style=False)


#region Helper functions
# These functions could be static methods, but I'm inluding at the 
# module level.


def process_valhalla_matrix_result(
    result: bytes, 
    origins: gpd.GeoDataFrame, 
    destinations: gpd.GeoDataFrame
) -> pd.DataFrame:
    """ Parse the sources-to-target result coming from Valhalla """ 

    # The result string seems to start with information and then turns to a 
    # dictionary later on. We'll look for the dictionary by finding the 
    # first instance of '{'
    result = result.decode('UTF-8')
    first_bracket = result.find('{')
    result = result[first_bracket:]
    # this next line parses out the return dictionary, which is in JSON format
    result_json = json.loads(result)  
    print(result_json)
    # we only care about the matrix results portion
    sources_to_targets = result_json['sources_to_targets']

    # setup a dataframe to hold the results
    # Valhalla appears to output results by origin, then by destination, 
    # hence the following index should match its output order.
    ttm_index = pd.MultiIndex.from_product(
        [origins['id'], destinations['id']], names=['from_id', 'to_id'])
    ttm = pd.Series(index=ttm_index, name='travel_time')

    if len(origins) != len(sources_to_targets):
        raise RuntimeError(
            "Unexpected number of sources_to_targets, look into this.")
    # copy the data over to the dataframe, one by one
    # it first loops over the origins
    i = 0
    for origin_results in sources_to_targets:
        # then the results loop over the destinations
        for od_result in origin_results:
            ttm.iloc[i] = od_result['time'] / 60.0
            i += 1
            print(od_result)

    # we only really want the time, so just return this in the r5py form
    return ttm.reset_index()


def test_docker():
    """ Test that docker can be run. """
    result = subprocess.run("docker", shell=True) 
    if not isinstance(result, subprocess.CompletedProcess):
        raise RuntimeError(
            "Error running Docker. Docker Desktop must be installed "
            "and the Docker Engine started to run Valhalla routing.")


def create_sources_or_targets(gdf) -> list:
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


def test_file_existence(path, msg_on_fail):
    """ Test if a file exists. """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(msg_on_fail)


def test_dir_existence(path, msg_on_fail):
    """ Test if a directory exists. """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(msg_on_fail)


def empty_directory_recursive(dir_path):
    """ Remove all files in a directory, including all subdirectories. """
    for path in Path(dir_path).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
