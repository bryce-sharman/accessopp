""" 
Module with functions to help:
    - read/write points from/to files
    - read matrices from/to files
    - create rectangular and hexagonal point arrays

"""

from geopandas import GeoSeries, points_from_xy
from math import pi, cos, sin, sqrt
import numpy as np 
from os import PathLike
import pandas as pd 
from pathlib import Path
from shapely import Point, Polygon
from typing import List, Optional, Tuple

from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN, ALL_COLUMNS

def read_points_from_csv(
        fp: PathLike, 
        id_colname: str, 
        x_colname: str, 
        y_colname: str, 
        other_columns: Optional[str | List[str]]=None,
        crs: Optional[str] = "EPSG:4326"
    ) -> Tuple[GeoSeries, Optional[pd.DataFrame]]:
    """ Reads a points csv file into geopandas GeoDataFrame. 

    Coordinates are currently assumed to be in lat/lon, EPSG:4326 (WGS84).
   
    Args:
        - fp: Path to points CSV file
        - id_colname: Name of the column containing the identifiable index.
        - x_colname: Name of the column containing the X coordinates (longitude). 
        - y_colname: Name of the column containing the Y coordinates (latitude).
        - other_columns: Optional column name[s] specifying additional columns 
            to import, such as weights.
        - crs: CRS string of input coordinates. Default is "EPSG:4326", which 
          corresponds to: WGS84 - World Geodetic System 1984, used in GPS.
    Returns:
        - Input points as geopandas GeoSeries.
        - If other_columns are specified, these columns are returned 
          separately in a Pandas DataFrame.

    """
    fp = Path(fp)
    if not fp.is_file():
        raise FileNotFoundError(f"File does not exist: {fp}")
    if isinstance(other_columns, str):
        other_columns = [other_columns]
    usecols = [id_colname, x_colname, y_colname]
    if other_columns:
        usecols.extend(other_columns)
    df = pd.read_csv(fp, usecols=usecols, index_col=id_colname)
    df.index.name = 'id'
    gs = GeoSeries(
        index=df.index, 
        data=points_from_xy(df[x_colname], df[y_colname]), 
        crs=crs
    )
    if other_columns:
        return gs, df.drop([x_colname, y_colname], axis=1)
    else:
        return gs


def build_rectangle_grid(
        lower_left: Point, 
        x_incr: float, 
        y_incr: float, 
        width: float, 
        height: float, 
        angle: float=0, 
        crs: Optional[str]= "EPSG:4326"
    ) -> Tuple[GeoSeries, GeoSeries]:
    """ Builds a rectangular grid of points and corresponding polygons.

    Args:
        - lower_left: x, y coordinates of lower-left corner of the grid.
        - x_incr: Grid point increment in the x direction, in metres
        - y_incr: Grid point increment in the y direction, in metres
        - width: Grid width in the x direction, in metres. 
        - height: Grid height in the y direction, in metres
        - angle: Angle in degrees of the x-coordinate of the grid with respect 
          to an east vector. Default is 0.
        - crs: CRS string of the projection system of the input lower_left
           point. Default is "EPSG:4326", which corresponds to: WGS84 - World  
          Geodetic System 1984, used in GPS.

    Returns: 
        - centroids: Point GeoSeries containing the centroid locations 
          of all squares in the grid
        - polygons: Polygon GeoSeries containing the geometry of each square.

    """
    ll, utm_proj_str = project_to_utm_coordinates(lower_left, crs)

    # Make a grid based on a 0, 0 lower corner
    # We'll transform it at the end using geoSeries affine transformation
    n_x = round(width / x_incr)
    n_y = round(height / y_incr)
    mi = pd.MultiIndex.from_product(
        [range(0, n_x), range(0, n_y)], names=['x', 'y'])
    gs_cntds = GeoSeries(index=mi, crs=utm_proj_str)
    gs_plgns = GeoSeries(index=mi, crs=utm_proj_str)

    for j in range(0, n_y):
        y_j_ll = j * y_incr
        y_j_centr = y_j_ll + y_incr / 2
        for i in range(0, n_x):
            x_i_ll = i * x_incr
            x_i_centr = x_i_ll + x_incr / 2
            # first the centroid
            gs_cntds.loc[(i, j)] = Point(x_i_centr, y_j_centr)
            # and now the polygons
            coords = ((x_i_ll, y_j_ll), 
                      (x_i_ll + x_incr, y_j_ll), 
                      (x_i_ll + x_incr, y_j_ll + y_incr), 
                      (x_i_ll, y_j_ll + y_incr )
                      )
            gs_plgns.loc[(i, j)] = Polygon((coords))

    # We can do an affine transformation on the series to 1) rotate, and 
    # 2) translate to lower bound convert angle to radians
    angle = angle * pi / 180.0
    gs_cntds.geometry = gs_cntds.geometry.affine_transform(
        [cos(angle), -sin(angle), sin(angle), cos(angle), ll.x, ll.y])
    gs_plgns.geometry = gs_plgns.geometry.affine_transform(
        [cos(angle), -sin(angle), sin(angle), cos(angle), ll.x, ll.y])
    
    # Convert back to original coordinate system
    gs_cntds = gs_cntds.to_crs(crs)
    gs_plgns = gs_plgns.to_crs(crs)
    # Reset to a single index
    return gs_cntds.reset_index(drop=True), gs_plgns.reset_index(drop=True)


def build_haxagonal_grid(
        lower_left: Point, 
        incr: float, 
        width: float, 
        height: float, 
        angle: float=0, 
        crs: Optional[str]= "EPSG:4326"
    )-> Tuple[GeoSeries, GeoSeries]:
    """Builds a hexagonal grid of points and corresponding polygons.

    Args:
        - lower_left: x, y coordinates of lower-left corner of the grid.
        - x_incr: Grid point increment in metres
        - width: Grid width in the x direction, in metres. 
        - height: Grid height in the y direction, in metres
        - angle: Angle in degrees of the x-coordinate of the grid with respect 
          to an east vector. Default is 0.
        - crs: CRS string of the projection system of the input lower_left
           point. Default is "EPSG:4326", which corresponds to: WGS84 - World  
          Geodetic System 1984, used in GPS.

    Returns: 
        - centroids: Point GeoSeries containing the centroid locations 
          of all squares in the grid
        - polygons: Polygon GeoSeries containing the geometry of each square.

    Notes
    -----
    Hexagons may appear to be flattened vertically even though the length of 
    each side is the same. This occurs due to projection lat/lon coordinate 
    system, as the distance covered by one degree of latitude is larger than 
    that covered by one degree of longitude when away from the equator.

    """
    ll, utm_proj_str = project_to_utm_coordinates(lower_left, crs)

    # Make the hex grid 
    sqrt_3 = sqrt(3)
    sqrt_3_inv = 1 / sqrt(3)   
    ri = 0.5 * incr                       
    ro = incr * sqrt_3_inv                 # Distance of origin to a vertex, also equals length of each side
    vert_offset = sqrt_3 * ri              # ro + 1/2 length of one side = 3 / sqrt(3) * r_i

    # Make a grid based on a 0, 0 lower corner
    # Then we'll transform it using geoSeries affine transformation
    n_x = round(width / incr)
    n_y = round(height / vert_offset)

    mi = pd.MultiIndex.from_product([range(0, n_x), range(0, n_y)])
    gs_cntds = GeoSeries(index=mi, crs=utm_proj_str)
    gs_plgns = GeoSeries(index=mi, crs=utm_proj_str)
    for j in range(0, n_y):
        if j % 2 == 0:
            # No offset to first row
            x_offset = 0.0
        else:
            # This is an offset row in the hex
            x_offset = ri

        y_bottom = j * vert_offset    # 0 for bottom row
        y_lower = y_bottom + ri * sqrt_3_inv
        y_middle = y_bottom + ro
        y_upper = y_lower + ro
        y_top = y_middle + ro

        for i in range(0, n_x):
            x_left = i * incr + x_offset
            x_middle = x_left + ri
            x_right = x_left + incr

            # first the centroid
            gs_cntds.loc[(i, j)] = Point(x_middle, y_middle)
            # and now the polygons
            coords = (
                (x_middle, y_bottom), 
                (x_right, y_lower), 
                (x_right, y_upper), 
                (x_middle, y_top), 
                (x_left, y_upper), 
                (x_left, y_lower)
            )
            gs_plgns.loc[(i, j)] = Polygon((coords))
    
    # We can do an affine transformation on the series to 1) rotate, and 
    # 2) translate to lower bound convert angle to radians
    angle = angle * pi / 180.0
    gs_cntds.geometry = gs_cntds.geometry.affine_transform(
        [cos(angle), -sin(angle), sin(angle), cos(angle), ll.x, ll.y])
    gs_plgns.geometry = gs_plgns.geometry.affine_transform(
        [cos(angle), -sin(angle), sin(angle), cos(angle), ll.x, ll.y])
    
    # Convert back to original coordinate system
    gs_cntds = gs_cntds.to_crs(crs)
    gs_plgns = gs_plgns.to_crs(crs)

    # Reset to a single index
    return gs_cntds.reset_index(drop=True), gs_plgns.reset_index(drop=True)


def read_matrix(fp: PathLike) -> pd.Series:
    """ Reads a previously calculated matrix from CSV file.

    Args:
        - fp: Filepath to file containing the matrix

    Returns:
        - Matrix in tall format.

    """
    matrix = pd.read_csv(
        fp, usecols=ALL_COLUMNS, index_col=INDEX_COLUMNS).squeeze()
    matrix.name = COST_COLUMN
    return matrix

def project_to_utm_coordinates(pt: Point, crs) -> Tuple[Point, str]:
    """ Convert point into UTM projection system. 
    
    Args:
        pt: Point to convert
        crs: crs projection system of point

    Returns:
        - x, y coordinates of point in UTM zone system.
        - UTM projection string
    Notes: 
        First ensuring that coordinates are in EGSG:4326 as this makes it
        easy to find the proper UTM zone.
    """

    ll_gs_init = GeoSeries([pt], crs=crs)
    ll_gs_4326 = ll_gs_init.to_crs("EPSG:4326")
    utm_proj_str = find_utm_proj_str(ll_gs_4326.iloc[0])
    ll_gs = ll_gs_init.to_crs(utm_proj_str)
    return  ll_gs.iloc[0], utm_proj_str

def find_utm_proj_str(pt: Point) -> str:
    """ 
    Find the suitable Universal Transverse Mercator zone for a given latitude, 
    longitude location. 

    Args:
        - pt: longitude, latitude of location to search for. Expected to
          have been converted to EPSG: 4326 before calling this function.

    Returns:
        - projection string

    Notes:
        For regions in the Northern Hemisphere between the equator and 84°N,  
        EPSG codes begin with the prefix 326 and are followed by a two-digit  
        number indicating the respective UTM zone.

        For regions in the Southern Hemisphere between the equator and 80°S, 
        EPSG codes begin with the prefix 327 and are followed by a two-digit 
        number indicating the respective UTM zone.

    """
    zone = int((pt.x  + 180.0) // 6) + 1
    if zone < 1 or zone > 60:
        raise ValueError("Invalid longitude.")
    if 0.0 <= pt.y < 84:
        return f"EPSG:326{zone:02d}"
    elif -80.0 < pt.y < 0:
        return f"EPSG:327{zone:02d}"
    else:
        raise ValueError("Invalid latitude.")
