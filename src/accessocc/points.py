""" Module with functions to help read/write points from file, or create an array of points. """
import geopandas as gpd
from math import ceil, pi, cos, sin, sqrt
import numpy as np 
import pandas as pd 
from pathlib import Path
from shapely import Point, Polygon


def read_points_from_csv(fp, id_colname, x_colname, y_colname, other_columns=None):
    """ Reads a points csv file into geopandas GeoDataFrame. 

    Coordinates are currently assumed to be in lat/lon, EPSG:4326 (WGS84).
   
    Parameters
    ----------
        fp:   pathlib.Path or str
            Path to points CSV file
        id_colname: str
            Name of the column containing the identifiable index.
        x_colname: str
            Name of the column containing the X coordinates (longitude). 
        y_colname: 
            Name of the column containing the Y coordinates (latitude).
        other_columns: str or list[str] or None, optional
            Optional column name specifying additional columns to also import.
            These will likely be weights.
        
    Returns
    -------
        gpd.GeoDataFrame
            Input points as geopandas GeoDataFrame
    """
    fp = Path(fp)
    if not fp.is_file():
        raise FileNotFoundError(f"File does not exist: {fp}")

    if isinstance(other_columns, str):
        other_columns = [other_columns]

    usecols = [id_colname, x_colname, y_colname]
    if other_columns is not None:
        usecols.extend(other_columns)
    df = pd.read_csv(fp, usecols=usecols, index_col=False)

    # The point identifier is expected to be in a column called 'id', rename this now
    if id_colname != 'id':
        df = df.rename({id_colname: 'id'}, axis=1)

    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_colname], df[y_colname]), crs="EPSG:4326")
    gdf = gdf.drop([x_colname, y_colname], axis=1)
    return gdf


def find_utm_proj_str(pt):
    """ Find the suitable Universal Transverse Mercator zone for a given latitude, longitude location. 
    
        For regions in the Northern Hemisphere between the equator and 84°N, EPSG codes begin with the prefix 326, 
        followed by a two-digit number indicating the respective UTM zone.

        For regions in the Southern Hemisphere between the equator and 80°S, EPSG codes begin with the prefix 327, 
        followed by a two-digit number indicating the respective UTM zone.

        Arguments
        ---------
        pt: shapely.pt
            longitude, latitude of location to search for.

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



def build_rectangle_grid(lower_left, x_incr, y_incr, width, height, angle=0, crs= "EPSG:4326"):
    """ Builds a grid of points, and corresponding polygons, in a region.

    Parameters
    ----------
    lower_left: shapely.Point
        x, y coordinates of lower-left corner of the bounding area.
    x_incr: float
        Grid point increment in the x direction, in meters
    y_incr: float
        Grid point increment in the y direction, in meters
    width: float
        Grid width in the x direction, in meters. 
    height: float
        Grid height in the y direction, in meters
    angle: float, optional
        Angle iun degrees of the x-coordinate of the grid with respect to an east vector. Default is 0.
    crs: str
        CRS string of the projection system of the input lower_left point.
        Default is "EPSG:4326", which corresponds to: WGS84 - World Geodetic System 1984, used in GPS

    Returns
    -------
    centroids: geopandas.GeoDataFrame
        Point GeoDataFrame containing the centroid locations of all squares in the grid
    polygons: geopandas.GeoDataFrame
        Polygon GeoDataFrame containing the geometry of each square.

    """
    # Find the latitude/longitude coordinates, so that we can identify the proper UTM projection system
    # that we'll use to make the grid
    ll_gs_init = gpd.GeoSeries([lower_left], crs=crs)
    ll_gs_4326 = ll_gs_init.to_crs("EPSG:4326")
    utm_proj_str = find_utm_proj_str(ll_gs_4326.iloc[0])

    # Convert lat-lon coordinates to projected coordinate system
    # It's easiest to use geopandas for this purpose
    ll_gs = ll_gs_init.to_crs(utm_proj_str)
    ll_x = ll_gs.iloc[0].x
    ll_y = ll_gs.iloc[0].y

    # Make a grid based on a 0, 0 lower corner
    # Then we'll transform it using geoSeries affine transformation
    n_x = round(width / x_incr)
    n_y = round(height / y_incr)
    
    mi = pd.MultiIndex.from_product([range(0, n_x), range(0, n_y)])
    gs_cntds = gpd.GeoDataFrame(index=mi, columns=['id', 'geometry'], crs=utm_proj_str)
    gs_plgns = gpd.GeoDataFrame(index=mi, columns=['id', 'geometry'], crs=utm_proj_str)

    cell_id = 1
    for j in range(0, n_y):
        y_j_ll = j * y_incr
        y_j_centr = y_j_ll + y_incr / 2
        for i in range(0, n_x):
            x_i_ll = i * x_incr
            x_i_centr = x_i_ll + x_incr / 2
            # first the centroid
            gs_cntds.loc[(i, j), ['id', 'geometry']] = cell_id, Point(x_i_centr, y_j_centr)
            # and now the polygons
            coords = ((x_i_ll, y_j_ll), (x_i_ll + x_incr, y_j_ll), (x_i_ll + x_incr, y_j_ll + y_incr), (x_i_ll, y_j_ll + y_incr ))
            gs_plgns.loc[(i, j), ['id', 'geometry']] = cell_id, Polygon((coords))
            cell_id += 1

    # We can do an affine transformation on the series to 1) rotate, and 2) translate to lower bound
    # convert angle to radians
    angle = angle * pi / 180.0
    gs_cntds.geometry = gs_cntds.geometry.affine_transform([cos(angle), -sin(angle), sin(angle), cos(angle), ll_x, ll_y])
    gs_plgns.geometry = gs_plgns.geometry.affine_transform([cos(angle), -sin(angle), sin(angle), cos(angle), ll_x, ll_y])
    
    # Convert back to original coordinate system
    gs_cntds = gs_cntds.to_crs(crs)
    gs_plgns = gs_plgns.to_crs(crs)
    return gs_cntds, gs_plgns


def build_haxagonal_grid(lower_left, incr, width, height, angle=0, crs= "EPSG:4326"):
    """ Builds a grid of points, and corresponding polygons, in a region.

    Parameters
    ----------
    lower_left: shapely.Point
        x, y coordinates of lower-left corner of the bounding area. This point lies just below and 
        to the left of the lower-left cell
    incr: float
        Distance between two cells in the x-direction, in metres. Due to offset, the difference in the
        y direction is lower 
    width: float
        Grid width in the x direction, in meters. 
    height: float
        Grid height in the y direction, in meters
    n_x: float
        Grid point increment in the x direction, in meters
    n_y: float
        Grid point increment in the y direction, in meters
    angle: float, optional
        Angle iun degrees of the x-coordinate of the grid with respect to an east vector. Default is 0.
    crs: str
        CRS string of the projection system of the input lower_left point.
        Default is "EPSG:4326", which corresponds to: WGS84 - World Geodetic System 1984, used in GPS

    Returns
    -------
    centroids: geopandas.GeoDataFrame
        Point GeoDataFrame containing the centroid locations of all squares in the grid
    polygons: geopandas.GeoDataFrame
        Polygon GeoDataFrame containing the geometry of each square.

    Notes
    -----
    Hexagons may appear to be flattened vertically even though the length of each side is the same.
    This occurs due to projection lat/lon coordinate system, as the distance covered by one degree
    of latitude is larger than that covered by one degree of longitude when away from the equator.

    """

    # Find the latitude/longitude coordinates, so that we can identify the proper UTM projection system
    # that we'll use to make the grid
    ll_gs_init = gpd.GeoSeries([lower_left], crs=crs)
    ll_gs_4326 = ll_gs_init.to_crs("EPSG:4326")
    utm_proj_str = find_utm_proj_str(ll_gs_4326.iloc[0])

    # Convert lat-lon coordinates to projected coordinate system
    # It's easiest to use geopandas for this purpose
    ll_gs = ll_gs_init.to_crs(utm_proj_str)
    ll_x = ll_gs.iloc[0].x
    ll_y = ll_gs.iloc[0].y

    # Make the hex grid 
    sqrt_3 = sqrt(3)
    sqrt_3_inv = 1 / sqrt(3)   
    
    ri = 0.5 * incr                       
    ro = incr * sqrt_3_inv                 # Distance of origin to a vertex, also equals length of each side
    half_ro = ri * sqrt_3_inv              # Length of 1/2 of each side
    vert_offset = sqrt_3 * ri              # ro + 1/2 length of one side = 3 / sqrt(3) * r_i

    # Make a grid based on a 0, 0 lower corner
    # Then we'll transform it using geoSeries affine transformation
    n_x = round(width / incr)
    n_y = round(height / vert_offset)

    mi = pd.MultiIndex.from_product([range(0, n_x), range(0, n_y)])
    gs_cntds = gpd.GeoDataFrame(index=mi, columns=['id', 'geometry'], crs=utm_proj_str)
    gs_plgns = gpd.GeoDataFrame(index=mi, columns=['id', 'geometry'], crs=utm_proj_str)
    cell_id = 1
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
            gs_cntds.loc[(i, j), ['id', 'geometry']] = cell_id, Point(x_middle, y_middle)
            # and now the polygons
            coords = ((x_middle, y_bottom), (x_right, y_lower), (x_right, y_upper), (x_middle, y_top), (x_left, y_upper), (x_left, y_lower))
            gs_plgns.loc[(i, j), ['id', 'geometry']] = cell_id, Polygon((coords))
            cell_id += 1
    
    # We can do an affine transformation on the series to 1) rotate, and 2) translate to lower bound
    # convert angle to radians
    angle = angle * pi / 180.0
    gs_cntds.geometry = gs_cntds.geometry.affine_transform([cos(angle), -sin(angle), sin(angle), cos(angle), ll_x, ll_y])
    gs_plgns.geometry = gs_plgns.geometry.affine_transform([cos(angle), -sin(angle), sin(angle), cos(angle), ll_x, ll_y])
    
    # Convert back to original coordinate system
    gs_cntds = gs_cntds.to_crs(crs)
    gs_plgns = gs_plgns.to_crs(crs)

    return gs_cntds, gs_plgns




# def to_csv():
#     pass


