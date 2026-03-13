""" Module to compute intrazonal costs. """

import geopandas as gpd
import numpy as np
import pandas as pd

def compute_intrazonal_costs_from_polygons(
        gs: gpd.GeoSeries,
        intrazonal_dist_expr: str='area**0.5 * 2 / 6',
        dist_to_cost_factor: float=1 / (4.0 * 1000 / 60)
    ) -> pd.Series:
    """ Compute intrazonal costs from a polygon and a cost function. 

    This function computes an approximate intrazonal cost for each zone based 
    on the area of the zone. This is useful for larger zones where the 
    cost of travel within the zone is not negligible.
    
    Args:
        gs: GeoSeries containing the polygons of the zones. 
            The index of the GeoSeries should be the zone id.

        intrazonal_dist_expr: A string expression to compute 
            intrazonal distance. Must be a valid Python expression
            that can calculate the intrazonal distance based on the
            area of the zone, which is available as the variable `area`. 

            The default expression is `area**0.5 * 2 / 6`, which is 
            consistent with the formulation used in the GTAv4 model.

        dist_to_cost_factor: A factor to convert distance to cost. 
            Note that the distance is calculated in the same units as the 
            coordinate system of the GeoSeries (often this is in metres).
            Default is 1 / (4.0 * 1000 / 60), which corresponds to a speed
            of 4 km/hr converted to minutes per metre.


    Returns:
        A Series containing the intrazonal costs for each zone. 
        The index of the Series is the zone id. 
    
    """

    if not gs.crs.is_projected:
        raise ValueError(
            "The GeoSeries must be in a projected coordinate system.")
    area = gs.area
    intrazn_dist = pd.eval(intrazonal_dist_expr)
    return intrazn_dist * dist_to_cost_factor


def insert_intrazonals_into_cost_matrix(
        c_ij: pd.Series,
        iz_costs: pd.Series
    ) -> pd.Series:
    """ Insert intra-zonal costs into cost matrix.

    The cost matrices produced by routing software often
    have a zero cost for intrazonal trips as the origin
    matches the destination. This function overwrites
    the intrazonal travel costs.

    Args:
        c_ij: Cost matrix as produced by travel time calculators.
        iz_costs: Intrazonal costs for each zone. The index of the Series is 
            the zone id. 

    Returns:
        Revised cost matrix with intrazonal costs included.
    """
    c_ij = c_ij.copy()
    # Only keep intrazonal costs where they are in the cost matrix
    c_ij_origins = pd.Index(
        np.sort(c_ij.index.get_level_values(0).unique())
    )
    c_ij_destinations = pd.Index(
        np.sort(c_ij.index.get_level_values(1).unique())
    )
    c_ij_hasiz = c_ij_origins.intersection(c_ij_destinations)
    iz_costs = iz_costs.loc[c_ij_hasiz].copy()
    # Now look through the intrazonal costs and inject into cost matrix
    for i, cost in iz_costs.items():
        c_ij.at[(i, i)] = cost 
    return c_ij