""" Calibration module. """

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from typing import Callable, Tuple

from accessopp.utilities import weighted_median

ERR_MSG_INV_MATRIX = \
    "pandas.Series with the observed trips (for the travel segment) between " \
    "zones. This series is expected to have two index levels. The first is " \
    "the origin zone while the second is the destination zone."
ERR_MSG_INV_VCT = \
    "pandas.Series with the observed productions or attractions (for the "\
    "travel segment). This series is expected to have a single index " \
    "corresponding to the zone."
ERR_MSG_INV_FNC = \
    "Currently, this function can only calibrate 'neg_exponential', " \
    "'power' and 'gamma' functions."
ERR_MSG_INV_METHOD = \
    "method argument must be either 'median' or 'trip_list'."
ERR_MSG_INV_MEDIAN = \
    "When using method='median', median argument must be a float > 0"
WARN_ZEROTT_INTRAZN = \
    "Zero travel time found on intrazonal trip for zone %s."
WARN_ZEROTT_INTERZN = \
    "Zero travel time found for zone pair (%s-%s)."


def neg_exponential(beta: float, cost: ArrayLike) -> np.array:
    return np.exp(beta * cost)


def power(alpha: float, cost: ArrayLike) -> np.array:
    return np.pow(cost, alpha)


def gaussian(sigma: float, cost: ArrayLike) -> np.array:
    return np.exp(-cost*cost / (2 * sigma * sigma))


def _calculate_median_tripcost(
        t_ij: pd.Series, 
        c_ij: pd.Series, 
        cost_fillvalue: float
    ) -> float:
    """ 
    Calculate the weighted median given number of trips between zones (t_ij)
    and cost matrix (c_ij).

    Args:
        t_ij: Observed trips for desired travel segment from zone i to zone j. 
        c_ij: Cost matrix
        cost_fillvalue: Cost to apply to blank cells in the cost matrix.

    Returns:
        Median travel time.
    """
    # Create a temporary copy of the trips Series and merge in the cost matrix
    t2_ij = pd.DataFrame(t_ij)
    t2_ij.index.names = c_ij.index.names
    t2_ij = t2_ij.merge(c_ij, how='inner', left_index=True, right_index=True)
    cost_col = '__cost__'
    weight_col = '__ntrips__'
    t2_ij.columns = [weight_col, cost_col]
    # Fill in blanks in the cost matrix
    t2_ij[cost_col] = t2_ij[cost_col].fillna(cost_fillvalue)
    return weighted_median(t2_ij, cost_col, weight_col)


def _objective_function(
        delta_t: pd.Series, 
        median: float, f: 
        Callable, 
        p: float
    ) -> float:
    """ Objective function for the approach of Merlin (2020).

    This objective function looks at the difference between
    weighted opportunities below and above the median
    trip length.
    """
    
    below_mdn = delta_t.loc[:median].copy()
    below_mdn_costs = below_mdn.index.to_numpy()
    below_mdn_f = f(p, below_mdn_costs)
    below_mdn_opp_times_cost = below_mdn.to_numpy().dot(below_mdn_f)

    above_mdn = delta_t.loc[median+1:].copy()
    above_mdn_costs = above_mdn.index.to_numpy()
    above_mdn_f = f(p, above_mdn_costs)
    above_mdn_opp_times_cost = above_mdn.to_numpy().dot(above_mdn_f)

    return above_mdn_opp_times_cost - below_mdn_opp_times_cost


def _estimate_parameter(
        median: int, 
        delta_t: pd.Series, 
        f: Callable, 
        initial_parameter: float,
        max_stepsize: float
    ) -> float:
    """ 
    Uses a Newton-Raphson technique to estimate the parameter value that 
    minimizes the difference between impedane-weighted opportunities for trips 
    less than or equal to the median length and for trips above 
    the median length.

    This function is currently designed for negative exponential, power and 
    Gaussian functions, but can be extended to other single-parameter functions. 

    Args:

        median: median travel time
        delta_t: average number of destinations reached from all origins
            Calculated in calibrate_gravity_access_measure
        f: The impedance function to use: Must be one of 
            neg_exponential, power or gaussian
        initial_parameter: Initial parameter estimate to start the algorithm.
        max_stepsize: Maximum step size for the parameter estimate in each 
        iteration. Used to improve convergence and prevent overshooting.
                
    Returns:
        float:  estimated parameter

    Notes:
        This is the utility function that we are trying to minimize
            min(abs(
                \sum_{t=1}^{median} \delta_t * f(p, cost) - 
                \sum_{t=median+1}^{T} \delta_t * f(p, cost)
        ))
        where f is the impedance function and p is the parameter estimate

    """
    INCR = 1.0e-6
    THRESHOLD = 1.0e-5
    MAX_ITERATIONS = 1000
    initial_parameter = float(initial_parameter)
    max_stepsize = abs(float(max_stepsize))
    p = float(initial_parameter)

    print('Estimating parameter')
    i = 0
    while i <= MAX_ITERATIONS:
        sc = _objective_function(delta_t, median, f, p)

        print(f' iteration {i}:')
        print(f'    current parameter estimate: {p}')
        print(f'    current score: {sc}')

        if np.isnan(sc) or np.isinf(sc):
            raise RuntimeError("Objective function is either infinite or NaN, exiting algorithm")
        if abs(sc) < THRESHOLD:
            print(f'Solution converged. Final parameter estimate: {p}')
            return p

        # An objective function > 0 means that f_ij*O_j is higher above the 
        # median than below This is corrected by reducing the parameter, 
        # for negative exponential, power (making it more negative) 
        # # and Gaussian (reducing sigma).    
        sc_p = _objective_function(delta_t, median, f, p + INCR)
        grad = (sc_p - sc) / INCR
        # Find delta_x using Newton-Raphson technique
        delta_x = -sc / abs(grad)
        # scale delta_x to max stepsize if needed
        if abs(delta_x) > max_stepsize: 
            delta_x *= max_stepsize / abs(delta_x)
        p += delta_x
        i += 1

    print(f'Could not estimate parameter within {MAX_ITERATIONS} iterations')
    print(f'Try changing the initial parameter estimate and/or max_stepsize')
    return None


def calibrate_gravity_access_measure(
        c_ij: pd.Series,
        o_i: pd.Series,
        d_j: pd.Series,
        function: str,
        method: str,
        *,
        t_ij: pd.Series | None = None,
        median: float | None = None,
        initial_parameter: float | None =None,
        max_stepsize: float | None = None,
        output_reachable_destinations=False,
    ) -> float | Tuple[float, pd.Series]:
    """ Calibrate a gravity access parameters.

    Uses the method of Merlin (2020) to calibrate parameters for a gravity 
    access measure.
        
    Args:
        c_ij: Cost matrix as produced by travel time calculators.
            The origins and destinations should be in the same zone system.
            The cost matrix should include an estimate of intrazonal travel
            times. 
        o_i: Population at each origin, i.
        d_j: Opportunities at each destination, j. 
        function: The distance decay function to use. Current options are:
            'neg_exponential':  $e^{-\\beta c_{ij}}$ 
            'power': $c_{ij}^(-\alpha) = e^(-\alpha \ln(c_{ij}))$
            'gaussian': &e^(c_{ij}^2 / (-2 \sigma^2))
        method: The method to use for calibration. Current options are:
            'median', or 'trip_list'.
        t_ij: Observed trips for desired travel segment from zone i to zone j. 
            In Toronto region, this matrix can be created from TTS data.
            Used if method is 'trip_list'. Default is None.
        median: The observed median travel cost for the desired travel segment.
            Used if method is 'median'. Default is None.
        initial_parameter:
            Float containing initial parameter estimate. If None, sets default as:
                negative exponential: -0.1
                power: -1.0
                gaussian function: 20.0
        max_stepsize:
            Maximum stepsize to improve covergence by reducing overshooting
            in Newton-Raphson optimization. If None, sets default as:
                negative exponential: 0.01
                power: 0.1
                gaussian function: 2.0
        output_reachable_destinations:
            Optional flag on whether to also output the weighted average of 
            reachable destinations by cost interval.

    Returns:
        if output_reachable_destinations is False, output a single float, which
            is the calibrated parameter value for the specified function.
        if output_reachable_destinations is True, output is a tuple as follows:
            - float: the calibrated parameter value for the specified function.
            - pd.Series: weighted average of reachable destinations by cost 
              interval.
    
    Notes:
        This method requires the median travel cost for the desired travel
        segment. As this can be difficult to obtain, this function allows
        to either provide the median directly, which is the 'median' method,
        or to provide the number of observed trips between origins and 
        destinations -- such as from the Transportation Tomorrow Survey (TTS) 
        in the Toronto region -- which is the 'trip_list' method. 
        In this case this function will calculate the median travel cost from t
        the observed trips and cost matrix.

        This method is currently implemented for only single-parameter distance 
        decay functions, but can conceptually be extended to multi-parameter 
        functions.

        Intrazonal costs can be estimated and inserted into the cost matrix 
        using the following functions, respectively:
        accessopp.travel_time_computer.intrazonal_costs.compute_intrazonal_costs_from_polygons
        accessopp.travel_time_computer.intrazonal_costs.import insert_intrazonals_into_cost_matrix

    References:
        Merlin L. A. (2020). " A new method using medians to calibrate single-
        parameter spatial interaction models", The Journal of Transport and 
        Land Use, 13(1), 49-70.
    
    """
    if not isinstance(c_ij, pd.Series) or not c_ij.index.nlevels == 2:
            raise ValueError(
                f"Invalid travel cost matrix, c_ij. c_ij must be a "
                f"{ERR_MSG_INV_MATRIX}"
            )
    # Fill in the cost matrix, which may have NaNs
    max_cost = c_ij.max()
    c_ij = c_ij.fillna(max_cost)
    c_ij.name = 'c_ij'

    if not isinstance(d_j, pd.Series) or not d_j.index.nlevels == 1:
            raise ValueError(
                f"Invalid destination opportunities, d_j. d_j must be a "
                f"{ERR_MSG_INV_VCT}"
            )
    if not isinstance(o_i, pd.Series) or not o_i.index.nlevels == 1:
            raise ValueError(
                f"Invalid origin population, o_i. o_i must be a "
                f"{ERR_MSG_INV_VCT}"
            )        
    if method == 'median':
        if not (isinstance(median, int) or isinstance(median, float)):
            raise ValueError(ERR_MSG_INV_MEDIAN)
    elif method == 'trip_list':
        if not isinstance(t_ij, pd.Series) or not t_ij.index.nlevels == 2:
            raise ValueError(
                f"When using method='trip_list', t_ij must be a "
                f"{ERR_MSG_INV_MATRIX}"
            )
        # Calculate the median trip cost
        median = _calculate_median_tripcost(t_ij, c_ij, max_cost)
        print(f'Calculated median cost is {median}.')
    else:
        raise ValueError(ERR_MSG_INV_METHOD)

    if function == 'neg_exponential':
        f = neg_exponential
        if initial_parameter is None:
            initial_parameter = -0.1
        if max_stepsize is None:
            max_stepsize = 0.01
    elif function == 'power':
        f = power
        if initial_parameter is None:
            initial_parameter = -1.0
        if max_stepsize is None:
            max_stepsize = 0.1
    elif function == 'gaussian':
        f = gaussian
        if initial_parameter is None:
            initial_parameter = 20.0
        if max_stepsize is None:
            max_stepsize = 2
    else:
        raise ValueError(ERR_MSG_INV_FNC)

    # Ensure that origin population has all zones (as found in c_ij)
    i_zones = pd.Index(np.sort(c_ij.index.get_level_values(0).unique()))
    o_i = o_i.reindex(i_zones, fill_value=0)
    o_i.name = 'o_i'
    
    # Ensure the destination attractions has all zones (as found in c_ij)
    j_zones = pd.Index(np.sort(c_ij.index.get_level_values(1).unique()))
    d_j = d_j.reindex(j_zones, fill_value=0)
    d_j.name = 'd_j'

    # Integerize cost matrix, then merge in the destination opportunities
    c_ij_int = c_ij.round(0).astype(np.int32)
    c_ij_with_dj = pd.DataFrame(c_ij_int).reset_index().merge(
        d_j, left_on='destination_id', right_index=True)

    # Step 1: Create a table of the number of destinations D_it reachable
    # for every minute t=1, 2, 3, ... T from each origin zone, i
    # # Call this table d_it
    d_it = c_ij_with_dj.groupby(
        ['origin_id', 'c_ij'])['d_j'].sum().unstack().fillna(0)

    # Step 2: Create a vector of the average number of destinations 
    # reachable from all origins for every cost interval
    # $\sum_{i=0}^{N} o_i \times d_{it} / \sum_{i=0}^N o_i$
    # call this delta_t
    denominator = o_i.sum()
    oi_times_dit = d_it.multiply(o_i, axis=0)
    numerator = oi_times_dit.sum(axis=0)
    delta_t = numerator / denominator

    # Step 3: Use iterative procedure to minimize the difference between
    # weighted opportunities below and above the median.
    estimated_parameter = _estimate_parameter(
        median, delta_t, f, initial_parameter, max_stepsize)

    if not output_reachable_destinations:
        return estimated_parameter
    else:
        return (estimated_parameter, delta_t)


