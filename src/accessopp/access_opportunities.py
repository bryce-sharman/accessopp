import numpy as np
import pandas as pd

""" Module with function that calculate access to opportunities given a cost matrix and optional weights. """

def process_cost_matrix(df, fill_value=np.NaN):
    """ 
    Process cost matrix from tall format produced by OTP2TravelTimeComputer and
    R5PYTravelTimeComputer to a wide format suitable for accessibility calculations.
    
    Paramters
    ---------
    df: pandas.DataFrame
        Cost matrix in format produced by OTP2TravelTimeComputer and R5PYTravelTimeComputer.
        This format expects the following columns: 'from_id', 'to_id', 'travel_time'.
        The index is not used.
    fill_value: float, optional
        Value with which to fill any blanks in the matrix after post-processing

    Returns:
    pd.DataFrame
        cost matrix in wide format, with indices corresponding to 'from_id`, `to_id`

    """
    s = df.set_index(['from_id', 'to_id']).squeeze()
    return s.unstack(fill_value=fill_value)


#region Impedance Functions
def within_threshold(df, threshold):
    """ Calculates impedance matrix assuming cumulative opportunities (1 if cost is within threshold, 0 otherwise)

    Parameters
    ----------
    df: pd.DataFrame
        Cost matrix in format produced by OTP2TravelTimeComputer and R5PYTravelTimeComputer.
    threshold: float or int
        threshold to test, should be real number > 0

    Returns
    -------
    pd.DataFrame
        Impedance matrix in wide format.

    """
    cost_matrix = process_cost_matrix(df)
    cm = cost_matrix.to_numpy()
    impedance = np.where(cm <= threshold, 1, 0)
    return_df = pd.DataFrame(
        data=impedance, 
        index=cost_matrix.index, 
        columns=cost_matrix.columns, 
        dtype=np.int32
    )
    return_df.index.name = "from_id"
    return_df.columns.name = "to_id"
    return_df.name="within_threshold_impedance"
    return return_df

def negative_exp(df, beta):
    """ Calculates impedance matrix assuming negative exponential decay function.

    Parameters
    ----------
    df: pd.DataFrame
        Cost matrix in format produced by OTP2TravelTimeComputer and R5PYTravelTimeComputer.
    beta: float
        beta parameter of negative exponential function. Should be a real number < 0.

    Returns
    -------
    pd.DataFrame
        Impedance matrix in wide format.

    """
    cost_matrix = process_cost_matrix(df)
    if beta >= 0:
        raise ValueError("Expecting negative `beta` parameter.")
    cm = cost_matrix.to_numpy()
    impedance = np.exp(beta * cm)
    return_df = pd.DataFrame(
        data=impedance, 
        index=cost_matrix.index, 
        columns=cost_matrix.columns, 
        dtype=np.float64
    )
    return_df.index.name = "from_id"
    return_df.columns.name = "to_id"
    return_df.name="neg_exp_impedance"
    return return_df


def gaussian(df, sigma):
    """ Calculates impedance matrix assuming Gaussian decay function.

    Parameters
    ----------
    df: pd.DataFrame
        Cost matrix in format produced by OTP2TravelTimeComputer and R5PYTravelTimeComputer.
    sigma: float
        standard deviation parameter of Guassian function, should be float > 0.
    
    Returns
    -------
    pd.DataFrame
        Impedance matrix in wide format.

    """    
    cost_matrix = process_cost_matrix(df)
    if sigma <= 0:
        raise ValueError("Expecting positive `sigma` parameter.")
    cm = cost_matrix.to_numpy()
    impedance = np.exp(-(cm * cm) / (2 * sigma * sigma))
    return_df = pd.DataFrame(
        data=impedance, 
        index=cost_matrix.index, 
        columns=cost_matrix.columns, 
        dtype=np.float64
    )
    return_df.index.name = "from_id"
    return_df.columns.name = "to_id"
    return_df.name="gaussian_impedance"
    return return_df

# #endregion
    
#region Primary access measures  (cumulative opportunities)
def calc_impedance_matrix(df, impedance_function, **kwargs):
    """ Calculates the impedance matrix given the stored cost matrix, saving in .impedance_matrix attribute. 
    
    Parameters
    ----------
    df: pandas.DataFrame
        Cost matrix in format produced by OTP2TravelTimeComputer and R5PYTravelTimeComputer.
        This format expects the following columns: 'from_id', 'to_id', 'travel_time'.
        The index is not used.

    impedance_function: function
        One of the impedance functions specified in this module. Current options are:
            within_threshold - for cumulative opportunities access
            negative_exp - for negative exponential gravity model
            gaussian - for gaussian weighted gravity model

    **kwargs:
        parameters expected by impedance function.

    Returns
    -------
    pd.DataFrame
        Impedance matrix in wide format.

    """
    return impedance_function(df, **kwargs)

def calc_spatial_access(
        c_ij, impedance_func, o_j=None, p_i=None, normalize="none", **kwargs):
    """ Calculates spatial access to opportunities. 

    These measures include cumulative opportunities, weighted gravity and 
    competitive (not yet implemented) measures.
    
    Parameters
    ----------
    c_ij: pandas.DataFrame
        Cost matrix from origin i to destination j in format produced by 
        OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
        the following columns: 'from_id', 'to_id', 'travel_time'. 
        The index is not used.

    impedance_func: function
        One of the impedance functions specified in this module.
            within_threshold - for cumulative opportunities access
            negative_exp - for negative exponential gravity model
            gaussian - for gaussian weighted gravity model

    o_j: pd.Series, optional
        Opportunties at destination j. If None then 
        weights 1.0 are assigned. Index must match cost_matrix columns.

    p_i: pd.Series, optional
        Population at origin i. If defined, will calculate a weighted
        access to opportunities all origins into a single number. 
        Index must match cost_matrix index.

    normalize: str
        one of the following options:
            "median": normalize access with respect to median access
            "average": normalize access with respect to average access
            "maximum": normalize access with respect to highest access
            "none": do not normalize. This is the default option
        This parameter is ignored if origin_weights is defined.

    **kwargs:
        parameters expected by impedance function.

    Returns
    -------
    pandas.Series or float
        if origin_weights is not defined, returns pandas.Series with the 
            access to opportinities for each origin.
        if origin_weights is defined, retuns float with the total access to 
            opportunities for all origins.
    """
    if normalize not in ['median', 'average', 'maximum', 'none']:
        raise ValueError('Invalid `normalize` parameter. ')

    f_ij = calc_impedance_matrix(c_ij, impedance_func, **kwargs)

    if o_j is None:
        destination_access = pd.Series(
            index=f_ij.index, data=f_ij.sum(axis=1))
    else:
        if not o_j.index.equals(f_ij.columns):
            print('Reindexing destination weights vector '
                  'to match cost matrix columns.')
            o_j = o_j.reindex(f_ij.columns, fill_value=0.0)

        mul = f_ij.multiply(o_j)
        destination_access = mul.sum(axis=1)

    if p_i is not None:
        if not p_i.index.equals(f_ij.index):
            p_i = p_i.reindex(f_ij.index, fill_value=0.0)
            print('Reindex origin weights vector to mach cost_matrix` index.')
        return destination_access.dot(p_i)
    else:
        # Apply normalization
        if normalize == "median":
            return destination_access / destination_access.median()
        elif normalize == "average":
            return destination_access / destination_access.mean()
        elif normalize == "max":
            return destination_access / destination_access.max()
        else:
            return destination_access


def calc_spatial_availability(
        c_ij, impedance_func, o_j, p_i, alpha=1.0, **kwargs):
    """ Calculates competitive access to opportunities.

    This function calculates the availability -- new name as it reflects 
    opportunities that can be 'claimed' and not just accessed -- 
    presented in Soukhov et al., see citation in the note, below.
    
    Parameters
    ----------
    c_ij: pandas.DataFrame
        Cost matrix from origin i to destination j in format produced by 
        OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
        the following columns: 'from_id', 'to_id', 'travel_time'. 
        The index is not used.

    impedance_func: function
        One of the impedance functions specified in this module.
            within_threshold - for cumulative opportunities access
            negative_exp - for negative exponential gravity model
            gaussian - for gaussian weighted gravity model

    o_j: pd.Series
        Opportunities at destination j. Index must match cost_matrix 
        destinations.

    p_i: pd.Series
        Population at origin i. Index must match cost_matrix origins.

    alpha: optional
        Modulates the effect of demand by population. When alpha < 1, 
        opportunities are allocated more rapidly to smaller centers relative 
        to larger ones; alpha > 1 achieves the opposite effect.
        Defaults to 1.0. 

    **kwargs:
        parameters expected by impedance function.

    Returns
    -------
    pandas.Series
        Availability of opportunities by origin i. 

    Note
    ----
    This code is a single mode version of the competitive accessibility shown
    in the following paper. 

    Soukhov A, Pa´ez A, Higgins CD,Mohamed M (2023) Introducing spatial 
    availability, a singly-constrained measure of competitive accessibility. 
    PLoS ONE 18(1): e0278468. https://doi.org/10.1371/journal.pone.0278468

    """

    # calculate population balancing factor, F^p_{ij}
    p_i_alpha = p_i.pow(alpha)
    sum_p_i_alpha = p_i_alpha.sum()
    f_p_i = p_i_alpha / sum_p_i_alpha

    # Calculate the cost balancing factor, F^c_{ij}
    f_ij = calc_impedance_matrix(c_ij, impedance_func, **kwargs)
    sum_i_fij = f_ij.sum(axis=0)
    f_c_ij = f_ij / sum_i_fij

    # Calculate the final balancing factor, F^t_{ij}
    numerator = f_c_ij.mul(f_p_i, axis=0)
    numerator_sum = numerator.sum(axis=0)
    f_t_ij = numerator.divide(numerator_sum)

    # Calculate availability
    mul = o_j.multiply(f_t_ij)
    v_i = mul.sum(axis=1)
    return v_i


def calc_spatial_heterogeneous_availability(population_segments, o_j, alpha=1.0):
    """ Calculates competitive access for a heterogeneous population.

    This function calculates the availability -- new name as it reflects 
    opportunities that can be 'claimed' and not just accessed -- 
    presented in Soukhov et al., see citation in the note, below.
    
    Parameters
    ----------
    population_segments: dictionary
        Defined as a one-character label for each key, and a sub-dictionary 
        defining the cost_impedance function for that mode using the following 
        keys:

        'p_i': pd.Series
            Population this category at origin i. Index must match 
            cost_matrix origins.
        'c_ij': pandas.DataFrame
            Cost matrix from origin i to destination j in format produced by 
            OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
            the following columns: 'from_id', 'to_id', 'travel_time'. 
            The index is not used.
        'impedance_func': function
            One of the impedance functions specified in this module.
                within_threshold - for cumulative opportunities access
                negative_exp - for negative exponential gravity model
                gaussian - for gaussian weighted gravity model
        'threshold': int or float:
            threshold used for within_threshold impedance function.
        'beta': float
            beta parameter of negative_exp function. Should be real number < 0.
        'sigma': float
            standard deviation parameter of Guassian function.
            Should be a real number > 0.


    o_j: pd.Series
        Opportunities at destination j. Index must match cost_matrix 
        destinations. Opportunities are assumed to be accessible by 
        all people, regardless of population category.

    alpha: optional
        Modulates the effect of demand by population. When alpha < 1, 
        opportunities are allocated more rapidly to smaller centers relative 
        to larger ones; alpha > 1 achieves the opposite effect.
        Defaults to 1.0. 

    Returns
    -------
    pandas.Series
        Availability of opportunities by origin i. 

    Note
    ----
    This code is a single mode version of the competitive accessibility shown
    in the following paper. 

    Soukhov A, Tarriño-Ortiz J, Soria-Lara JA, Pa´ez A (2024) Multimodal spatial 
    availability: A singly-constrained measure of accessibilityconsidering 
    multiple modes. PLoS ONE 19(2): e0299077. 
    https://doi.org/10.1371/journal.pone.0299077

    """
    # Basic input validation and create dictionary to hold results
    reqd_keys = ['p_i', 'c_ij', 'impedance_func']
    kwargs = {}
    for pop_label, pop_def in population_segments.items():

        if len(pop_label) != 1:
            raise ValueError("Population segments must be defined by a "
                             "one-character label.")
        for rk in reqd_keys:
            if rk not in pop_def.keys():
                raise AttributeError(f"Keys {reqd_keys} must be defined for "
                                     f"each population segment.")
        kwargs[pop_label] = {}
        if pop_def['impedance_func'] == within_threshold:
            if 'threshold' not in pop_def.keys():
                raise AttributeError("threshold must be defined for "
                                     "within_threshold impedance function.")
            kwargs[pop_label]['threshold'] = pop_def['threshold']
        elif pop_def['impedance_func'] == negative_exp:
            if 'beta' not in pop_def.keys():
                raise AttributeError("beta must be defined for "
                                     "negative_exp impedance function.")
            kwargs[pop_label]['beta'] = pop_def['beta']
        elif pop_def['impedance_func'] == gaussian:
            if 'sigma' not in pop_def.keys():
                raise AttributeError("sigma must be defined for "
                                     "gaussian impedance function.")
            kwargs[pop_label]['sigma'] = pop_def['sigma']
        else:
            raise ValueError(f"impedance_func must be one of: "
                             "within_threshold, negative_exp, gaussian")

    # calculate scaled population, by population subgroup, $f^{pm}_i$  
    p_i = {}       # This term holds the scaled population
    f_pm_i = {}    # $f^{pm}_i$ term for final equation
                   # for each mode, this is an origin vector
    total_scaled_pop = 0.0
    for pop_label, pop_def in population_segments.items():
        p_i[pop_label] = pop_def['p_i'].pow(alpha)
        total_scaled_pop += p_i[pop_label].sum()
        p_i[pop_label]
    # $f^{pm}_i$   is the population divided by the total population
    for pop_label, pop_def in population_segments.items():
        f_pm_i[pop_label] = p_i[pop_label] / total_scaled_pop

    # Calculate the cost balancing factor, $F^{cm}_{ij}$
    # Read a cost matrix, just to find the indices to create our 
    # summation vector
    int_f_cm_ij = {}   # numerator (impedance function of each mode)
    f_cm_ij = {}       # final scaled factor
                       # for each mode this a matrix
    first_key = list(population_segments.keys())[0]
    c_ij = process_cost_matrix(population_segments[first_key]['c_ij'])
    denom_f_cm_ij = pd.Series(index=c_ij.columns, data=0.0)
    for pop_label, pop_def in population_segments.items():
        int_f_cm_ij[pop_label] = calc_impedance_matrix(
            pop_def['c_ij'], pop_def['impedance_func'], **kwargs[pop_label])
        sum_over_i = int_f_cm_ij[pop_label].sum(axis=0)
        denom_f_cm_ij = denom_f_cm_ij + sum_over_i
    for pop_label, pop_def in population_segments.items():
        f_cm_ij[pop_label] = int_f_cm_ij[pop_label] / denom_f_cm_ij
    # Now perform the multiplication, which is the numerator of 
    # equation 2 of the cited paper
    f_pm_i_times_c_m_ij = {}
    f_t_ij = {}
    denom_f_t_ij = pd.Series(index=c_ij.columns, data=0.0)
    for pop_label, pop_def in population_segments.items():
        f_pm_i_times_c_m_ij[pop_label] = f_cm_ij[pop_label].multiply(
            f_pm_i[pop_label], axis=0)
        sum_over_i = f_pm_i_times_c_m_ij[pop_label].sum(axis=0)
        denom_f_t_ij = denom_f_t_ij + sum_over_i
    for pop_label, pop_def in population_segments.items():
        f_t_ij[pop_label] = f_pm_i_times_c_m_ij[pop_label] / denom_f_t_ij

    # Calculate the availability for each population group
    v_i = {}
    for pop_label, pop_def in population_segments.items():
        mul = o_j.multiply(f_t_ij[pop_label])
        v_i[pop_label] = mul.sum(axis=1)

    return v_i

    
#region Dual access measures
def has_opportunity(c_ij: pd.DataFrame, threshold: int | float):
    """ Calculates whether any opportunities are within threshold cost. 

    Parameters
    ---------
    c_ij: pandas.DataFrame
        Cost matrix from origin i to destination j in format produced by 
        OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
        the following columns: 'from_id', 'to_id', 'travel_time'. 
        The index is not used.
    threshold: float or int
        threshold to test, should be real number > 0

    Returns
    -------
    pandas.Series
        1 if any opportunity is within threshold, 0 otherwise

     """
    test_within_threshold = calc_impedance_matrix(
        c_ij, within_threshold, threshold=threshold)
    return test_within_threshold.max(axis=1)
    
def closest_opportunity(c_ij: pd.DataFrame):
    """ Calculates cost to the closest opportunity from each origin.

    Parameters
    ---------
    c_ij: pandas.DataFrame
        Cost matrix from origin i to destination j in format produced by 
        OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
        the following columns: 'from_id', 'to_id', 'travel_time'. 
        The index is not used.

    Returns
    -------
    pandas.Series
        Closest opportunity from each origin.

    """
    cost_matrix = process_cost_matrix(c_ij)
    cm = cost_matrix.to_numpy()
    data = np.min(cm, axis=1, initial=10000.0, where=np.isfinite(cm))
    index = cost_matrix.index
    return pd.Series(data=data, index=index)

def nth_closest_opportunity(c_ij, n):
    """ Calculates cost to the nth closest opportunity from each origin.

    Parameters
    ----------
    c_ij: pandas.DataFrame
        Cost matrix from origin i to destination j in format produced by 
        OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
        the following columns: 'from_id', 'to_id', 'travel_time'. 
        The index is not used.
    n: int
        Nth-opportunity to which to calculate cost. Expecting an integer >= 2.

    Returns
    -------
    pandas.Series
        pandas series where the index is the same as that of the cost_matrix
        Each value corresponds to the nth-closest opportunity from the origin.

    """
    if not isinstance(n, int):
        raise AttributeError('Parameter `n` should be an integer >= 2.')
    if not n >= 2:
        raise ValueError('Parameter `n` should be an integer >= 2.')
    cost_matrix = process_cost_matrix(c_ij)
    cm = cost_matrix.to_numpy()
    data = np.sort(cm, axis=1)[:, n-1]
    index=cost_matrix.index
    return pd.Series(data=data, index=index)

#endregion
