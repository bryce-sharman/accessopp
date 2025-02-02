""" 
Module with function that calculate various access to opportunities 
measures given a cost matrix and optional weights. 
"""

from collections.abc import Callable
import numpy as np
import pandas as pd
from typing import Dict, Optional

from accessopp.enumerations import INDEX_COLUMNS



#region Impedance Functions
def within_threshold(cm: pd.Series, threshold: float) -> pd.DataFrame:
    """ 
    Calculates impedance matrix assuming cumulative opportunities 
    (1 if cost is within threshold, 0 otherwise).

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - threshold: threshold to test, should be real number > 0

    Returns:
        - Impedance matrix in wide format.

    """
    if threshold <= 0:
        return ValueError('threshold argument must be > 0')
    cm = cm.unstack()
    cmnp = cm.to_numpy()
    return_df = pd.DataFrame(
        data=np.where(cmnp <= threshold, 1, 0), 
        index=cm.index, 
        columns=cm.columns, 
        dtype=np.int64
    )
    return_df.name = "within_threshold_impedance"
    return return_df

def negative_exp(cm: pd.Series, beta: float) -> pd.DataFrame:
    """ Calculates impedance matrix assuming negative exponential decay function.

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - beta: beta parameter of negative exponential function. 
                Should be a real number < 0.

    Returns:
        - Impedance matrix in wide format.

    """
    if beta >= 0:
        raise ValueError("Expecting negative `beta` parameter.")
    cm = cm.unstack()
    cmnp = cm.to_numpy()
    return_df = pd.DataFrame(
        data=np.exp(beta * cmnp), 
        index=cm.index, 
        columns=cm.columns, 
        dtype=np.float64
    )
    return_df.name="neg_exp_impedance"
    return return_df

def gaussian(cm: pd.Series, sigma: float) -> pd.DataFrame:
    """ Calculates impedance matrix assuming Gaussian decay function.

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - sigma: standard deviation parameter of Guassian function, 
                 should be float > 0.
    
    Returns:
        - Impedance matrix in wide format.

    """   
    if sigma <= 0:
        raise ValueError("Expecting positive `sigma` parameter.")
    cm = cm.unstack()
    cmnp = cm.to_numpy()
    return_df = pd.DataFrame(
        data=np.exp(-(cmnp * cmnp) / (2 * sigma * sigma)), 
        index=cm.index, 
        columns=cm.columns, 
        dtype=np.float64
    )
    return_df.name="gaussian_impedance"
    return return_df

#endregion

#region Dual access measures
def has_opportunity(cm: pd.Series, threshold: float) -> pd.Series:
    """ Calculates if opportunity exists within threshold for each origin.

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - threshold: threshold to test, should be real number > 0

    Returns:
       - pandas Series: 1 if any opportunity is within threshold, 0 otherwise

    """
    return within_threshold(cm, threshold).max(axis=1)

def closest_opportunity(cm: pd.Series) -> pd.Series:
    """ Calculates cost to the closest opportunity from each origin.

    Args:
        - cm: Cost matrix as produced by travel time calculators

    Returns:
       - Closest opportunity from each origin.

    """
    cm = cm.unstack()
    cmnp = cm.to_numpy()
    return pd.Series(
        index=cm.index,
        data=np.min(cmnp, axis=1, initial=10000.0, where=np.isfinite(cmnp))
    )
 
def nth_closest_opportunity(cm: pd.Series, n: int):
    """ Calculates cost to the nth closest opportunity from each origin.

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - n: Nth-opportunity to which to calculate cost,  >= 2.

    Returns:
       - nth-closest opportunity from each origin.

    """
    if n < 2:
        raise ValueError('Parameter `n` should be an integer >= 2.')
    cm = cm.unstack()
    cmnp = cm.to_numpy()
    return pd.Series(
        index=cm.index,
        data=np.sort(cmnp, axis=1)[:, n-1]
    )

#endregion

#region Primary access measures  (cumulative opportunities)
def calc_spatial_access(
        cm: pd.Series, 
        impedance_func: Callable, 
        o_j: pd.Series,
        p_i: Optional[pd.Series]=None, 
        normalize: str="none", 
        **kwargs
    ) -> pd.Series | float:
    """ 
    Calculates non-competitive -- such as cumulative opportunities
    and weighted gravity -- spatial access to opportunities. 

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - impedance_func: One of the provided impedance functions:
            - within_threshold - for cumulative opportunities access
            - negative_exp - for negative exponential weighted gravity model
            - gaussian - for gaussian weighted gravity model
        - o_j: Opportunties at destination j.
        - p_i: Population at origin i, optional. 
            If defined, will calculate a weighted access to opportunities all 
            origins to produce a total (float) for the region.
        - normalize: one of the following options:
            "median": normalize access with respect to median access
            "average": normalize access with respect to average access
            "maximum": normalize access with respect to highest access
            "none": do not normalize. This is the default option
            This parameter is ignored if p_i is defined.
        - **kwargs: parameters expected by impedance function.

    Returns:
        pandas.Series or float
            if p_i is not defined, returns pandas.Series with the 
                access to opportinities for each origin.
            if p_i is defined, retuns float with the total access to 
                opportunities for all origins.
    """
    if normalize not in ['median', 'average', 'maximum', 'none']:
        raise ValueError('Invalid `normalize` parameter. ')
    f_ij = impedance_func(cm, **kwargs)

    if not o_j.index.equals(f_ij.columns):
        print('Reindexing destination weights to match cost matrix columns.')
        o_j = o_j.reindex(f_ij.columns, fill_value=0.0)
    mul = f_ij.multiply(o_j)
    destination_access = mul.sum(axis=1)

    if isinstance(p_i, pd.Series) and len(p_i) > 0:
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
        else:  # 'none'
            return destination_access
  



def calc_spatial_availability(
        cm: pd.Series, 
        impedance_func: Callable, 
        o_j: pd.Series,
        p_i: pd.Series, 
        alpha: float=1.0, 
        **kwargs
    ):
    """ Calculates competitive access to opportunities.

    This function calculates the availability -- new name as it reflects 
    opportunities that can be 'claimed' and not just accessed -- 
    presented in Soukhov et al., see citation in the note, below.

    This code is a single mode version of the competitive accessibility shown
    in the following paper. 

    Soukhov A, Pa´ez A, Higgins CD,Mohamed M (2023) Introducing spatial 
    availability, a singly-constrained measure of competitive accessibility. 
    PLoS ONE 18(1): e0278468. https://doi.org/10.1371/journal.pone.0278468

    Args:
        - cm: Cost matrix as produced by travel time calculators
        - impedance_func: One of the provided impedance functions:
            - within_threshold - for cumulative opportunities access
            - negative_exp - for negative exponential weighted gravity model
            - gaussian - for gaussian weighted gravity model
        - o_j: Opportunties at destination j.
        - p_i: Population at origin i.
        - alpha: Modulates the effect of demand by population. When alpha < 1, 
            opportunities are allocated more rapidly to smaller centers relative 
            to larger ones; alpha > 1 achieves the opposite effect.
            Defaults to 1.0. 
        - **kwargs: parameters expected by impedance function.

    Returns:
        - availability of opportunities by origin i. 

    """
    # calculate population balancing factor, F^p_{ij}
    p_i_alpha = p_i.pow(alpha)
    sum_p_i_alpha = p_i_alpha.sum()
    f_p_i = p_i_alpha / sum_p_i_alpha

    # Calculate the cost balancing factor, F^c_{ij}
    f_ij = impedance_func(cm, **kwargs)
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


def calc_spatial_heterogeneous_availability(
        population_segments: Dict, 
        o_j, 
        alpha=1.0
    ) -> Dict:
    """ Calculates competitive access for a heterogeneous population.

    This function calculates the availability -- new name as it reflects 
    opportunities that can be 'claimed' and not just accessed -- 
    presented in the following paper:
    
    Soukhov A, Tarriño-Ortiz J, Soria-Lara JA, Pa´ez A (2024) Multimodal spatial 
    availability: A singly-constrained measure of accessibilityconsidering 
    multiple modes. PLoS ONE 19(2): e0299077. 
    https://doi.org/10.1371/journal.pone.0299077

    Args:
        - population_segments: dictionary
            Dictonary with defining population and impedances
            for all population segments. Each dictionary value is a 
            sub-dictionary defined as follows:
            - 'p_i': pd.Series
                Population this category at origin i. Index must match 
                cost_matrix origins.
            - 'c_ij': pandas.DataFrame
                Cost matrix from origin i to destination j in format produced by 
                OTP2TravelTimeComputer and R5PYTravelTimeComputer.This format expects 
                the following columns: 'from_id', 'to_id', 'travel_time'. 
                The index is not used.
            - 'impedance_func': function
                One of the impedance functions specified in this module.
                    within_threshold - for cumulative opportunities access
                    negative_exp - for negative exponential gravity model
                    gaussian - for gaussian weighted gravity model
            - 'threshold': int or float:
                threshold used for within_threshold impedance function.
            - 'beta': float
                beta parameter of negative_exp function. 
                Should be real number < 0.
            - 'sigma': float
                standard deviation parameter of Guassian function.
                Should be a real number > 0.
        - o_j: Opportunities at destination j. Index must match cost_matrix 
            destinations. Opportunities are assumed to be accessible by 
            all people, regardless of population category.
        - alpha: Modulates the effect of demand by population. When alpha < 1, 
            opportunities are allocated more rapidly to smaller centers relative 
            to larger ones; alpha > 1 achieves the opposite effect.
            Defaults to 1.0. 

    Returns:
        - Dictionary contain availability of opportunities by origin i for
            each population segment. 

    """
    # Basic input validation and create dictionary to hold results
    reqd_keys = ['p_i', 'c_ij', 'impedance_func']
    kwargs = {}
    for pop_label, pop_def in population_segments.items():
        for rk in reqd_keys:
            if rk not in pop_def.keys():
                raise AttributeError(
                    f"{reqd_keys} must be defined for each population segment.")
        kwargs[pop_label] = {}
        if pop_def['impedance_func'] == within_threshold:
            if 'threshold' not in pop_def.keys():
                raise AttributeError(
                    "threshold must be defined for "
                    "within_threshold impedance function.")
            kwargs[pop_label]['threshold'] = pop_def['threshold']
        elif pop_def['impedance_func'] == negative_exp:
            if 'beta' not in pop_def.keys():
                raise AttributeError(
                    "beta must be defined for "
                    "negative_exp impedance function.")
            kwargs[pop_label]['beta'] = pop_def['beta']
        elif pop_def['impedance_func'] == gaussian:
            if 'sigma' not in pop_def.keys():
                raise AttributeError(
                    "sigma must be defined for "
                    "gaussian impedance function.")
            kwargs[pop_label]['sigma'] = pop_def['sigma']
        else:
            raise ValueError(
                "impedance_func must be one of: "
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
    # Read a cost matrix just to find the indices to create our 
    # summation vector
    int_f_cm_ij = {}   # numerator (impedance function of each mode)
    f_cm_ij = {}       # final scaled factor
                       # for each mode this a matrix
    first_key = list(population_segments.keys())[0]
    c_ij = population_segments[first_key]['c_ij'].unstack()
    denom_f_cm_ij = pd.Series(index=c_ij.columns, data=0.0)
    for pop_label, pop_def in population_segments.items():
        int_f_cm_ij[pop_label] = pop_def['impedance_func'](
            pop_def['c_ij'], **kwargs[pop_label])
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

    
