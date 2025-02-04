import numpy as np
import pandas as pd
from pandas import testing as tm
import pytest

from accessopp.access_opportunities import within_threshold
from accessopp.access_opportunities import negative_exp
from accessopp.access_opportunities import gaussian
from accessopp.access_opportunities import has_opportunity
from accessopp.access_opportunities import closest_opportunity
from accessopp.access_opportunities import nth_closest_opportunity
from accessopp.access_opportunities import calc_spatial_access
from accessopp.access_opportunities import calc_spatial_availability
from accessopp.access_opportunities import calc_spatial_heterogeneous_availability


from accessopp.enumerations import INDEX_COLUMNS, COST_COLUMN, ALL_COLUMNS

REF_INDEX = pd.Index([1, 2], name=INDEX_COLUMNS[0])
REF_COLUMNS = pd.Index([20, 21, 22], name=INDEX_COLUMNS[1])

COST_MATRIX = pd.Series(
        index=pd.MultiIndex.from_product(
            [[1, 2], [20, 21, 22]], 
            names=INDEX_COLUMNS
        ),
        data=[5, 10, 15, 18, 13, 8],
        dtype=np.int64,
        name=COST_COLUMN
    )

SOUKHOV_TESTS_INDEX = pd.Index(['A', 'B', 'C'], name=INDEX_COLUMNS[0])
SOUKHOV_TESTS_COLUMNS = pd.Index([1, 2, 3], name=INDEX_COLUMNS[1])
SOUKHOV_TESTS_MI = pd.MultiIndex.from_product(
        [['A', 'B', 'C'], [1, 2, 3]], names = INDEX_COLUMNS)

SOUKHOV_2023_P_I = pd.Series(
        index=SOUKHOV_TESTS_INDEX,
        data=[50000, 150000, 10000],
        dtype=np.int64,
        name='p_i'
    )
SOUKHOV_2023_O_J = pd.Series(
        index=SOUKHOV_TESTS_COLUMNS,
        data=[100000, 100000, 10000],
        dtype=np.int64,
        name='o_j'
    )

SOUKHOV_2023_C_IJ = pd.Series(
        index=SOUKHOV_TESTS_MI,
        data=[15, 30, 100, 30, 15, 100, 100, 100, 15],
        name=COST_COLUMN
    )

SOUKHOV_2024_P_I_X  = pd.Series(
        index=SOUKHOV_TESTS_INDEX,
        data=[50000 * 0.33, 150000 * 0.40, 10000 * 0.30],
        dtype=np.int64
    )
    
SOUKHOV_2024_P_I_Z = pd.Series(
        index=SOUKHOV_TESTS_INDEX,
        data=[50000 * 0.67, 150000 * 0.60, 10000 * 0.70],
        dtype=np.int64
    )

SOUKHOV_2024_O_J = pd.Series(
        index=SOUKHOV_TESTS_COLUMNS,
        data=[100000, 100000, 10000],
        dtype=np.int64
    )

SOUKHOV_2024_C_IJ_X = pd.Series(
        index=SOUKHOV_TESTS_MI,
        data=[15, 30, 100, 30, 15, 100, 100, 100, 15],
        name=COST_COLUMN
    )
    
SOUKHOV_2024_C_IJ_Z = pd.Series(
            index=SOUKHOV_TESTS_MI,
            data=[10, 25, 80, 25, 10, 80, 80, 80, 10],
            name=COST_COLUMN
        )

def test_impedance_within_threshold():
    """ Tests within-threshold impedance function."""
    cm = within_threshold(COST_MATRIX, 13)
    ref_cm = pd.DataFrame(
        index = REF_INDEX,
        columns = REF_COLUMNS,
        data = [[1, 1, 0], [0, 1, 1]],
        dtype=np.int64
    )
    tm.assert_frame_equal(cm, ref_cm)


def test_impedance_negative_exp():
    """ Tests negative exponential impedance function.
    
    Note: 
        ref matrix was calculated in Excel to avoid using the same code 
        in the test as the implementation.
    """
    cm = negative_exp(COST_MATRIX, -0.1)
    ref_cm = pd.DataFrame(
        index = REF_INDEX,
        columns = REF_COLUMNS,
        data = [
            [0.60653066, 0.36787944, 0.22313016], 
            [0.16529889, 0.27253179, 0.44932896]
        ],
        dtype=np.float64
    )
   
    tm.assert_frame_equal(cm, ref_cm)


def test_impedance_gaussian():
    """ Test for Gaussian impedance function.
    
    Note: 
        ref matrix was calculated in Excel to avoid using the same code 
        in the test as the implementation.
    """
    cm = gaussian(COST_MATRIX, sigma=10)
    ref_cm = pd.DataFrame(
        index = REF_INDEX,
        columns = REF_COLUMNS,
        data = [
            [0.88249690, 0.60653066, 0.32465247], 
            [0.19789870, 0.42955736, 0.72614904]
        ],
        dtype=np.float64
    )
    tm.assert_frame_equal(cm, ref_cm)

def test_dual_has_opportunity_within_threshold_3():
    """ Test if has opportunity within threshold cost of 3 minutes. """
    result = has_opportunity(COST_MATRIX, 3)
    ref_result = pd.Series(
        index=REF_INDEX, 
        data=[0, 0],
        dtype=np.int64
    )
    tm.assert_series_equal(result, ref_result)

def test_dual_has_opportunity_within_threshold_7():
    """ Test dual access: has opportunity within threshold cost of 7 minutes. """
    result = has_opportunity(COST_MATRIX, 7)
    ref_result = pd.Series(
        index=REF_INDEX, 
        data=[1, 0],
        dtype=np.int64
    )
    tm.assert_series_equal(result, ref_result)

def test_dual_closest_opportunity():
    """ Test dual access: closest opportunity."""
    result = closest_opportunity(COST_MATRIX)
    ref_series = pd.Series(
        index=REF_INDEX, 
        data=[5, 8],
        dtype=np.int64
    )
    tm.assert_series_equal(result, ref_series)

def test_dual_2nd_closest_opportunity():
    """Test dual access: second closest opportunity"""
    result = nth_closest_opportunity(COST_MATRIX, 2)
    ref_series = pd.Series(
        index=REF_INDEX, 
        data=[10, 13],
        dtype=np.int64
    )
    tm.assert_series_equal(result, ref_series)


def test_primal_gaussian():
    """ Test the access to opportunities with Gaussian weighting. """
    destination_weights = pd.Series(
        data=[1, 3, 5], 
        index=REF_COLUMNS, 
        dtype=np.float64
    )
    result = calc_spatial_access(
        COST_MATRIX, gaussian, destination_weights, sigma=10)
    ref_series = pd.Series(
        index=REF_INDEX, 
        data=[
            1*0.88249690 + 3*0.60653066 + 5*0.3246524, 
            1*0.19789870 + 3*0.42955736 + 5*0.72614904
        ],
        dtype=np.float64
    )
    tm.assert_series_equal(result, ref_series)

def test_primal_gaussian_reindex_needed():
    """ Test the access to opportunities when reindex is needed. """
    # Note that this destination_weights vector is missing final index
    destination_weights = pd.Series(
        data=[1, 3], 
        index=[20, 21], 
        dtype=np.float64
    )
    result = calc_spatial_access(
            COST_MATRIX, gaussian, destination_weights, sigma=10)
    ref_series = pd.Series(
        index=REF_INDEX, 
        data=[
            1*0.88249690 + 3*0.60653066, 
            1*0.19789870 + 3*0.42955736
        ],
        dtype=np.float64
    )
    tm.assert_series_equal(result, ref_series)

def test_primal_gaussian_with_origin_weights():
    """ Test the access to opportunities with both origin weights. """
    origin_weights = pd.Series(
        index=REF_INDEX, 
        data=[12, 15], 
        dtype=np.float64
    )
    destination_weights = pd.Series(
        data=[1, 3, 5], 
        index=REF_COLUMNS, 
        dtype=np.float64
    )
    result = calc_spatial_access(
        COST_MATRIX, gaussian, destination_weights, origin_weights, sigma=10)
    ref_result = \
        (1*0.88249690 + 3*0.60653066 + 5*0.32465247) * 12 + \
        (1*0.19789870 + 3*0.42955736 + 5*0.72614904) * 15
    assert np.isclose(result, ref_result)


def test_soukhov_2023_spatial_access(soukhov_2023_c_ij, soukhov_2023_o_j):
    """ Test uncontrained access using test values from paper, see note.

    Note: 
        Soukhov A, Pa´ez A, Higgins CD,Mohamed M (2023) Introducing spatial 
        availability, a singly-constrained measure of competitive accessibility. 
        PLoS ONE 18(1): e0278468. https://doi.org/10.1371/journal.pone.0278468

    """
    result = calc_spatial_access(
            soukhov_2023_c_ij, 
            negative_exp, 
            soukhov_2023_o_j, 
            beta=-0.1
    )
    ref_index=pd.Index(['A', 'B', 'C'], name=INDEX_COLUMNS[0])
    ref_series = pd.Series(
        index=ref_index, 
        data=[27292.176851, 27292.176851, 2240.381587],
        dtype=np.float64
    )
    tm.assert_series_equal(result, ref_series)

def test_soukhov_2023_spatial_availability():
    """ Test contrained access using test values from paper, see note.

    Note: 
        Soukhov A, Pa´ez A, Higgins CD,Mohamed M (2023) Introducing spatial 
        availability, a singly-constrained measure of competitive accessibility. 
        PLoS ONE 18(1): e0278468. https://doi.org/10.1371/journal.pone.0278468

    """
    result = calc_spatial_availability(
            SOUKHOV_2023_C_IJ, 
            negative_exp, 
            o_j=SOUKHOV_2023_O_J, 
            p_i=SOUKHOV_2023_P_I,
            beta=-0.1
    )
    ref_series = pd.Series(
        index=SOUKHOV_TESTS_INDEX, 
        data=[66833.5, 133203.4, 9963.2],
        dtype=np.float64
    )
    tm.assert_series_equal(
        result, ref_series, check_names=False, check_index_type=False)


def test_spatial_heterogeneous_availability():
    """ Test multimodal contrained access vs values from paper.

    Note: 
    Soukhov A, Tarriño-Ortiz J, Soria-Lara JA, Pa´ez A (2024) Multimodal  
    spatial availability: A singly-constrained measure of accessibility
    considering multiple modes. PLoS ONE 19(2): e0299077. 
    https://doi.org/10.1371/journal.pone.0299077

    """
    result = calc_spatial_heterogeneous_availability(
        {
            'x': {
                'p_i': SOUKHOV_2024_P_I_X,
                'c_ij': SOUKHOV_2024_C_IJ_X,
                'impedance_func': negative_exp,
                'beta': -0.1
            }, 
            'z':{
                'p_i': SOUKHOV_2024_P_I_Z,
                'c_ij': SOUKHOV_2024_C_IJ_Z,
                'impedance_func': negative_exp,
                'beta': -0.1
            }
        },
        SOUKHOV_2024_O_J
    )
    ref_result = {
        'x': pd.Series(
            index=SOUKHOV_TESTS_INDEX,
            data=[15696.89, 38170.03, 2035.86]
        ),
        'z': pd.Series(
            index=SOUKHOV_TESTS_INDEX,
            data=[51785.72, 94468.91, 7842.59]
        )
    }
    for k, in ref_result.keys():
        tm.assert_series_equal(
            result[k], 
            ref_result[k], 
            rtol=0.01
        )
