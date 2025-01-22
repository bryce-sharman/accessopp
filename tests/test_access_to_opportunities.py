import unittest
import numpy as np
import pandas as pd
from pandas import testing as tm

from accessopp.access_opportunities import within_threshold, negative_exp, gaussian
from accessopp.access_opportunities import calc_spatial_access, calc_spatial_availability, calc_spatial_heterogeneous_availability
from accessopp.access_opportunities import has_opportunity, closest_opportunity, nth_closest_opportunity

class TestAccessOpportunities(unittest.TestCase):

    def setUp(self):
        self.cost_matrix_df = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data = [
                [1, 20, 5],
                [1, 21, 10],
                [1, 22, 15],
                [2, 20, 18],
                [2, 21, 13],
                [2, 22, 8]
            ],
            dtype=np.int32
        )
        self.cost_matrix_df.name = "cost_matrix"


        self.soukhov_2023_p_i = pd.Series(
            index=['A', 'B', 'C'],
            data=[50000, 150000, 10000],
            dtype=np.int64
        )
        self.soukhov_2023_o_j = pd.Series(
            index=[1, 2, 3],
            data=[100000, 100000, 10000],
            dtype=np.int64
        )
        self.soukhov_2023_c_ij = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                ['A', 1,  15],
                ['A', 2,  30],
                ['A', 3, 100],
                ['B', 1,  30],
                ['B', 2,  15],
                ['B', 3, 100],
                ['C', 1, 100],
                ['C', 2, 100],
                ['C', 3,  15]
            ],
        )
        self.soukhov_2023_c_ij.name = "cost_matrix"

        self.soukhov_2024_p_i_x = pd.Series(
            index=['A', 'B', 'C'],
            data=[50000 * 0.33, 150000 * 0.40, 10000 * 0.30],
            dtype=np.int64
        )

        self.soukhov_2024_p_i_z = pd.Series(
            index=['A', 'B', 'C'],
            data=[50000 * 0.67, 150000 * 0.60, 10000 * 0.70],
            dtype=np.int64
        )

        self.soukhov_2024_o_j = pd.Series(
            index=[1, 2, 3],
            data=[100000, 100000, 10000],
            dtype=np.int64
        )


        self.soukhov_2024_c_ij_x = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                ['A', 1,  15],
                ['A', 2,  30],
                ['A', 3, 100],
                ['B', 1,  30],
                ['B', 2,  15],
                ['B', 3, 100],
                ['C', 1, 100],
                ['C', 2, 100],
                ['C', 3,  15]
            ],
        )
        self.soukhov_2024_c_ij_x.name = "cost_matrix"

        self.soukhov_2024_c_ij_z = pd.DataFrame(
            columns=['from_id', 'to_id', 'travel_time'],
            data=[
                ['A', 1, 10],
                ['A', 2, 25],
                ['A', 3, 80],
                ['B', 1, 25],
                ['B', 2, 10],
                ['B', 3, 80],
                ['C', 1, 80],
                ['C', 2, 80],
                ['C', 3, 10]
            ],
        )
        self.soukhov_2024_c_ij_z.name = "cost_matrix"

    def test_impedance_within_threshold_3(self):
        """ Tests for within threshold impedance function with threshold of 3. This should be 0 everywhere. """
        ref_df = pd.DataFrame(
            index = [1, 2],
            columns = [20, 21, 22],
            data = [[0, 0, 0], [0, 0, 0]],
            dtype=np.int32
        )
        ref_df.index.name = "from_id"
        ref_df.columns.name = "to_id"
        result = within_threshold(self.cost_matrix_df, 3)
        tm.assert_frame_equal(result, ref_df, check_index_type=False, check_column_type=False)

    def test_impedance_within_threshold_25(self):
        """ Tests for within threshold impedance function with threshold of 25. This should be 1 everywhere. """
        ref_df = pd.DataFrame(
            index = [1, 2],
            columns = [20, 21, 22],
            data = [[1, 1, 1], [1, 1, 1]],
            dtype=np.int32
        )
        ref_df.index.name = "from_id"
        ref_df.columns.name = "to_id"
        result = within_threshold(self.cost_matrix_df, 25)
        tm.assert_frame_equal(result, ref_df, check_index_type=False, check_column_type=False)

    def test_impedance_within_threshold_10(self):
        """ Tests for within threshold impedance function with threshold of 10."""
        ref_df = pd.DataFrame(
            index = [1, 2],
            columns = [20, 21, 22],
            data = [[1, 1, 0], [0, 0, 1]],
            dtype=np.int32
        )
        ref_df.index.name = "from_id"
        ref_df.columns.name = "to_id"
        result = within_threshold(self.cost_matrix_df, 10)
        tm.assert_frame_equal(result, ref_df, check_index_type=False, check_column_type=False)


    def test_impedance_negative_exp(self):
        """ Test for negative exponential impedance function."""
        # ref matrix was calculated in Excel to avoid using the same code in the test as the implementation
        ref_df = pd.DataFrame(
            index = [1, 2],
            columns = [20, 21, 22],
            data = [[0.60653066, 0.36787944, 0.22313016], [0.16529889, 0.27253179, 0.44932896]],
            dtype=np.float64
        )
        ref_df.index.name = "from_id"
        ref_df.columns.name = "to_id"
        result = negative_exp(self.cost_matrix_df, beta=-0.1)
        tm.assert_frame_equal(result, ref_df, check_index_type=False, check_column_type=False)


    def test_impedance_gaussian(self):
        """ Test for Gaussian impedance function."""
        # ref matrix was calculated in Excel to avoid using the same code in the test as the implementation
        ref_df = pd.DataFrame(
            index = [1, 2],
            columns = [20, 21, 22],
            data = [[0.88249690, 0.60653066, 0.32465247], [0.19789870, 0.42955736, 0.72614904]],
            dtype=np.float64
        )
        ref_df.index.name = "from_id"
        ref_df.columns.name = "to_id"
        result = gaussian(self.cost_matrix_df, sigma=10)
        tm.assert_frame_equal(result, ref_df, check_index_type=False, check_column_type=False)

    def test_dual_has_opportunity_within_threshold_3(self):
        """ Test dual access: has opportunity within threshold cost of 3 minutes. """
        ref_series = pd.Series(
            index=[1, 2], 
            data=[0, 0],
            dtype=np.int32
        )
        result = has_opportunity(self.cost_matrix_df, 3)
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)

    def test_dual_has_opportunity_within_threshold_7(self):
        """ Test dual access: has opportunity within threshold cost of 7 minutes. """
        ref_series = pd.Series(
            index=[1, 2], 
            data=[1, 0],
            dtype=np.int32
        )
        result = has_opportunity(self.cost_matrix_df, 7)
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)

    def test_dual_has_opportunity_within_threshold_10(self):
        """ Test dual access: has opportunity within threshold cost of 10 minutes. """
        ref_series = pd.Series(
            index=[1, 2], 
            data=[1, 1],
            dtype=np.int32
        )
        result = has_opportunity(self.cost_matrix_df, 10)
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)


    def test_dual_closest_opportunity(self):
        """ Test dual access: closest opportunity."""
        ref_series = pd.Series(
            index=[1, 2], 
            data=[5, 8],
            dtype=np.int32
        )
        result = closest_opportunity(self.cost_matrix_df)
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)

    def test_dual_2nd_closest_opportunity(self):
        """Test dual access: second closest opportunity"""
        ref_series = pd.Series(
            index=[1, 2], 
            data=[10, 13],
            dtype=np.int32
        )
        result = nth_closest_opportunity(self.cost_matrix_df, 2)
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)

    def test_primal_gaussian_no_weights(self):
        """ Test the access to opportunities with no weights. """
        ref_series = pd.Series(
            index=[1, 2], 
            data=[
                0.88249690 + 0.60653066 + 0.32465247, 
                0.19789870 + 0.42955736 + 0.72614904],
            dtype=np.float64
        )
        result = calc_spatial_access(
             self.cost_matrix_df, gaussian, o_j=None, p_i=None, 
             normalize="none", sigma=10
        )
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)


    def test_primal_gaussian_dest_weights(self):
        """ Test the access to opportunities with destination weights. """
        destination_weights = pd.Series(data=[1, 3, 5], index=[20, 21, 22], dtype=np.float64)
        ref_series = pd.Series(
            index=[1, 2], 
            data=[
                0.88249690 + 3*0.60653066 + 5*0.3246524, 
                0.19789870 + 3*0.42955736 + 5*0.72614904],
            dtype=np.float64
        )
        result = calc_spatial_access(
             self.cost_matrix_df, gaussian, o_j=destination_weights, p_i=None, 
             normalize="none", sigma=10
        )
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)

    def test_primal_gaussian_dest_weights_reindex_needed(self):
        """ Test the access to opportunities with destination weights, when indices don't match and reindex is needed. """
        # Note that this destination_weights vector intentionally has the wrong final index, which is 25 and not 22.
        destination_weights = pd.Series(data=[1, 3, 5], index=[20, 21, 25], dtype=np.float64)

        ref_series = pd.Series(
            index=[1, 2], 
            data=[
                0.88249690 + 3*0.60653066, 
                0.19789870 + 3*0.42955736],
            dtype=np.float64
        )
        result = calc_spatial_access(
             self.cost_matrix_df, gaussian, o_j=destination_weights, p_i=None, 
             normalize="none", sigma=10
        )
        tm.assert_series_equal(result, ref_series, check_names=False, check_index_type=False)


    def test_primal_gaussian_orig_weights(self):
        """ Test the access to opportunities with origin weights. """
        origin_weights = pd.Series(index=[1, 2], data=[12, 15], dtype=np.float64)
        ref_result = (0.88249690 + 0.60653066 + 0.32465247) * 12 + (0.19789870 + 0.42955736 + 0.72614904) * 15
        result = calc_spatial_access(
             self.cost_matrix_df, gaussian, o_j=None, p_i=origin_weights, 
             normalize="none", sigma=10
        )
        self.assertAlmostEqual(result, ref_result, places=5)

    def test_primal_gaussian_orig_weights_reindex_needed(self):
        """ Test the access to opportunities with origin weights, when origin indices don't match and reindex is needed.. """
        # Note that this origin_weights vector intentionally has the wrong first index, which is 0 and not 1.
        origin_weights = pd.Series(index=[0, 2], data=[12, 15], dtype=np.float64)
        ref_result = (0.19789870 + 0.42955736 + 0.72614904) * 15
        result = calc_spatial_access(
             self.cost_matrix_df, gaussian, o_j=None, p_i=origin_weights, 
             normalize="none", sigma=10
        )
        self.assertAlmostEqual(result, ref_result, places=5)


    def test_primal_gaussian_both_weights(self):
        """ Test the access to opportunities with both origin and destination weights. """
        origin_weights = pd.Series(index=[1, 2], data=[12, 15], dtype=np.float64)
        destination_weights = pd.Series(data=[1, 3, 5], index=[20, 21, 22], dtype=np.float64)
        ref_result = (0.88249690 + 3*0.60653066 + 5*0.32465247) * 12 + \
                     (0.19789870 + 3*0.42955736 + 5*0.72614904) * 15
        result = calc_spatial_access(
             self.cost_matrix_df, gaussian, o_j=destination_weights, p_i=origin_weights, 
             normalize="none", sigma=10
        )
        self.assertAlmostEqual(result, ref_result, places=5)

    def test_soukhov_2023_spatial_access(self):
        """ Test uncontrained (Hansen) access vs values from paper, see note.

        Note: 
        Soukhov A, Pa´ez A, Higgins CD,Mohamed M (2023) Introducing spatial 
        availability, a singly-constrained measure of competitive accessibility. 
        PLoS ONE 18(1): e0278468. https://doi.org/10.1371/journal.pone.0278468

        """
        ref_series = pd.Series(
            index=['A', 'B', 'C'], 
            data=[27292.176851, 27292.176851, 2240.381587],
            dtype=np.float64
        )
        result = calc_spatial_access(
             self.soukhov_2023_c_ij, 
             negative_exp, 
             o_j=self.soukhov_2023_o_j, 
             beta=-0.1
        )
        tm.assert_series_equal(
            result, ref_series, check_names=False, check_index_type=False)

    def test_spatial_availability(self):
        """ Test contrained access vs values from paper.

        Note: 
        Soukhov A, Pa´ez A, Higgins CD,Mohamed M (2023) Introducing spatial 
        availability, a singly-constrained measure of competitive accessibility. 
        PLoS ONE 18(1): e0278468. https://doi.org/10.1371/journal.pone.0278468

        """
        ref_series = pd.Series(
            index=['A', 'B', 'C'], 
            data=[66833.5, 133203.4, 9963.2],
            dtype=np.float64
        )
        result = calc_spatial_availability(
             self.soukhov_2023_c_ij, 
             negative_exp, 
             o_j=self.soukhov_2023_o_j, 
             p_i=self.soukhov_2023_p_i,
             beta=-0.1
        )
        tm.assert_series_equal(
            result, ref_series, check_names=False, check_index_type=False)

    def test_spatial_heterogeneous_availability(self):
        """ Test multimodal contrained access vs values from paper.

        Note: 
        Soukhov A, Tarriño-Ortiz J, Soria-Lara JA, Pa´ez A (2024) Multimodal  
        spatial availability: A singly-constrained measure of accessibility
        considering multiple modes. PLoS ONE 19(2): e0299077. 
        https://doi.org/10.1371/journal.pone.0299077

        """
        ref_result = {
            'x': pd.Series(
                index=['A', 'B', 'C'],
                data=[15696.89, 38170.03, 2035.86]
            ),
            'z': pd.Series(
                index=['A', 'B', 'C'],
                data=[51785.72, 94468.91, 7842.59]
            )
        }
        result = calc_spatial_heterogeneous_availability(
            {
                'x': {
                    'p_i': self.soukhov_2024_p_i_x,
                    'c_ij': self.soukhov_2024_c_ij_x,
                    'impedance_func': negative_exp,
                    'beta': -0.1
                }, 
                'z':{
                    'p_i': self.soukhov_2024_p_i_z,
                    'c_ij': self.soukhov_2024_c_ij_z,
                    'impedance_func': negative_exp,
                    'beta': -0.1
                }
            },
            self.soukhov_2024_o_j 
        )
        for k, in ref_result.keys():
            tm.assert_series_equal(
                result[k], 
                ref_result[k], 
                check_names=False, 
                check_index_type=False,
                rtol=0.01)
