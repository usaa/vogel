import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal

from vogel.preprocessing import ColumnExtractor
from vogel.preprocessing import ConstantColumnRemover
from vogel.preprocessing import Imputer
from vogel.preprocessing import FunctionTransformer
from vogel.preprocessing import LabelEncoder
from vogel.preprocessing import MinMaxScaler
from vogel.preprocessing import NullEncoder
from vogel.preprocessing import QueryTransformer
from vogel.preprocessing import Shuffler
from vogel.preprocessing import FeatureUnion
from vogel.preprocessing import InteractionGroups
from vogel.preprocessing import PolynomialTransform
from vogel.preprocessing import LogTransform
from vogel.preprocessing import Binning
from vogel.preprocessing import GrouperTransformer
from vogel.preprocessing import Interactions

import warnings

# Test Data
df = pd.DataFrame({
      'a': [1., 2., 3., 4.]
    , 'b': [100., 2., 3., np.nan]
    , 'c': ['texas', 'texas', 'michigan', 'colorado']
    , 'd': ['texas', 'texas', 'michigan', np.nan]
    , 'e': [1., 1., 1., 1.]
    , 'f': [0., -1., np.nan, 1.]
})

data_dict = {
    'grp_numeric': ['a', 'b']
  , 'grp_cat': ['c', 'd']
  , 'grp_other': ['a', 'c']
}


class TestColumnExtractor(object):
    def test_numeric_by_group(self):
        tx = ColumnExtractor(cols_or_groups=['grp_other'], data_dict=data_dict, want_numeric=True)
        assert tx.fit_transform(df).columns == ['a']

    def test_numeric_by_group_with_duplicates(self):
        tx = ColumnExtractor(cols_or_groups=['a', 'grp_other'], data_dict=data_dict, want_numeric=True)
        assert tx.fit_transform(df).columns == ['a'] 

    def test_cat_by_grp(self):
        tx = ColumnExtractor(cols_or_groups=['grp_other'], data_dict=data_dict, want_numeric=False)
        assert tx.fit_transform(df).columns == ['c']

    def test_num_to_cat(self):
        tx = ColumnExtractor(cols_or_groups=['grp_other'], data_dict=data_dict, want_numeric=False, num_to_cat=['a'])
        assert_array_equal(tx.fit_transform(df).columns, ['a', 'c'])
        tx = ColumnExtractor(cols_or_groups=['grp_other'], data_dict=data_dict, want_numeric=True, num_to_cat=['a'])
        assert tx.fit_transform(df).shape[1] == 0
       
class TestRemoveConstantColumns(object):
    def test_remove_constant_columns(self):
        assert_array_equal(ConstantColumnRemover().fit_transform(df).columns, ['a', 'b', 'c', 'd', 'f'])
        
class TestImputer(object):
    def test_imputer(self):
        df_imputed = df[['a', 'b']].copy()
        df_imputed.iloc[3, 1] = df['b'].mean()
        assert Imputer().fit_transform(df[['a', 'b']]).equals(df_imputed)

    def test_median_imputation(self):
        df_imputed = df[['a', 'b']].copy()
        df_imputed.iloc[3, 1] = df['b'].median()
        assert Imputer(strategy='median').fit_transform(df[['a', 'b']]).equals(df_imputed)
        
class TestFunctionTransformer(object):
    def test_function_transformer(self):
        def f(X):
            return X.sum()
        tx = FunctionTransformer(f)
        assert tx.fit_transform(df[['a', 'b']]).equals(df[['a', 'b']].sum())
   
class TestLabelEncoder(object):
    def test_encode_cats(self):
        df_encoded = pd.DataFrame({'c__colorado': [0, 0, 0, 1], 
                                   'c__michigan': [0, 0, 1, 0],                                   
                                   'd__michigan': [0, 0, 1, 0], 
                                   'd__nan': [0, 0, 0, 1]})
        tx = LabelEncoder(feature_filter=['c', 'd'], drop='non_trans')
        assert tx.fit_transform(df).equals(df_encoded)
        
    def test_encode_cats_min_value(self):
        df2 = pd.concat([df[['c', 'd']], pd.DataFrame({'c':['texas', 'michigan'], 'd':['texas', 'michigan']})]).reset_index(drop=True)
        tx = LabelEncoder(min_count=2, feature_filter=['c', 'd'], drop='non_trans')
        df_encoded = pd.DataFrame({'c__michigan': [0, 0, 1, 0, 0, 1], 'd__michigan': [0, 0, 1, 0, 0, 1]})
        assert tx.fit_transform(df2).equals(df_encoded)
        

class TestMinMaxScaler(object):
    def test_min_max(self):
        df_rescaled = (df[['a']] - df['a'].min()) / (df['a'].max() - df['a'].min())
        tx = MinMaxScaler()
        assert_array_almost_equal(tx.fit_transform(df[['a']]), df_rescaled)
        
class TestNullEncoder(object):
    def test_null_encoder(self):
        df_encoded = pd.DataFrame({'b': [100., 2., 3., np.nan], 'b_nan': [0, 0, 0, 1]})
        tx = NullEncoder()
        assert tx.fit_transform(df[['b']]).equals(df_encoded)
        
class TestQueryTransformer(object):
    def test_query(self):
        df_filtered = df.query('b > 2.')
        tx = QueryTransformer('b > 2.')
        assert tx.fit_transform(df).equals(df_filtered)
        

class TestShuffler(object):
    def test_shuffler(self):
        tx = Shuffler(random_state=42)
        assert_array_equal(tx.fit_transform(df)['a'], [2., 4., 1., 3.])
        
class TestFeatureUnion(object):
    def test_feature_union(self):
        ce1 = ColumnExtractor(data_dict=data_dict, cols_or_groups=['grp_other'], want_numeric=True)
        ce2 = ColumnExtractor(data_dict=data_dict, cols_or_groups=['grp_numeric'], want_numeric=True)
        tx = FeatureUnion([('p1', ce1), ('p2', ce2)])
        assert tx.fit_transform(df).equals(df[['a', 'a', 'b']])
        
class TestInteractionGroups(object):
    def test_interaction_groups(self):
        tx = InteractionGroups(by=['a', 'e'], features=['b'], drop='non_trans')
        assert list(tx.fit_transform(df).columns) == ['b__a', 'b__e']
        
class TestInteractions(object):
    def test_interactions(self):
        tx = Interactions(feature_filter=['a', 'b', 'e'])
        temp = tx.fit_transform(df)
        print(set(temp.columns))
#         assert set(temp.columns) == set(['c', 'd', 'f', 'a__b', 'a__e', 'b__e'])
        assert temp['a__b'].iloc[0] == 100.0
        
class TestPolynomialTransform(object):
    def test_orthoganal(self):
        tx = PolynomialTransform(poly_type='orthogonal', drop='non_trans', feature_filter=['a']) 
        df_encoded = pd.DataFrame({'a__or^1': [-1.5, -0.5, 0.5, 1.5], 'a__or^2': [1.0, -1.0, -1.0, 1.0]})
        assert tx.fit_transform(df).equals(df_encoded)
        
    def test_power(self):
        tx = PolynomialTransform(poly_type='power', drop='non_trans', feature_filter=['a'])   
        df_encoded = pd.DataFrame({'a^2': [1.0, 4.0, 9.0, 16.0]})
        assert tx.fit_transform(df).equals(df_encoded)

class TestLogTransform(object):
    def test_log(self):
        tx = LogTransform(drop='non_trans', feature_filter=['a'])   
        df_encoded = pd.DataFrame({'a__ln': [0.0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906]})
        assert tx.fit_transform(df).equals(df_encoded)
    
    def test_log_bad_col(self):
        with warnings.catch_warnings(record=True) as ws:
            tx = LogTransform(fill_0=True, drop='none')   
            df_encoded = pd.DataFrame({'f': [0.0, -1.0, np.nan, 1.0], 'f__ln': [0.0, 0.0, 0.0, 0.0]})
            assert tx.fit_transform(df[['f']].copy()).equals(df_encoded)

            assert ws[0].message.args[0] == 'divide by zero encountered in log'
            assert ws[1].message.args[0] == 'invalid value encountered in log'
        
class TestBinning(object):
    def test_qcut_stand(self):
        tx = Binning(bin_type='qcut', bins=3, drop='non_trans', feature_filter=['a'])
        df_encoded = pd.DataFrame({'a_q_g3': ['(-inf, 2.0]', '(-inf, 2.0]', '(2.0, 3.0]', '(3.0, inf]']})
        assert tx.fit_transform(df[['a']]).astype('str').equals(df_encoded.astype('str'))
    
    def test_cut_mean(self):
        tx = Binning(bin_type='cut', bins=3, bin_id='mean', drop='non_trans', feature_filter=['a'])
        df_encoded = pd.DataFrame({'a_c_g3': [1.5, 1.5, 3.0, 4.0]})
        assert tx.fit_transform(df).astype('str').equals(df_encoded.astype('str'))
        
    def test_ManhaveSSM_wavg(self):
        tx = Binning(bin_type='ManhaveSSM', bins=3, bin_id='wavg', weight=df['a'], drop='non_trans', feature_filter=['a'])
        df_encoded = pd.DataFrame({'a_M_g3': [1.6666666666666667, 1.6666666666666667, 3.0, 4.0]})
        assert tx.fit_transform(df).astype('str').equals(df_encoded.astype('str'))
                     
    def test_qcut_stand_zero(self):
        tx = Binning(bin_type='qcut', bins=3, zero_bucket=True, drop='non_trans', feature_filter=['a'])
        df_encoded = pd.DataFrame({'a_q_g3': ['(0.0, 2.5]', '(0.0, 2.5]', '(2.5, inf]', '(2.5, inf]']})
        assert tx.fit_transform(df).astype('str').equals(df_encoded.astype('str'))
        
    def test_overrides(self):
        bins = {
            'a' : {
                'cutoffs' : [-np.inf, 2, 3, np.inf],
                'ids' : ['<=2', '3', '>3']
            }
        }
        tx = Binning(overrides=bins, drop='non_trans', feature_filter=['a'])
        df_encoded = pd.DataFrame({'a_cust': ['<=2', '<=2', '3', '>3']})
        assert tx.fit_transform(df).astype('str').equals(df_encoded.astype('str'))
        
class TestGrouper(object):
    def test_grouper(self):
        group_dict = {"c" : {"michigan":"other", "colorado":"other"}}
        tx = GrouperTransformer(group_dict)
        df_encoded = pd.DataFrame({'c': ['texas', 'texas', 'other', 'other']})
        assert tx.fit_transform(df[['c']].copy()).astype('str').equals(df_encoded.astype('str'))
