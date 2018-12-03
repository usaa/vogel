import numpy as _np
import pandas as _pd

import matplotlib.pyplot as _plt
_plt.switch_backend('agg')

# from sklearn.utils.testing import assert_array_equal
# from sklearn.utils.testing import assert_array_almost_equal

import vogel.utils as _v_utils
import vogel.preprocessing as _v_prep
import vogel.utils.stats as _v_stats

# Test Data
df = _pd.DataFrame({
      'a': [1., 2., 3., 4.]
    , 'b': [100., 2., 3., _np.nan]
    , 'c': ['texas', 'texas', 'michigan', 'colorado']
    , 'd': ['texas', 'texas', 'michigan', _np.nan]
    , 'e': [1., 1., 1., 1.]
    , 'f': [0., -1., _np.nan, 1.]
})

data_dict = {
    'grp_numeric': ['a', 'b']
  , 'grp_cat': ['c', 'd']
  , 'grp_other': ['a', 'c']
}

#TODO work out issues with this
class TestFindLabelDicts(object):
    def test_ld(self):
        df_encoded = {'c': {'items': ['colorado', 'michigan'], 'hold': 'texas', 'sep': '()'}}
        pipe = _v_utils.make_pipeline(_v_prep.LabelEncoder(feature_filter=['c'], sep='()'))
        out_df = pipe.fit_transform(df)
        print(pipe.named_steps['labelencoder'].feature_filter)
        print(pipe.named_steps['labelencoder'].cat_dict)
        print(out_df)
        labels = _v_utils.find_label_dicts(pipe, {})
        print(labels)
        assert labels == df_encoded
#         assert 1==2
        
class TestPlots(object):
    # no asserts, only test if plot crashes
    def test_pareto(self):
        temp = df['c'].value_counts()
        _v_stats.plot_pareto(temp.values, temp.keys())
    
    def test_one_way_plot(self):
        _v_stats.plot_one_way_fit(df['a'], df['e'])
