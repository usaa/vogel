import numpy as np
import pandas as pd

import types
import functools

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion

import itertools
import json
import math


def _droper(feature_names, drop, all_cols, features_to_transform):
    
    if drop == 'none':
        feature_names = list(all_cols) + feature_names 
    elif drop == 'replace':
        feature_names = [x for x in list(all_cols) if x not in features_to_transform] + feature_names 
    elif drop == 'non_trans':
        pass
    else:
        raise ValueError('{} is not suppoertd as a drop type.'.format(drop))

    return feature_names

class ColumnExtractor(TransformerMixin):
    """Select columns from Pandas DataFrame"""
    """
        cols_or_groups:
            list: list of column or group names. If Null, it takes all columns based on is_numeric
        data_dict: ({})
            dict: dictionary of data groups. If Null, it takes cols_or_groups as column names
        want_numeric: (True)
            bool: limmit selection of columns by type, numeric. If "None", don't check type (return all).
        num_to_cat: ([])
            list: list of numeric type columns that should be treated as catigorical 
    """
    def __init__(self, cols_or_groups, data_dict={}, want_numeric=None, num_to_cat=[]):
        self.data_dict = data_dict
        self.cols_or_groups = cols_or_groups
        self.want_numeric = want_numeric
        self.num_to_cat = num_to_cat
        self.feature_names = []
    
    def unique_everseen(self, iterable, key=None):
        "List unique elements, preserving order. Remember all elements ever seen."
        # unique_everseen('AAAABBBCCDAABBB') --> A B C D
        # unique_everseen('ABBCcAD', str.lower) --> A B C D
        seen = set()
        seen_add = seen.add
        if key is None:
            for element in itertools.filterfalse(seen.__contains__, iterable):
                seen_add(element)
                yield element
        else:
            for element in iterable:
                k = key(element)
                if k not in seen:
                    seen_add(k)
                    yield element

    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        out_cols_temp = []
        if self.cols_or_groups is None:
            out_cols_temp = X.columns
        else:
            if self.data_dict is None:
                out_cols_temp = self.cols_or_groups
            else:
                for col in self.cols_or_groups:
                    # get col dict from json
                    try:
                        out_cols_temp += self.data_dict[col]
                    except:
                        out_cols_temp.append(col)
        
        # test if col matches wanted type
        for col in self.unique_everseen(out_cols_temp): 
            if self.want_numeric is None:
                self.feature_names.append(col)
            else:
                if (not self.want_numeric) and (col in self.num_to_cat):
                        self.feature_names.append(col)
                elif np.issubdtype(X[col], np.number)  == self.want_numeric:
                    if (self.want_numeric) and (col not in self.num_to_cat):
                        self.feature_names.append(col)
                    elif not self.want_numeric:
                        self.feature_names.append(col)
        return self
    
    def transform(self, X):
        return X[self.feature_names].copy()
    

class ConstantColumnRemover(TransformerMixin):
    """Drops constant columns from Pandas DataFrame"""
    def __init__(self):
        self.drop_cols = None
        
    def get_feature_names(self):
        return self.feature_names
    
    def fit(self, X, y=None):
        self.drop_cols = X.columns[X.nunique(dropna=False) == 1]
        print("Droped Columns:", self.drop_cols)
        self.feature_names = [col for col in X.columns if col not in self.drop_cols]
        return self
    
    def transform(self, X):
        return X[self.feature_names]

    
class Imputer(SimpleImputer):
    """scikit-learn Imputer transform with pandas output
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    """
    def __init__(self, missing_values=np.nan, strategy="mean",
                 axis=0, verbose=0, copy=True):
        super().__init__(missing_values, strategy,
                 axis, verbose, copy)
        self.feature_names = None
    
    def get_feature_names(self):
        return self.feature_names
    
    def fit(self, X, y=None):
        super().fit(X)
        self.feature_names = list(X.columns)
        return self
    
    def transform(self, X):
        out_np = super().transform(X)
        return pd.DataFrame(out_np, columns=self.feature_names)
    
    
class FunctionTransformer(TransformerMixin):
    """
    Transforms based on function passed in.
    Warning: columns names may not capture all function changes before 
    a transform is performed on a pipeline.
    """
    def __init__(self, func):
#         def copy_func(f):
#             """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
#             g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__ + 'copy',
#                                    argdefs=f.__defaults__,
#                                    closure=f.__closure__)
#             g = functools.update_wrapper(g, f)
#             g.__kwdefaults__ = f.__kwdefaults__
#             return g
        
#         self.func = copy_func(func)
#         self.func.__name__ = 'temp_func'
        self.func = func

        self.feature_names = []
        
    def get_feature_names(self):
        return self.feature_names
    
    def fit(self, X, y=None):
        self.feature_names = X.columns
        return self
    
    def transform(self, X):
        X = self.func(X)
        self.feature_names = X.columns
        return X
    
class LabelEncoder(TransformerMixin):
    """Similar to pandas get_dummies with pandas dataframe output
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    When this transformer is fit on a training data set separate columns
    are created for each level of each variable (null is treated as a 
    separate level).  However, the (a) level with the most occurrences is
    left out for each variable.
    
    When this tranformer is applied to a new data set, any new levels (not
    appearing in the original dataset used for fitting) will be treated as
    the left out level (one with the most occurrences).
    
    Args:
        min_count: (None) 
            int: Minimum number of occurrences to create new
            dummy feature for a level
        sep: ("__")
            string: Select how the feature name and value will be saved as a new column name
            options: 
                "__" [feature__value],
                "()" [feature (value)],
                "."  [feature.value]
        overrides: ({})
            dict: Set base level of feature. 
            {feature : base}
        drop: ("replace") 
            "replace": keep all features but the old transformed. Replaces feature  
            "none": keep all features
            "non_trans" : Keep only the new transformed features
        feature_filter: (None)
            list: list of features to work on 
    """
    def __init__(self, min_count=None, sep="__", overrides = {}, drop = 'replace', feature_filter = None):
        self.cat_dict = {}
        self.min_count = min_count
        self.sep = sep
        self.overrides = overrides
        self.drop = drop
        self.feature_filter = feature_filter
        self.feature_names = []
        
    def get_feature_names(self):
        return self.feature_names
        
    def get_all_cats_minus_largest(self, X, col):
        val_cnts = X[col].value_counts(dropna=False)
        base = None
        # Check for overrides
        if col in self.overrides:
            base = self.overrides[col]
            val_cnts = val_cnts.drop(base)
        else:
            # Leave out max value       
            val_cnts = pd.DataFrame(val_cnts).reset_index().sort_values([col, 'index'], ascending=[False, True])
            base = val_cnts['index'].iloc[0]
            val_cnts = val_cnts.set_index('index')[col].iloc[1:]
            del val_cnts.index.name
        return val_cnts, base
    
    def _formater(self, col, cat):
        if self.sep == "__":
            return '{0}__{1}'.format(col, cat)
        elif self.sep == "()":
            return '{0} ({1})'.format(col, cat)
        elif self.sep == ".":
            return '{0}.{1}'.format(col, cat)
        else:
            raise "Format {} not supported. Change sep arg.".format(self.sep)
    
    def fit(self, X, y=None):
        features_to_transform = None
        if self.feature_filter is None:
            features_to_transform = X.columns
        else:
            features_to_transform = self.feature_filter
        for col in features_to_transform:
            val_cnts, hold_out = self.get_all_cats_minus_largest(X, col)
            cat_list = []
            for cat, count in val_cnts.iteritems():
                new_col = self._formater(col, cat)
                if self.min_count is None:
                    cat_list.append(cat)
                    self.feature_names.append(new_col)
                elif count >= self.min_count:
                    cat_list.append(cat)
                    self.feature_names.append(new_col)
            self.cat_dict[col] = {}
            self.cat_dict[col]['items'] = cat_list
            self.cat_dict[col]['hold'] = hold_out
            
        self.feature_names = _droper(self.feature_names, self.drop, X.columns, features_to_transform)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, cats in self.cat_dict.items():
            for cat in cats['items']:
                cat_name = self._formater(col, cat)
                if pd.isna(cat):
                    # Separate handling when cat is missing
                    X[cat_name] = np.where(X[col] != X[col], 1, 0)
                else:
                    X[cat_name] = np.where(X[col] == cat, 1, 0)

        return X[self.feature_names]
    
class MinMaxScaler(MinMaxScaler):
    """scikit-learn MinMaxScaler transform with pandas output
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    """
    
    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(feature_range, copy)
        self.feature_names = None
    
    def get_feature_names(self):
        return self.feature_names
    
    def fit(self, X, y=None):
        super().fit(X)
        self.feature_names = list(X.columns)
        return self
    
    def transform(self, X):
        out_np = super().transform(X)
        return pd.DataFrame(out_np, columns=self.feature_names)

    
class NullEncoder(TransformerMixin):
    """Create dummy columns for null features
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    Arrgs:
        min_count: (0)
            int: minimum number of nulls needed in a feature to create a dummy collumn for nulls
    
    ...
    """
    def __init__(self, min_count=0):
        self.features_with_nulls = []
        self.min_count = min_count
        self.feature_names = []
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        self.feature_names = list(X.columns)
        for col in X.columns:
            if X[col].isnull().sum() > self.min_count:
                self.features_with_nulls.append(col)
        
        # Add column names for dummy columns
        for col in self.features_with_nulls:
            idx = self.feature_names.index(col)
            self.feature_names = self.feature_names[:(idx+1)] + [col + '_nan'] + self.feature_names[(idx+1):]
        
        return self
    
    def transform(self, X):
        temp_df = X.copy()
        for col in self.features_with_nulls:
            temp_df[col + '_nan'] = pd.to_numeric(np.where(
                X[col].isnull()
                , 1
                , 0
            ))

        return temp_df    
    
    
class QueryTransformer(TransformerMixin):
    """Transforms a Pandas DataFrame by evaluating a query."""
    def __init__(self, expr=None):
        self.expr = expr
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        self.feature_names = X.columns
        return self
        
    def transform(self, X):
        return X.query(self.expr)

    
class Shuffler(TransformerMixin):
    """Shuffles data"""
    def __init__(self, random_state):
        self.random_state = random_state
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        self.feature_names = X.columns
        return self

    def transform(self, X):
        return X.sample(frac=1., random_state=self.random_state).reset_index(drop=True)
    
class FeatureUnion(FeatureUnion):
    """scikit-learn FeatureUnion transform with pandas output
    
    Feature name should not start with '__'!
   
    Modified get_feature_names method to remove transformer name prefix
    which the FeatureUnion transformer adds.
    """
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self._validate_transformers()
        
    def get_feature_names(self):
        # remove prefix added by FeatureUnion get_feature_name method
        feature_names = super().get_feature_names()
        return [x[x.find('__')+2:] for x in feature_names]
        
    def transform(self, X):
        out_np = super().transform(X)
        return pd.DataFrame(out_np, columns=self.get_feature_names())
    
    def fit_transform(self, X, y=None):
        out_np = super().fit_transform(X, y)
        return pd.DataFrame(out_np, columns=self.get_feature_names())

class Binning(TransformerMixin):
    """Provides multiple binning methods
    
    If a feature has less than 3 values, it will not be transformed. -inf & inf appended to begening and end of cutoffs.
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    
    Args:
        bin_type: (qcut) 
            "qcut": equal sized bins
            , "cut": equal range bins
            , "ManhaveSSM": This code takes the last quantile (quantiles are determined based on record counts, not weights) and puts it in its own bucket. The rest of the data is binned into equal width buckets. This approach should preserve the underlying distribution of the data, while not allowing the extreme values to adversely impact the bucket creation.
            
        bins: (10) 
            int: number of bins to return
            
        bin_id: ("stand") 
            "stand": standard bin count id
            , "mean": mean of bin
            , "wavg": weighted avg using defined weight feature
            , "min": min of bin            
        duplicates: ('drop')
            "drop": ignore dups
            "raise": If bin edges are not unique, raise ValueError or drop non-uniques.
            
        drop: ("replace") 
            "replace": keep all features but the old transformed. Replaces feature  
            "none": keep all features
            "non_trans" : Keep only the new transformed features
        overrides: ({})
            dict: overrides given binning method
            example:
            bins = {
                        'NPC_5_YR_TOT_QTY': {
                            'cutoffs': [-np.inf, 0, 1, 2, np.inf],
                            'ids': ['0', '1', '2', '3+']
                        }
                    }
        feature_filter: (None)
            list: list of features to work on 
    """
    def __init__(self, bin_type = 'qcut', bins=10, bin_id='stand'
                 , duplicates='drop'
                 , drop='replace'
                 , zero_bucket=False
                 , weight=None
                 , overrides={}
                 , feature_filter=None
                ):
        self.bin_dict = {}
        self.bin_type = bin_type
        self.bins = bins
        self.bin_id = bin_id
        self.duplicates = duplicates
        self.drop = drop
        self.weight = weight
        self.feature_names = []
        self.zero_bucket = zero_bucket
        self.overrides = overrides
        self.feature_filter = feature_filter
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        def weighted_average(var, weight):
            if np.sum(weight) == 0:
                return 0
            else:
                return np.sum(var * weight) / np.sum(weight)
            
        def manhaveSSM(col, temp_df, temp_bins):
            manhave_min_matemp_df = []
            _, arr_quantiles = pd.qcut(temp_df[col], q=100, duplicates=self.duplicates, retbins=True)
            manhave_min_matemp_df = [arr_quantiles[-2], arr_quantiles[-1]]

            df_limited = temp_df[temp_df[col] <= manhave_min_matemp_df[0]]
            cut, cutoffs = pd.cut(df_limited[col], temp_bins-1, retbins=True)
            cutoffs = np.append(cutoffs, manhave_min_matemp_df[1])
            cut, cutoffs = pd.cut(temp_df[col], cutoffs, retbins=True)
            df_limited=None
            
            return cut, cutoffs
            
        def set_col_bins(col, X):
            self.bin_dict[col] = {}
            
            # test if col only has 2 or less values
            if X[col].nunique(dropna=False) <=2:
                self.bin_dict[col]['cutoffs'] = np.array([-np.inf,  np.inf])
                self.bin_dict[col]['ids'] = ['(-inf, inf]']

            else:
                temp_df = None
                temp_bins = None
                if self.zero_bucket:
                    if self.bin_id == 'wavg':
                        temp_df = X[[col, 'weight']][X[col]>0].copy()
                    else:
                        temp_df = X[[col]][X[col]>0].copy()
                    temp_bins = self.bins - 1
                else:
                    if self.bin_id == 'wavg':
                        temp_df = X[[col, 'weight']].copy()
                    else:
                        temp_df = X[[col]].copy()
                    temp_bins = self.bins

                # get cutoffs
                cut, cutoffs = None, None

                bin_range = list(range(1, temp_bins+1))

                if self.bin_type == 'qcut':
                    cut, cutoffs = pd.qcut(temp_df[col], q=temp_bins, duplicates=self.duplicates, retbins=True)
                elif self.bin_type == 'cut':
                    cut, cutoffs = pd.cut(temp_df[col], bins=temp_bins, duplicates=self.duplicates, retbins=True)
                elif self.bin_type == 'ManhaveSSM':
                    cut, cutoffs = manhaveSSM(col, temp_df, temp_bins)

                # save cutoffs
                self.bin_dict[col]['cutoffs'] = cutoffs
                
                if self.zero_bucket:
                    self.bin_dict[col]['cutoffs'] = np.insert(self.bin_dict[col]['cutoffs'], 1, 0)

                # remplace matemp_df and min with -/+ infinity
                self.bin_dict[col]['cutoffs'][0]  = -np.inf
                self.bin_dict[col]['cutoffs'][-1] = np.inf

                if self.zero_bucket:
                    self.bin_dict[col]['cutoffs'][1]  = 0
                    
        def set_col_ids(col, X):
            cut = pd.cut(X[col], self.bin_dict[col]['cutoffs'])
            
            if self.bin_id == 'stand':
                self.bin_dict[col]['ids'] = []
                for x in range(0, len(self.bin_dict[col]['cutoffs'])-1):
                    self.bin_dict[col]['ids'].append(('({0}, {1}]').format(self.bin_dict[col]['cutoffs'][x], self.bin_dict[col]['cutoffs'][x+1]))
            else:
                temp_mean = None
                cut.name = 'temp'
                if self.bin_id == 'wavg':
                    Y = pd.concat([X[[col, 'weight']], cut], axis=1)
                else:
                    Y = pd.concat([X[[col]], cut], axis=1)
                if self.bin_id == 'mean':
                    temp_mean = Y[[col, 'temp']].groupby('temp').mean().fillna(0).iloc[:,0]
                elif self.bin_id == 'wavg':
                    temp_mean = Y[[col, 'temp', 'weight']].groupby('temp').apply(lambda x: weighted_average(x[col],x['weight']))
                elif self.bin_id == 'median':
                    temp_mean = Y[[col, 'temp']].groupby('temp').median().fillna(0).iloc[:,0]
                elif self.bin_id == 'min':
                    temp_mean = Y[[col, 'temp']].groupby('temp').min().fillna(0).iloc[:,0]
                else:
                    raise('Unkown bin_id type!')


                self.bin_dict[col]['ids'] = temp_mean.values.tolist()
                    
                # Fix if feature has less values then (This shouldn't be needed)
                self.bin_dict[col]['ids'] = self.bin_dict[col]['ids'][0:len(self.bin_dict[col]['cutoffs'])-1]

                # Handel Zeros for id names. 
                for i in range(1, len(self.bin_dict[col]['ids'])):
                    if self.bin_dict[col]['ids'][i] == 0:
                        avg_of_bins = (self.bin_dict[col]['cutoffs'][i] + self.bin_dict[col]['cutoffs'][i+1])/2
                        self.bin_dict[col]['ids'][i] = avg_of_bins
                        
        # set up weight for weighted avg                    
#         if self.bin_id == 'wavg':
#             if self.weight is None:
#                 raise('Weight must be provided if using wavg')
# #             print(X.shape, self.weight.shape)
#             print(X.columns)
# #             print(self.weight[self.weight.index.duplicated()])
#             X['weight'] = self.weight
            
        self.features_to_transform = None
        if self.feature_filter is None:
            self.feature_filter = []
            self.features_to_transform = X.columns
        else:
            self.features_to_transform = self.feature_filter
            
        for col in self.features_to_transform:
            if col != 'weight':
                
                if col in self.overrides:
                    self.bin_dict[col] = {}
                    self.bin_dict[col]['cutoffs'] = np.array(self.overrides[col]['cutoffs'])
                    self.bin_dict[col]['ids'] = self.overrides[col]['ids']
                    
                    self.feature_names.append('{0}_{1}'.format(col, 'cust'))
                else:
                    temp_X = X[[col]].copy()

                    if self.bin_id == 'wavg':
                        if self.weight is None:
                            raise('Weight must be provided if using wavg')
                        temp_X['weight'] = self.weight

                    set_col_bins(col, temp_X)
                    set_col_ids(col, temp_X)
 
                    self.feature_names.append('{0}_{1}_g{2}'.format(col, self.bin_type[0], self.bins))
                
        if 'weight' in X.columns:
            X.drop('weight', 1, inplace=True)
            
        self.feature_names = _droper(self.feature_names, self.drop, X.columns, self.features_to_transform)
            
        return self
    
    def transform(self, X):
        out_df = X.copy()
        for col in self.bin_dict:
            if col in self.overrides:
                bin_name = '{0}_{1}'.format(col, 'cust')
            else:
                bin_name = '{0}_{1}_g{2}'.format(col, self.bin_type[0], self.bins)
            
            # Set as original value if binary 
            if len(self.bin_dict[col]['ids']) == 1:
                out_df[bin_name] = out_df[col]
            else:
                out_df[bin_name] = pd.cut(X[col], bins=self.bin_dict[col]['cutoffs'], labels=self.bin_dict[col]['ids'])
            
            if self.bin_id != 'stand':
                try:
                    out_df[bin_name] = pd.to_numeric(out_df[bin_name])
                except:
                    print('{} not converted to Numeric.'.format(bin_name))

        return out_df[self.feature_names]
    
class LogTransform(TransformerMixin):
    """Create ln transform columns for features
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    Arrgs:
        pre_add: (2)
            float: number to add before log transform
        fill_0: (False)
            bool: add 1 to features before log transform
        drop: ("replace") 
            "replace": keep all features but the old transformed. Replaces feature  
            "none": keep all features
            "non_trans" : Keep only the new transformed features
    """
    def __init__(self, pre_add=0, fill_0=False, drop='replace', feature_filter=None):
        self.features_to_transform = []
        self.pre_add = pre_add
        self.fill_0 = fill_0
        self.drop = drop
        self.feature_names = []
        self.feature_filter = feature_filter
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        if self.feature_filter is None:
            # Don't transfrom features with >= 2 values
            self.features_to_transform = X.columns[X.nunique(dropna=False) > 2]
        else:
            self.features_to_transform = self.feature_filter
            
        self.feature_names = _droper(self.feature_names, self.drop, X.columns, self.features_to_transform)
            
        for col in list(self.features_to_transform):
            self.feature_names.append('{0}__{1}'.format(col, 'ln'))
        
        return self
    
    def transform(self, X):

        for col in self.features_to_transform:
            new_col = '{0}__{1}'.format(col, 'ln')
            
            X[new_col] = X[col] + self.pre_add
            
            if self.fill_0:
                X[new_col] = pd.to_numeric(np.where(
                    X[new_col] < 0
                    , 0
                    , X[new_col]
                ))
                X[new_col] = np.log(X[new_col]).fillna(0)
                X[new_col][X[new_col] == -math.inf] = 0
            else:
                X[new_col] = np.log(X[new_col])

        return X[self.feature_names]
    
class PolynomialTransform(TransformerMixin):
    """Create Polynomial columns for features
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    Arrgs:
        levels: (2)
            int: number of interactions
        poly_type: (power)
            string: [power, orthogonal]
        weight: (None)
            array: weight values for orthogonal transforms. defaults to array of ones
        drop: ("replace") 
            "replace": keep all features but the old transformed. Replaces feature  
            "none": keep all features
            "non_trans" : Keep only the new transformed features
        override: (None)
            dict: overrides fit settings. Should be the same format as the trans_dict
            {feature: {level : {'poley' : polynomial, 'name' : 'new_name'}}}
    """
    def __init__(self, levels=2, poly_type='power', drop='replace', weight=None, override=None, feature_filter=None):
        self.features_to_transform = []
        self.levels = levels
        self.poly_type = poly_type
        self.weight = weight
        self.drop = drop
        self.feature_names = []
        self.override = override
        self.trans_dict = {}
        self.feature_filter = feature_filter
        
    def get_feature_names(self):
        return self.feature_names 
        
    def fit(self, X, y=None):
        def polynomial_gs(w, x):  
            def inner_product(p1, p2):
                return np.sum(w * p1(x) * p2(x))

            def gs_coefficient(p1, p2):
                return inner_product(p1, p2) / inner_product(p1, p1)

            def proj(p1, p2):
                return gs_coefficient(p1, p2) * p1

            def gs(X):
                Y = []
                for i in range(len(X)):
                    temp_vec = X[i]
                    for inY in Y:
                        proj_vec = proj(inY, X[i])
                        temp_vec = temp_vec - proj_vec
                    Y.append(temp_vec)
                return Y

            return gs

        def get_polynomial_trans(x, w, poly_order):
            trans_funcs = []
            ps = []
            coef_vecs = list(np.identity(poly_order))
            for coef_vec in coef_vecs:
                ps.append(np.polynomial.polynomial.Polynomial(coef_vec))
            trans_funcs = polynomial_gs(w,x)(ps)[1:]

            return trans_funcs

        if self.feature_filter is not None:
            self.features_to_transform = self.feature_filter
        else:
            self.features_to_transform = list(X.columns)
            
        

        if self.override is None:
            # set array of ones is weight is missing 
            if (self.poly_type == 'orthogonal') & (self.weight is None):
                    self.weight = np.ones(len(X))

            for col in list(self.features_to_transform):
                orth_polys = None
                if self.poly_type == 'orthogonal':
                    orth_polys = get_polynomial_trans(X[col], self.weight, self.levels+1)
                    self.trans_dict[col] = {}
                    self.trans_dict[col][1] = {}
                    self.trans_dict[col][1]['poly'] = orth_polys[0]
                    self.trans_dict[col][1]['name'] = '{0}__or^1'.format(col)
                    self.feature_names.append('{0}__or^1'.format(col))
                for poly in range(2, self.levels+1):
                    if self.poly_type == 'power':
                        self.feature_names.append('{0}^{1}'.format(col, poly))
                    elif self.poly_type == 'orthogonal':
                        self.feature_names.append('{0}__or^{1}'.format(col, poly))
                        self.trans_dict[col][poly] = {}
                        self.trans_dict[col][poly]['poly'] = orth_polys[poly-1]
                        self.trans_dict[col][poly]['name'] = '{0}__or^{1}'.format(col, poly)
                        
                    else:
                        raise('{} Not a recognized type.'.format(poly_type))
        else:
            self.trans_dict = self.override
            for col, col_dict in self.trans_dict.items():
                for lvl, lvl_dict in col_dict.items():
                    self.feature_names.append(lvl_dict['name'])

        self.feature_names = _droper(self.feature_names, self.drop, X.columns, self.features_to_transform)
            
        return self
    
    def transform(self, X):
        if self.poly_type == 'power':
            for col in list(self.features_to_transform):
                for poly in range(2, self.levels+1):
                    X['{0}^{1}'.format(col, poly)] = np.power(X[col], poly)
        elif self.poly_type == 'orthogonal':
            for col, col_dict in self.trans_dict.items():
                for lvl, lvl_dict in col_dict.items():
                    X[lvl_dict['name']] = lvl_dict['poly'](X[col])              

        return X[self.feature_names]
    
class Interactions(TransformerMixin):
    """Create interaction columns for features
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    Arrgs:
        levels: (2)
            int: number of interactions
        drop: ("replace") 
            "replace": keep all features but the old transformed. Replaces feature  
            "none": keep all features
            "non_trans" : Keep only the new transformed features
    """
    def __init__(self, levels=2, drop='replace', feature_filter=None):
        self.features_to_transform = []
        self.levels = levels
        self.drop = drop
        self.feature_names = []
        self.feature_filter = feature_filter
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        if self.feature_filter is not None:
            self.features_to_transform = self.feature_filter
        else:
            self.features_to_transform = list(X.columns)
            
        self.feature_names = _droper(self.feature_names, self.drop, X.columns, self.features_to_transform)
            
        for inter_col in itertools.combinations(self.features_to_transform, self.levels):
            self.feature_names.append('{0}__{1}'.format(inter_col[0], inter_col[1]))
        
        return self
    
    def transform(self, X):
#         temp_df = pd.DataFrame()
            
        for inter_col in itertools.combinations(self.features_to_transform, self.levels):
            X['{0}__{1}'.format(inter_col[0], inter_col[1])] = X[inter_col[0]] * X[inter_col[1]]
            
        return X[self.feature_names]
        
class InteractionGroups(TransformerMixin):
    """Create interaction columns for features in by group.
    a = ['foo', 'melon', 'a']
    b = [True, False]
    [('foo', True),
     ('foo', False),
     ('melon', True),
     ('melon', False),
     ('a', True),
     ('a', False)]
   
    Added get_feature_names function to be able to pass
    feature names to a customized FeatureUnion transformer.
    
    Arrgs:
        by:
            list: features to be grouped by 
        features: (None)
            list: featurees to group (if None use all features not in "by")
        drop: ("replace") 
            "replace": keep all features but the old transformed. Replaces feature  
            "none": keep all features
            "non_trans" : Keep only the new transformed features
    """
    def __init__(self, by, features=None, drop='replace'):
        self.features_to_transform = []
        self.by = by
        self.features = features
        self.drop = drop
        self.feature_names = []
        
    def get_feature_names(self):
        return self.feature_names
        
    def fit(self, X, y=None):
        self.features_to_transform = list(X.columns)
            
        if self.features is None:
            self.features = [x for x in X.columns if x not in self.by]
            
        for inter_col in list(itertools.product(self.features, self.by)):
            self.feature_names.append('{0}__{1}'.format(inter_col[0], inter_col[1]))
            
        self.feature_names = _droper(self.feature_names, self.drop, X.columns, self.features_to_transform)
        
        return self
    
    def transform(self, X):
            
        for inter_col in list(itertools.product(self.features, self.by)):
            X['{0}__{1}'.format(inter_col[0], inter_col[1])] = X[inter_col[0]] * X[inter_col[1]]
            
        return X[self.feature_names]
        
class GrouperTransformer(TransformerMixin):
    """
        Groups values of a column
    Arrgs:
    
    group_dict:
        dict: dict of features to group & groupings
            Example: {"feature_a" : {"a":"a", "b":"a"}, "feature_b" : {0:np.nan}}
    how: ('replace')
        how grouper groups
        'replace': missing values are set at previous value
        'map': missing values are set to None
    
    """
    def __init__(self, group_dict, how='replace'):
        self.group_dict = group_dict
        self.how = how
        
    def get_feature_names(self):
        return self.feature_names
    
    def fit(self, X, y=None):
        self.feature_names = list(X.columns)
        return self
    
    def transform(self, X):
        for key, val in self.group_dict.items():
            if self.how == 'replace':
                X = X.replace({key : val})
            elif self.how == 'map':
                X[key] = X[key].map(val) 
        return X
