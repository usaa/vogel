import numpy as np
import pandas as pd

import vogel.preprocessing as v_prep
import vogel.utils as v_utils
import vogel.utils.stats as v_stats
import vogel.train as v_train

import statsmodels.api as sm

import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

pipeline = v_utils.make_pipeline(
    v_prep.FeatureUnion([
        ('numeric', v_utils.make_pipeline(
            v_prep.ColumnExtractor(['grp_numeric', 'grp_cat', 'grp_other'], data_dict, want_numeric = True),
            v_prep.Imputer()
        )),
        ('cats', v_utils.make_pipeline(
            v_prep.ColumnExtractor(['grp_numeric', 'grp_cat', 'grp_other'], data_dict, want_numeric = False),
            v_prep.LabelEncoder()
        ))
    ])
)

model_data = pipeline.fit_transform(df)

pd.testing.assert_frame_equal(model_data, pd.DataFrame({'a': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
    'b': {0: 100.0, 1: 2.0, 2: 3.0, 3: 35.0},
    'c__colorado': {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
    'c__michigan': {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
    'd__michigan': {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
    'd__nan': {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}})
)

class TestStatsModels(object):
    def test_sm_simple_reg(self):
        run_list = [{
            'model_type': v_train.V_SM_GLM,
            'model_name': 'SM',
            'model_params': {
            },
            'fit_params': {
            }
        }]

        train_data_dict = {'X': model_data[['c__colorado','c__michigan','d__michigan','d__nan']], 'y': model_data['a']}

        model_runner = v_train.ModelRunner('reg', run_list, 
                                           train_data_dict,
                                           None,
                                           pipeline)

        sm_eval = model_runner.evaluate_models()

        assert round(sm_eval['rmse'].iloc[0], 4) == 0.3536
        assert round(sm_eval['mae'].iloc[0], 2) == 0.25
        assert round(sm_eval['gini'].iloc[0], 2) == .25
#         assert round(sm_eval['weighted_rmse'].iloc[0], 4) == 0.3536
        
        # Plot test have no asserts, checking only for crashing
        
        model_runner.plot_models(False)

        model_runner.models[0].plot_glm_one_way_fit()
        model_runner.models[0].plot_glm_one_way_fit(plot_error=False,
                             round_at=1,
                             rotate_x=False,
                             predicted_value=False,
                             scale_to_zero=True,
                             base_line=False,
                             pad_bar_chart=False)
        
    def test_sm_adv_tweedie(self):
        run_list = [{
            'model_type': v_train.V_SM_GLM,
            'model_name': 'SM',
            'model_params': {
                'custom_weight': model_data['b'],
                'family': sm.families.Tweedie(var_power=1.5)
            },
            'fit_params': {
                'maxiter': 10
            }
        }]

        train_data_dict = {'X': model_data[['c__colorado','c__michigan','d__michigan','d__nan']], 'y': model_data['a'], 'w':model_data['b']}

        model_runner = v_train.ModelRunner('tweedie', run_list, 
                                           train_data_dict,
                                           train_data_dict,
                                           pipeline)

        sm_eval = model_runner.evaluate_models()

        assert round(sm_eval['rmse'].iloc[0], 4) == 0.4903
        assert round(sm_eval['mae'].iloc[1], 2) == 0.25
        assert round(sm_eval['gini'].iloc[0], 4) == 0.3261
        assert round(sm_eval['weighted_rmse'].iloc[1], 4) == 0.1183
        assert round(sm_eval['tweedie_deviance'].iloc[0], 4) == 1.3337
        assert round(sm_eval['hl_tweedie'].iloc[1], 30) == 0.0
        
class TestXGB(object):
    def test_xgb_simple_reg(self):
        run_list = [{
            'model_type': v_train.V_xgb,
            'model_name': 'xgb',
            'model_params': {
            },
            'fit_params': {
            }
        }]

        train_data_dict = {'X': model_data[['c__colorado','c__michigan','d__michigan','d__nan']], 'y': model_data['a']}

        model_runner = v_train.ModelRunner('reg', run_list, 
                                           train_data_dict,
                                           None,
                                           None)

        xgb_eval = model_runner.evaluate_models()

        assert round(xgb_eval['rmse'].iloc[0], 4) == 0.3538
        assert round(xgb_eval['mae'].iloc[0], 2) == 0.26
        assert round(xgb_eval['gini'].iloc[0], 2) == .25
#         assert round(xgb_eval['weighted_rmse'].iloc[0], 4) == 0.3538
        
        # Plot test have no asserts, checking only for crashing
        
        model_runner.plot_models(False)
        
    def test_xgb_adv_tweedie(self):
        run_list = [{
            'model_type': v_train.V_xgb,
            'model_name': 'xgb',
            'model_params': {
                'objective' : 'reg:tweedie',
                'n_estimators' : 80,
                'learning_rate' : .1,
                'n_jobs':-1
            },
            'fit_params': {
                'sample_weight' : model_data['b'],
                'eval_metric' : 'rmse',
                'eval_set' : [(model_data[['c__colorado','c__michigan','d__michigan','d__nan']], model_data['a'])],
                'verbose' : False
            }
        }]

        train_data_dict = {'X': model_data[['c__colorado','c__michigan','d__michigan','d__nan']], 'y': model_data['a'], 'w':model_data['b']}

        model_runner = v_train.ModelRunner('tweedie', run_list, 
                                           train_data_dict,
                                           train_data_dict,
                                           pipeline)

        xgb_eval = model_runner.evaluate_models()

        assert round(xgb_eval['rmse'].iloc[0], 4) == 0.4902
        assert round(xgb_eval['mae'].iloc[1], 2) == 0.25
        assert round(xgb_eval['gini'].iloc[0], 2) == .33
        assert round(xgb_eval['weighted_rmse'].iloc[1], 2) == 0.12
        assert round(xgb_eval['hl_tweedie'].iloc[1], 2) == 0
        assert round(xgb_eval['tweedie_deviance'].iloc[1], 2) == 1.33
        
        # Plot test have no asserts, checking only for crashing
        
        model_runner.plot_models()
    
    def test_xgb_gridsearch(self):
        run_list = [{
            'model_type': v_train.V_xgb,
            'model_name': 'xgb',
            'model_params': {
                'objective' : ['reg:linear', 'reg:tweedie'],
                'n_jobs': [-1]
            },
            'fit_params': {
                'sample_weight' : model_data['b']
            },
            'search_params' : {
                'type' : 'GridSearch'
                , 'params' : {
                    'cv' : 3
                }
            }
        }]

        train_data_dict = {'X': model_data[['c__colorado','c__michigan','d__michigan','d__nan']], 'y': model_data['a'], 'w':model_data['b']}

        model_runner = v_train.ModelRunner('tweedie', run_list, 
                                           train_data_dict,
                                           train_data_dict,
                                           pipeline)

        xgb_eval = model_runner.evaluate_models()

        assert round(xgb_eval['rmse'].iloc[0], 2) == 0.49
        assert round(xgb_eval['mae'].iloc[1], 2) == 0.25
        assert round(xgb_eval['gini'].iloc[0], 2) == .33
        assert round(xgb_eval['weighted_rmse'].iloc[1], 2) == 0.12
        assert round(xgb_eval['hl_tweedie'].iloc[1], 2) == 0
        assert round(xgb_eval['tweedie_deviance'].iloc[1], 2) == 1.33

class TestMultiModel(object):
    def test_multi_model(self):
        run_list = [{
            'model_type': v_train.V_xgb,
            'model_name': 'xgb',
            'model_params': {},
            'fit_params': {}
        }
            , {
            'model_type': v_train.V_SM_GLM,
            'model_name': 'SM',
            'model_params': {},
            'fit_params': {}
        }
        ]

        train_data_dict = {
            'X': model_data[['c__colorado', 'c__michigan', 'd__michigan', 'd__nan']],
            'y': model_data['a']
        }

        model_runner = v_train.ModelRunner('tweedie', run_list, train_data_dict, None,
                                           None)

        evals = model_runner.evaluate_models()

        # Plot test have no asserts, checking only for crashing
        model_runner.plot_models(False)
        v_stats.plot_compare_stats(evals, valid_only=False)