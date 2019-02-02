import pandas as pd
import numpy as np

from sklearn import linear_model

import statsmodels.api as sm
## This is a temp fix for statsmodes setting breaking pickling ##
## https://github.com/statsmodels/statsmodels/issues/4772 ##
import types, pickle
if types.MethodType in pickle.dispatch_table.keys():
    del pickle.dispatch_table[types.MethodType]

import xgboost as xgb
import catboost as catb

import vogel.utils.stats as v_stats
import vogel.utils as v_utils

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ModelRunner():
    """
    Class to handle model fit and analysis
    Weight will not be used in a model unless specified in model_dict
    
    Args:
        pipeline: (None)
            pipeline: pre model ETL pipeline. Will be non-optional in future versions
        target_type:
            string: The target characteristics for this model. (bool, reg, tweedie)
        run_list: 
            dict: models and paramters to be run.
            Example:
             [{
                'model_type': d_train.D_SM_GLM,
                'model_name': 'cl' + '_SM_LinReg_tweedie',
                'model_params': {
                    'custom_weight': train_w,
                    'family': sm.families.Tweedie(var_power=1.5)
                },
                'fit_params': {
                    'maxiter': 10
                }
            }]
        train_set:
            dict: {'X': train_X, 'y': train_y, 'w': train_w}
        valid_set: (None)
            dict: 'X': valid_X, 'y': valid_y, 'w': valid_w}
        pipeline:
            Pipeline: can be used to maintain meta-data throught modeling & analysis process
    """

    def __init__(self,
                 target_type,
                 run_list,
                 train_set,
                 valid_set=None,
                 pipeline=None):

        self.target_type = target_type
        self.run_list = run_list

        self.train_set = train_set
        self.valid_set = valid_set

        self.pipeline = pipeline

        self.models = self.train_models()

    def train_models(self):
        temp_models = []
        for model_dict in self.run_list:
            try:
                temp_models.append(model_dict['model_type'](
                    model_dict, self.target_type,
                    self.train_set, self.valid_set, self.pipeline).train())
            except Exception as e:
                print('{} Failed'.format(model_dict['model_name']))
                print(e)

        return temp_models

    def evaluate_models(self):
        out_df = pd.DataFrame()
        for model in self.models:
            temp_df = self.create_model_record(model)
            out_df = pd.concat([out_df, temp_df])
        return out_df.reset_index(drop=True)

    def create_model_record(self, model):
        temp_df = pd.DataFrame()
        for key, val in model.evaluate().items():
            temp = pd.DataFrame.from_dict(val)
            try:
                temp['model_name'] = model.model_dict['model_name']
            except:
                temp['model_name'] = ''
            temp['model_dict'] = str(model.model_dict)
            temp['set'] = key
            temp_df = pd.concat([temp_df, temp], axis=0, sort=False)
        return temp_df

    def plot_models(self, valid_only=True):
        for model in self.models:
            model.plot_fit(valid_only)
        pass


class BaseModel():
    def __init__(self, model_dict, target_type, train_set,
                 valid_set, pipeline):

        self.model_dict = model_dict
        self.model_type = model_dict['model_type']
        self.model_params = model_dict['model_params']
        self.fit_params = model_dict['fit_params']
        self.target_type = target_type
        if 'search_params' in model_dict:
            self.search_params = model_dict['search_params']
        else:
            self.search_params = None

        self.pipeline = pipeline

        self.train_set = train_set
        self.valid_set = valid_set

        self.model = None
        self.search = None
    
    def fit(self):
        if self.search_params is None:
            self.model.set_params(**self.model_dict['model_params'])
            self.model = self.model.fit(**self.fit_params)
        elif self.search_params['type'] == 'GridSearch':
            if 'params' not in self.search_params:
                self.search_params['params'] = {}
            self.search = GridSearchCV(self.model, param_grid=self.model_params, **self.search_params['params'])
            self.search.fit(**self.fit_params)
            self.model = self.search.best_estimator_
        elif self.search_params['type'] == 'RandomSearch':
            if 'params' not in self.search_params:
                self.search_params['params'] = {}
            self.search = RandomizedSearchCV(self.model, param_distributions=self.model_params, **self.search_params['params'])
            self.search.fit(**self.fit_params)
            self.model = self.search.best_estimator_
        else:
            print(self.search_params['type'], ' is not supported yet.')

    def plot_fit(self, valid_only=True):
        print(self.model_dict['model_name'])
        train_pred = self.predict(self.train_set['X'])
        if self.valid_set is not None:
            valid_pred = self.predict(self.valid_set['X'])

        if self.target_type == 'bool':
            if not valid_only:
                print('Train Recall')
                v_stats.plot_recall(self.train_set['y'], train_pred)
            if self.valid_set is not None:
                print('Valid Recall')
                v_stats.plot_recall(self.valid_set['y'], valid_pred)

        if self.target_type == 'tweedie':
            if not valid_only:
                print('Train Hosmer–Lemeshow')
                v_stats.plot_hl(self.train_set['w'], self.train_set['y'],
                                train_pred)
            if self.valid_set is not None:
                print('Valid Hosmer–Lemeshow')
                v_stats.plot_hl(self.valid_set['w'], self.valid_set['y'],
                                valid_pred)

        if self.target_type == 'reg':
            if not valid_only:
                print('Train Actual Vs Predicted')
                pd.DataFrame({
                    'y': np.array(self.train_set['y']),
                    'y_hat': np.array(train_pred)
                }).plot.scatter(x='y', y='y_hat')
            if self.valid_set is not None:
                print('Valid Actual Vs Predicted')
                pd.DataFrame({
                    'y': np.array(self.valid_set['y']),
                    'y_hat': np.array(valid_pred)
                }).plot.scatter(x='y', y='y_hat')

    def predict(self, data):
        out_data = None
        if str(self.model)[1:12] == 'statsmodels':
            out_data = self.model.predict(sm.add_constant(data))
        else:
            out_data = self.model.predict(data)

        return out_data

    def evaluate(self, p=1.5):
        model_stats = None

        # predict train and validation
        train_pred = self.predict(self.train_set['X'])
        if self.valid_set is not None:
            valid_pred = self.predict(self.valid_set['X'])
        else:
            valid_pred = None
            
        if 'w' not in self.train_set:
            self.train_set['w'] = np.ones_like(self.train_set['y'])
        if self.valid_set is not None:
            if 'w' not in self.valid_set:
                self.valid_set['w'] = np.ones_like(self.valid_set['y'])

        if self.target_type == 'reg':
            if self.valid_set is not None:
                model_stats = v_stats.model_stats_reg(
                    {'y' : self.train_set['y'], 'pred' : train_pred, 'w' : self.train_set['w']},
                    {'y' : self.valid_set['y'], 'pred' : valid_pred, 'w' : self.valid_set['w']}
                )
            else:
                model_stats = v_stats.model_stats_reg(
                    {'y' : self.train_set['y'], 'pred' : train_pred, 'w' : self.train_set['w']},
                    None
                )
        elif self.target_type == 'tweedie':
            if self.valid_set is not None:
                model_stats = v_stats.model_stats_tweedie(
                    {'y' : self.train_set['y'], 'pred' : train_pred, 'w' : self.train_set['w']},
                    {'y' : self.valid_set['y'], 'pred' : valid_pred, 'w' : self.valid_set['w']},
                    p)
            else:
                model_stats = v_stats.model_stats_tweedie(
                {'y' : self.train_set['y'], 'pred' : train_pred, 'w' : self.train_set['w']},
                None,
                p)
        elif self.target_type == 'bool':
            if self.valid_set is not None:
                model_stats = v_stats.model_stats_bool(
                    {'y' : self.train_set['y'], 'pred' : train_pred, 'w' : self.train_set['w']},
                    {'y' : self.valid_set['y'], 'pred' : valid_pred, 'w' : self.valid_set['w']}
                )
            else:
                model_stats = v_stats.model_stats_bool(
                    {'y' : self.train_set['y'], 'pred' : train_pred, 'w' : self.train_set['w']},
                    None
                )
        else:
            raise ValueError('Target type {} not supported'.format(
                self.target_type))

        return model_stats


class V_xgb(BaseModel):
    def __init__(self, model_dict, target_type, train_set, valid_set, pipeline):
        super().__init__(model_dict, target_type, train_set, valid_set, pipeline)

    def train(self):
        self.fit_params['X'] = self.train_set['X']
        self.fit_params['y'] = self.train_set['y']

        model = None

        if self.target_type == 'bool':
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
        
        self.fit()
            
        return self

    def plot_fit(self, valid_only=True, error='error'):
        super().plot_fit(valid_only)
        
        eval_metric = error
        if 'eval_set' in self.model_dict['fit_params']:
            if 'eval_metric' in self.model_dict['fit_params']:
                eval_metric = self.model_dict['fit_params']['eval_metric']
            else:
                eval_metric = error
            v_stats.plot_xgb_fit(self.model.evals_result(), eval_metric)
            
class V_catb(BaseModel):
    def __init__(self, model_dict, target_type, train_set, valid_set, pipeline):
        super().__init__(model_dict, target_type, train_set, valid_set, pipeline)

    def train(self):
        self.fit_params['X'] = self.train_set['X']
        self.fit_params['y'] = self.train_set['y']

        model = None

        if self.target_type == 'bool':
            self.model = catb.CatBoostClassifier()
        else:
            self.model = catb.CatBoostRegressor()
        
        self.fit()
            
        return self


class V_LinReg(BaseModel):
    def __init__(self, model_dict, pipeline, target_type, train_set, valid_set):
        super().__init__(model_dict, target_type, train_set, valid_set)

    def train(self):
        self.fit_params['X'] = self.train_set['X']
        self.fit_params['y'] = self.train_set['y']

        model = linear_model.LinearRegression(**self.model_params)
        
        self.fit()
        
        return self


class V_SM_GLM(BaseModel):
    def __init__(self, model_dict, pipeline, target_type, train_set,
                 valid_set):
        super().__init__(model_dict, pipeline, target_type, train_set,
                         valid_set)

        self.feature_stats = None
        self.label_dicts = None
        
    def fit(self):
        self.model = self.model.fit(**self.fit_params)
        
    def train(self):
        self.model_params['exog'] = sm.add_constant(self.train_set['X'])
        self.model_params['endog'] = self.train_set['y'].tolist()
        
        # TODO// Manhave will add comments 
        def set_custom_weights():
            if 'custom_weight' in self.model_params:
                weight = self.model_params['custom_weight']
                self.model_params['freq_weights'] = weight * len(weight) / np.sum(weight)
                self.model_params['var_weights'] = np.ones_like(weight) * np.sum(weight) / len(weight)
                del self.model_params['custom_weight']
            
        set_custom_weights()
        
        self.model = sm.GLM(**self.model_params)
        
        self.fit()
        
        return self

    def get_label_dicts(self):
        self.label_dicts = v_utils.find_label_dicts(pipeline=self.pipeline, model_inputs=self.train_set['X'].columns)

    def create_feature_stats_table(self):
        #get glm stats
        mdl_fits = self.model.summary2().tables[1]
        mdl_cov = self.model.cov_params()[['const']]
        mdl_cov.columns = ['cov']

        mdl_stats = pd.concat([mdl_fits, mdl_cov], axis=1)

        if self.label_dicts is None:
            self.get_label_dicts()

        #get bin values from pipeline & counts
        labels_df = v_stats.get_glm_label_and_count_table(
            self.label_dicts, self.train_set['X'])

        mdl_stats = pd.concat([mdl_stats, labels_df], axis=1, sort=True)

        mdl_stats = v_stats.get_glm_error(mdl_stats, self.label_dicts)

        self.feature_stats = mdl_stats

    def plot_glm_one_way_fit(self,
                             features=None,
                             plot_error=True,
                             round_at=2,
                             rotate_x=True,
                             predicted_value=True,
                             scale_to_zero=False,
                             base_line=True,
                             pad_bar_chart=True,
                             plt_size=(10, 5),
                             return_fig_dict=False
                            ):
        
        """
        one way feature fits for a GLM

        Args:
        round_at: (2)
            int: rounding point
        predicted_value: (True)
            bool: plot predicted value or liniear prodictor
        rotate_x: (True)
            bool: rotate x axis lables
        scale_to_zero: (False)
            bool: scale all valuse where the base is 0
        base_line: (True)
            bool: draw line at base level on y axis
        pad_bar_chart: (False)
            bool: add 100% padding to the top of x axis for the bar chart
        plt_size: (10, 5)
            tuple: (width, height)
        return_fig_dict: (False)
            bool: return a dict of figs instead of showing plt
        """
        
        fig_dict = {}

        if self.feature_stats is None:
            self.create_feature_stats_table()

        for feature_set in self.label_dicts.items():
            feature_name = feature_set[0]
            feature_dict = feature_set[1]
            items = feature_dict['items']

            def plot_it():
                #filter data set to one feature group
                mod_features = [v_utils.label_dict_formater(feature_dict['sep'], feature_name, x)
                                    for x in items] + [feature_name + '___const']
                    
                mdl_stats_fltrd = self.feature_stats[
                    self.feature_stats.index.isin(mod_features)]

                data = mdl_stats_fltrd[[
                    'bin', 'count', 'coef_exp', 'error+', 'error-', 'hold'
                ]]
                if return_fig_dict:
                    temp_fig = v_stats.plot_glm_one_way_fit(
                        data,
                        plot_error=plot_error,
                        round_at=round_at,
                        rotate_x=rotate_x,
                        predicted_value=predicted_value,
                        scale_to_zero=scale_to_zero,
                        plt_size=plt_size,
                        return_fig=True
                    )
                
                    fig_dict[feature_name] = temp_fig
                else:
                    print(feature_name)
                    v_stats.plot_glm_one_way_fit(
                        data,
                        plot_error=plot_error,
                        round_at=round_at,
                        rotate_x=rotate_x,
                        predicted_value=predicted_value,
                        scale_to_zero=scale_to_zero,
                        plt_size=plt_size
                )

            plot_it()
    
        if return_fig_dict:
            return fig_dict
