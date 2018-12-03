<p align="center">
  <img src="/docs/artwork/vogel_flow.png" width="400" />
</p>

---
**Vogel** is a ML project flow tool, with the primary objective of simplifying actuarial ML processes. It tracks and manages model development from data preparation to results analysis and visualization.

## Install

* Clone the Vogle repo
* In the Vogel repo, pip install
    * `pip install -e .`

## Features
    
* Visualization
    * One-way plots (observed vs actual values)
    * Multi-variate plots (individual feature analysis)
    * Pareto charts
    * Model stats comparison chart
* Custom Variable Transformations
    * Maintains metadata
    * Multiple binning mechanisms
    * Model Comparison Statistics
    * Available statistics vary by model type
* Interfaces with multiple modeling platforms
    * [scikit-learn](https://scikit-learn.org/)
    * [StatsModels](https://www.statsmodels.org/)
    * [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    * [CatBoost](https://github.com/catboost/catboost)
    
## Example

Pandas in Pandas out pipelines. All metadata is carried to the transformed data.

<pre>
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display, HTML

import vogel.preprocessing as v_prep
import vogel.utils as v_utils
import vogel.utils.stats as v_stats
import vogel.train as v_train

# Test Data
df = pd.DataFrame({
      'a': [200., 40., 60., 100., 10., 10., 10.]
    , 'b': [100., 20., 30., np.nan, 5., 5., 5.]
    , 'c': ['texas', 'texas', 'michigan', 'colorado', 'michigan', 'michigan', 'michigan']
    , 'd': ['texas', 'texas', 'michigan', np.nan, 'michigan', 'michigan', 'michigan']
    , 'e': [1., 1., 1., 1., 1., 1., 1.]
    , 'f': [0., 10., 20, 1., 2., 20., 1000.]
})

display(df)

data_dict = {
    'grp_numeric': ['a', 'b']
  , 'grp_cat': ['c', 'd']
  , 'grp_other': ['a', 'c']
}

pipeline = v_utils.make_pipeline(
    v_prep.FeatureUnion([
        ('numeric', v_utils.make_pipeline(
            v_prep.ColumnExtractor(['grp_numeric', 'd'], data_dict, want_numeric = True),
            v_prep.NullEncoder(),
            v_prep.Imputer(),
            v_prep.Binning(bin_type='qcut', bins=3, bin_id='mean', drop='replace', feature_filter=['a'])
        )),
        ('cats', v_utils.make_pipeline(
            v_prep.ColumnExtractor(['grp_numeric', 'd'], data_dict, want_numeric = False),
            v_prep.LabelEncoder()
        ))
    ])
)

train_X = pipeline.fit_transform(df)

display(train_X)
</pre>

<p align="center">
  <img src="/docs/artwork/bf_aftr_data.png" width="400" />
</p>

We can now run a few models on this transformed data. We will ignore the validation and hyperparamter tunning options for now.
<pre>
train_y = df['f'] 

run_list = [
    {
        'model_type': v_train.V_SM_GLM,
        'model_name': 'simple' + '_SM_glm_tweedie',
        'model_params': {
            'family': sm.families.Gaussian()
        },
        'fit_params': {
        }
    }, 
    {
        'model_type': v_train.V_xgb,
        'model_name': 'simple_1' + '_xgb',
        'model_params': {
            'objective': 'reg:linear',
            'n_estimators': 1,
            'n_jobs': -1
        },
        'fit_params': {
            'eval_set': [(train_X, train_y)],
            'verbose': False
        }
    }
    ,
    {
        'model_type': v_train.V_xgb,
        'model_name': 'simple_80' + '_xgb',
        'model_params': {
            'objective': 'reg:linear',
            'n_estimators': 80,
            'n_jobs': -1
        },
        'fit_params': {
            'eval_set': [(train_X, train_y)],
            'verbose': False
        }
    }
]

train_data_dict = {
    'X': train_X, 
    'y': train_y
}

model_runner = v_train.ModelRunner('reg', run_list, train_data_dict,
                                   None, pipeline)

eval_set = model_runner.evaluate_models()
display(eval_set)
</pre>

<p align="center">
  <img src="/docs/artwork/model_stats.png" width="800" />
</p>

With the stats package we can vizualize how our modes fit. We will chose the GLM, as it is the simpelest best fitting model.
<pre>
v_stats.plot_compare_stats(eval_set, valid_only=False)
</pre>

<p align="center">
  <img src="/docs/artwork/plot_model_stats.png" width="800" />
</p>

We can see how indivitual features fit in out model.
<pre>
mdl_glm = model_runner.models[0]
print('b')
v_stats.plot_one_way_fit(train_X['b'], mdl_glm.predict(train_X), target=train_y, target_error=True, pad_bar_chart=True)
mdl_glm.plot_glm_one_way_fit(plot_error=False)
</pre>

<p align="center">
  <img src="/docs/artwork/plot_fit.png" width="800" />
</p>

[More examples](/examples)
