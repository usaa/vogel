from distutils.core import setup

setup(name='vogel',
      version='0.2.6',
      author=['Robert Dohner', 'Michael Waldmeier', 'Chris Manhave'],
      url='https://github.com/usaa/',
      packages=['vogel', 'vogel.preprocessing', 'vogel.train', 'vogel.utils', 'vogel.utils.stats'],
      description='Vogel is a ML project flow tool, with the primary objective of simplifying actuarial ML processes. It tracks and manages model development from data preparation to results analysis and visualization.',
      install_requires =[
          'scikit-learn>=0.20',
          'statsmodels>=0.9.0',
          'matplotlib>=2.0.2',
          'pandas>=0.20.3',
          'xgboost>=0.80',
          'catboost>=0.9.1' 
      ]
)
