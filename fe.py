#!/usr/bin/env python3.14

from fe_data import sub_sub_database_features
from machine_learning.machine_learning import model_regression
from matplotlib import cm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from machine_learning.utilities import makedir

# Preparing database for machine learning
sub_sub_database_features(sub_sub_database='results/sub-sub/data/sub_sub_database_v5.csv',
                          folder_output='results/sub-sub/data/',
                          name_output='sub_sub_database_features_v5.csv')

# Machine learning
# Features
features_select = ('include', ['Density_sum', 'Density_diff', 'Eneg_sum', 'Eneg_diff', 'Magnetic_sum', 'Magnetic_diff', 
                                'Melting_sum', 'Melting_diff', 'Number_sum', 'Number_diff', 
                                'Radius_sum', 'Radius_diff', 'Vacancy_1_sum', 'Vacancy_1_diff', 'Vacancy_2_sum',
                                'Vacancy_2_diff', 'Vacancy_3_sum', 'Vacancy_3_diff', 'Valence_sum', 'Valence_diff'])

# List of models
models = {     
                'xgb': {'folder_output': 'results/sub-sub/xgb/',
                        'folder_output_renew': True,
                        'model_type': xgb.XGBRegressor(),
                        'param_grid': {
                                    "n_estimators"    : [300 + 50*i for i in range(11)],
                                    "learning_rate"   : [0.01, 0.1, 0.2],
                                    "max_depth"       : [3 + i for i in range(4)],
                                    "min_child_weight": [1, 3, 5],
                                    "reg_alpha"       : [0, 0.1],                       # L1 Regularization
                                    "reg_lambda"      : [1.0, 2.0],                     # L2 Regularization
                                    "n_jobs"          : [1]
                                    },
                        'model_parameters': {},
                        'model_name': 'XGB'},

                'gbt': {'folder_output': 'results/sub-sub/gbt/',
                        'folder_output_renew': True,
                        'model_type': GradientBoostingRegressor(),
                        'param_grid': {
                                        'max_depth'        : [2, 3, 4],                       # Keep trees shallow
                                        'learning_rate'    : [0.01, 0.05, 0.1],               # Conservative step shrinkage
                                        'n_estimators'     : [300 + 50*i for i in range(11)], # Boosting rounds
                                        'min_samples_leaf' : [5, 10],                         # Prevent leaves with 1-2 samples
                                        'min_samples_split': [10, 20],                        # Minimum samples to consider a split
                                        'subsample'        : [0.8, 1.0],                      # Stochastic instance sampling
                                        'max_features'     : ["sqrt", 1.0]                    # Feature sampling
                                    },
                        'model_parameters': {},
                        'model_name': 'GBT'},

                'rf':  {'folder_output': 'results/sub-sub/rf/',
                        'folder_output_renew': True,
                        'model_type': RandomForestRegressor(),
                        'param_grid': {
                            'n_estimators': [550 + 50*i for i in range(3)],
                            'max_depth': [8 + i for i in range(3)],
                            'min_samples_split': [5, 10, 15],
                            'min_samples_leaf': [2, 4, 6],
                            'max_features': ['sqrt', 0.5, 0.8],
                            'bootstrap': [True]
                        },
                        'model_parameters': {},
                        'model_name': 'RF'}
            }

# Model
for model in models:
# Model optimize parameters, train, test
    model_regression(database_file       = 'results/sub-sub/data/sub_sub_database_features_v5.csv',
                        database_file_type  = 'csv',
                        folder_output       = models[model]['folder_output'],
                        folder_output_renew = models[model]['folder_output_renew'],
                        name_output         = model,

                        filters=[],
                        target='Energy_binding',
                        target_range=(-2.0, 2.0),
                        select_target_range_type='include',
                        features_select=features_select,
                        plot_correlation_parameters={'colorbar_position': [0.90, 0.20, 0.03, 0.75],
                                                    'colormap': cm.Spectral,
                                                    'dpi': 600,
                                                    'figure_size': (25, 25),
                                                    'margins': {'left': 0.15, 'right': 0.85, 'top': 0.95, 'bottom':0.20}},
                        plot_histogram_parameters={'color_histogram': '#00798C',
                                                'color_fill': '#EDAE49',
                                                'alpha': 0.50,
                                                'dpi': 600,
                                                'figure_size': (10, 10),
                                                'bins': 20,
                                                'margins': {'left': 0.15, 'right': 0.90, 'top': 0.95, 'bottom':0.10},
                                                'label_target': 'Energy binding, eV',
                                                'fontsize': 20,
                                                'tick_size': 20},

                        test_size = 0.1,
                        random_state = 0,
                        cross_validation_folds = 10,
                        unit_of_measure = 'eV',

                        model_optimize_parameters = False, # False for symbolic regression
                        model_type = models[model]['model_type'],
                        param_grid = models[model]['param_grid'],
                        n_jobs=4,
                        model_fit = True,
                        model_parameters = models[model]['model_parameters'],
                        model_name = models[model]['model_name'],

                        model_plot_test = True,
                        plot_test_parameters = {'feature_markersize': 'Number_sum',
                                                'markersize_multiply': 250,                                  
                                                'feature_color': 'Radius_sum',
                                                'colorbar_label': '$\\delta R_{sum}$, $R_{Fe}$',
                                                'colorbar_labelpad': 10,
                                                'colorbar_position': [0.90, 0.10, 0.03, 0.80],
                                                'colormap': cm.Spectral,
                                                'figure_size': (7, 20),
                                                'fontsize': 25,
                                                'tick_size': 20,
                                                'margins': {'left': 0.10, 'right': 0.89, 'top': 0.90, 'bottom':0.15, 'hspace':0.25, 'wspace':0.20},
                                                'text_position_mae': (0.10, 0.85),
                                                'text_position_r2': (0.10, 0.90),
                                                'label_horizontal_axis': 'DFT',
                                                'xlim': (-1.05, 0.55),
                                                'ylim': (-1.05, 0.55),
                                                'dpi': 600})