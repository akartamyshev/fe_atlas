#!/usr/bin/python3
# -*- coding: utf-8 -*- 
"""
siman is needed; Author: Aksyonov D.A.
"""

def apply_model(data=[], preprocessing_model_file='', model_file=''):
    import numpy as np
    import pickle

    # Change type of data to numpy array
    data = np.array([data])

    # Applying preprocessing model
    if preprocessing_model_file:
        with open(preprocessing_model_file, 'rb') as f:
            preprocessing_model = pickle.load(f)
        data = preprocessing_model.transform(data)

    # Applying model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    result = model.predict(data)    

    return result





def select_data(database=(), filters=[], target='', target_range=(-1.0, 1.0), select_target_range_type='include', features_select=[], 
                plot_correlation_parameters={}, plot_histogram_parameters={}, folder_output='', folder_output_renew=False, name_output=''):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from machine_learning.utilities import makedir

    # The output folder
    makedir(folder_output+'/', renew_folder = folder_output_renew)

    # Read database
    if database[1] == 'csv':
        data = pd.read_csv(database[0])
    elif database[1] == 'txt':
        f = open(database[0])
        l = f.readlines()
        f.close()
        data_features = l[0].split()
        data_dict = {}
        for i in data_features:
            data_dict[i] = []
        for i in l[1:]:
            i1 = [float(i2) for i2 in i.split()]
            for j in range(len(data_features)):
                data_dict[data_features[j]].append(i1[j])
        data = pd.DataFrame.from_dict(data_dict)

    # Remove data with missing values of features
    print(f'len(data) before dropna {len(data)}')
    data = data.dropna()
    print(f'len(data) after dropna {len(data)}')

    # Apply filters
    if filters:
        for i in filters:
            if i[2] == 'include':
                data = data[(data[i[0]] > i[1][0]) & (data[i[0]] < i[1][1])]
            elif i[2] == 'exclude':
                data = data[(data[i[0]] < i[1][0]) & (data[i[0]] > i[1][1])]

    # Select examples
    if select_target_range_type == 'include':
        data_selected = data[(data[target] > target_range[0]) & (data[target] < target_range[1])]
    elif select_target_range_type == 'exclude':
        data_selected = data[(data[target] < target_range[0]) & (data[target] > target_range[1])]
    data_selected.to_csv(folder_output+'/'+name_output+'_database.csv')

    # Select features
    features = list(data_selected.columns)
    features.remove(target)
    if features_select[0] == 'exclude':
        for i in features_select[1]:
            features.remove(i)
    elif features_select[0] == 'include':
        features = features_select[1]
    data_X = data_selected[features]
    data_y = data_selected[target]

    # Correlation between all features
    correlation = pd.concat([data_X, data_y], axis=1).corr()
    fig, ax = plt.subplots(figsize=plot_correlation_parameters['figure_size'])
    ax_cbar = fig.add_axes(plot_correlation_parameters['colorbar_position'])
    correlation_plot = sns.heatmap(correlation, annot=True, fmt=".2f", cmap=plot_correlation_parameters['colormap'], ax=ax, cbar_ax=ax_cbar)
    plt.subplots_adjust(top=plot_correlation_parameters['margins']['top'],
                        left=plot_correlation_parameters['margins']['left'], 
                        bottom=plot_correlation_parameters['margins']['bottom'], 
                        right=plot_correlation_parameters['margins']['right'])
    fig.savefig(folder_output+'/'+name_output+'_correlation.pdf', format='pdf', dpi=plot_correlation_parameters['dpi'])
    plt.clf()

    # Histogram of target value
    fig, ax = plt.subplots(figsize=plot_histogram_parameters['figure_size'])
    histogram_plot = sns.histplot(data[target], bins=plot_histogram_parameters['bins'], ax=ax, kde=True, color=plot_histogram_parameters['color_histogram'])
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    ax.fill_between(np.linspace(target_range[0], target_range[1], 100), ax.get_ylim()[0]*2, ax.get_ylim()[1]*2, color=plot_histogram_parameters['color_fill'], alpha=plot_histogram_parameters['alpha'])
    ax.tick_params(axis='both', which='major', labelsize=plot_histogram_parameters['tick_size'])
    ax.tick_params(axis='both', which='minor', labelsize=plot_histogram_parameters['tick_size'])
    ax.set_xlabel(plot_histogram_parameters['label_target'], fontsize=plot_histogram_parameters['fontsize'])
    ax.set_ylabel('Count', fontsize=plot_histogram_parameters['fontsize'])
    plt.subplots_adjust(top=plot_histogram_parameters['margins']['top'],
                        left=plot_histogram_parameters['margins']['left'], 
                        bottom=plot_histogram_parameters['margins']['bottom'], 
                        right=plot_histogram_parameters['margins']['right'])
    fig.savefig(folder_output+'/'+name_output+'_'+target+'.pdf', format='pdf', dpi=plot_histogram_parameters['dpi'])
    plt.clf()

    return data_selected, data_X, data_y





def train_test_split_data(data_X, data_y, test_size=0.2, random_state=0):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test





def model_regression_grid_search(data={}, test_size=0.2, random_state=0, model_type=[], model_name='', param_grid={}, n_jobs=1, **parameters):
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import mean_absolute_error
    from machine_learning.machine_learning import train_test_split_data
    from machine_learning.utilities import makedir

    # The output folder
    makedir(parameters['folder_output']+'/', renew_folder = parameters['folder_output_renew'])

    # Correlation between all features
    correlation = pd.concat([data['data_X'], data['data_y']], axis=1).corr()
    correlation_target = correlation[data['data_y'].name][:-1]

    # Splitting data into training and test datasets
    features = list(data['data_X'].columns)
    X_trainval, X_test, y_trainval, y_test = train_test_split_data(data['data_X'], data['data_y'], test_size=test_size, random_state=random_state)

    # Initializing model object and fitting
    cv_strategy = KFold(n_splits=parameters['cross_validation_folds'], shuffle=True, random_state=random_state)
    model_grid_search = GridSearchCV(model_type, param_grid, n_jobs=n_jobs, cv=cv_strategy)
    model_grid_search.fit(X_trainval, y_trainval)
    y_pred = model_grid_search.predict(X_test)

    # Write model
    with open(parameters['folder_output']+'/'+parameters['name_output']+'_grid_search_model_'+model_name+'.pkl', 'wb') as f:
        pickle.dump(model_grid_search, f)

    # Test results
    with open(parameters['folder_output']+'/'+parameters['name_output']+'_grid_search_'+model_name+'.out', 'w') as f:
        f.write("Data: {0:10d} examples and {1:10d} features\n".format(data['data'].shape[0], data['data'].shape[1]))
        f.write("Model: {}\n".format(model_type))
        f.write("Parameters grid "+str(param_grid)+"\n")
        f.write("Best cross-validation score: {}\n".format(model_grid_search.best_score_))
        f.write("Best parameters: {}\n".format(model_grid_search.best_params_))
        f.write("Accuracy on test set: {:.3f}\n".format(model_grid_search.score(X_test, y_test)))
        f.write("Mean absolute error on test set: {0:.3f} {1:s}\n".format(mean_absolute_error(y_test, y_pred), parameters['unit_of_measure']))
    for line in open(parameters['folder_output']+'/'+parameters['name_output']+'_grid_search_'+model_name+'.out'):
        print(line)

    return model_grid_search





def model_regression_fit(data={}, test_size=0.2, random_state=0, model_type=[], model_name='', **parameters):
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error
    from machine_learning.machine_learning import train_test_split_data
    from machine_learning.utilities import makedir

    # The output folder
    makedir(parameters['folder_output']+'/', renew_folder = parameters['folder_output_renew'])

    # Correlation between all features
    correlation = pd.concat([data['data_X'], data['data_y']], axis=1).corr()
    correlation_target = correlation[data['data_y'].name][:-1]

    # Splitting data into training and test datasets
    features = list(data['data_X'].columns)
    X_trainval, X_test, y_trainval, y_test = train_test_split_data(data['data_X'], data['data_y'], test_size=test_size, random_state=random_state)

    # Fit model
    model = model_type
    model.set_params(**parameters['model_parameters'])
    if parameters['cross_validation_folds'] >= 5:
        scores = cross_val_score(model, X_trainval, y_trainval, cv=parameters['cross_validation_folds'])
    model.fit(X_trainval, y_trainval)

    # Write model
    with open(parameters['folder_output']+'/'+parameters['name_output']+'_model_'+model_name+'.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Calculate predictions and metrics
    y_pred_total = model.predict(data['data_X'])
    y_pred_trainval = model.predict(X_trainval)
    y_pred_test = model.predict(X_test)
    total_score = model.score(data['data_X'], data['data_y'])
    train_score = model.score(X_trainval, y_trainval)
    test_score = model.score(X_test, y_test)

    # Write test results
    with open(parameters['folder_output']+'/'+parameters['name_output']+'_test_'+model_name+'.out', 'w') as f:
        f.write("Data: {0:10d} examples and {1:10d} features\n".format(len(data['data_X']), len(data['data_X'].columns)+1))
        f.write("Model: {}\n\n".format(model.get_params()))
        if hasattr(model, 'sympy'):
            f.write("Model sympy: {}\n".format(model.sympy()))
        if parameters['cross_validation_folds'] >= 5:
            f.write("Cross-validation scores: {}\n".format(scores))
            average_train_score = np.mean(scores)
            f.write("Mean cross-validation score: {}\n".format(average_train_score))
        f.write("Accuracy on total set: {}\n".format(total_score))
        f.write("Accuracy on training set: {}\n".format(train_score))
        f.write("Accuracy on test set: {}\n\n".format(test_score))
        f.write("Mean absolute error on total set: {0:.3f} {1:s}\n".format(mean_absolute_error(data['data_y'], y_pred_total), parameters['unit_of_measure']))
        f.write("Mean absolute error on training set: {0:.3f} {1:s}\n".format(mean_absolute_error(y_trainval, y_pred_trainval), parameters['unit_of_measure']))
        f.write("Mean absolute error on test set: {0:.3f} {1:s}\n\n".format(mean_absolute_error(y_test, y_pred_test), parameters['unit_of_measure']))
        if hasattr(model, 'feature_importances_'):
            f.write("Feature importances and correlations with target\n")
            f.write("\\hline\n")
            f.write("{0:25s} & {1:^25s} & {2:^25s} \\\\ \n".format("Feature", "Importance, \\%", "Correlation"))
            f.write("\\hline\n")
            for i in range(len(features)):
                f.write("{0:25s} & {1:^25.3f} & {2:^25.3f} \\\\ \n".format(features[i].replace('_', '\\_'), model.feature_importances_[i]*100.0, correlation_target[i]))
            f.write("\\hline\n")
        else:
            f.write("Correlations with target\n")
            f.write("\\hline\n")
            f.write("{0:25s} & {1:^25s} \\\\ \n".format("Feature", "Correlation"))
            f.write("\\hline\n")
            for i in range(len(features)):
                f.write("{0:25s} & {1:^25.3f} \\\\ \n".format(features[i].replace('_', '\\_'), correlation_target[i]))
            f.write("\\hline\n")
    for line in open(parameters['folder_output']+'/'+parameters['name_output']+'_test_'+model_name+'.out'):
        print(line)

    return model





def model_regression_test_plot(model_file='', data={}, test_size=0.2, random_state=0, **parameters):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.gridspec as gspec
    import pickle
    from sklearn.metrics import mean_absolute_error
    from machine_learning.machine_learning import train_test_split_data
    from machine_learning.utilities import makedir

    # The output folder
    makedir(parameters['folder_output']+'/', renew_folder = parameters['folder_output_renew'])

    # Read model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split_data(data['data_X'], data['data_y'], test_size=test_size, random_state=random_state)

    # Calculate predictions and metrics
    y_pred_total = model.predict(data['data_X'])
    y_pred_trainval = model.predict(X_trainval)
    y_pred_test = model.predict(X_test)
    total_score = model.score(data['data_X'], data['data_y'])
    train_score = model.score(X_trainval, y_trainval)
    test_score = model.score(X_test, y_test)

    # GridSpec parameters
    margins = parameters['margins']
    gs = gspec.GridSpec(1, 3,  height_ratios = [1.0], width_ratios=[1.0, 1.0, 1.0]) 
    gs.update(bottom=margins['bottom'], top=margins['top'], right=margins['right'], left=margins['left'], hspace=margins['hspace'], wspace=margins['wspace'])

    # Text position
    text_position_r2 = parameters['text_position_r2']
    text_position_mae = parameters['text_position_mae']

    # Markersize
    try:
        markersize_total = data['data_X'][parameters['feature_markersize']]*parameters['markersize_multiply']
        markersize_train = X_trainval[parameters['feature_markersize']]*parameters['markersize_multiply']
        markersize_test  = X_test[parameters['feature_markersize']]*parameters['markersize_multiply']
    except KeyError:
        markersize_total = parameters['markersize']
        markersize_train = parameters['markersize']
        markersize_test = parameters['markersize']

    # Total set
    fig = plt.figure(1)
    ax_total = plt.subplot(gs[0,0])
    ax_total.scatter(data['data_y'], y_pred_total, s=markersize_total, c=data['data'][parameters['feature_color']], 
                     vmin=min(data['data'][parameters['feature_color']]), vmax=max(data['data'][parameters['feature_color']]), 
                     cmap=parameters['colormap'])
    ax_total.plot([min(data['data_y']), max(data['data_y'])], [min(data['data_y']), max(data['data_y'])], linestyle='-', linewidth=1, color='black')
    ax_total.tick_params(axis='both', which='major', labelsize=parameters['tick_size'])
    ax_total.tick_params(axis='both', which='minor', labelsize=parameters['tick_size'])
    if parameters['xlim']:
        ax_total.set_xlim(parameters['xlim'][0], parameters['xlim'][1])
    if parameters['ylim']:
        ax_total.set_ylim(parameters['ylim'][0], parameters['ylim'][1])
    ax_total.set_xlabel(parameters['label_horizontal_axis'], fontsize=parameters['fontsize'])
    ax_total.set_ylabel("Prediction", fontsize=parameters['fontsize'])
    ax_total.set_title("Total set = {:.1f} %".format(100), fontsize=parameters['fontsize'])
    plt.text(text_position_r2[0], text_position_r2[1], "$R^2$ = {:.3f}".format(total_score), fontsize=parameters['fontsize'], transform=ax_total.transAxes)
    plt.text(text_position_mae[0], text_position_mae[1], "MAE = {0:.3f} {1:s}".format(mean_absolute_error(data['data_y'], y_pred_total), parameters['unit_of_measure']), fontsize=parameters['fontsize'], 
             transform=ax_total.transAxes)

    # Train set
    ax_train = plt.subplot(gs[0,1])
    ax_train.scatter(y_trainval, y_pred_trainval, s=markersize_train, c=X_trainval[parameters['feature_color']], 
                     vmin=min(data['data'][parameters['feature_color']]), vmax=max(data['data'][parameters['feature_color']]), 
                     cmap=parameters['colormap'])
    ax_train.plot([min(y_trainval), max(y_trainval)], [min(y_trainval), max(y_trainval)], linestyle='-', linewidth=1, color='black')
    ax_train.tick_params(axis='both', which='major', labelsize=parameters['tick_size'])
    ax_train.tick_params(axis='both', which='minor', labelsize=parameters['tick_size'])
    if parameters['xlim']:
        ax_train.set_xlim(parameters['xlim'][0], parameters['xlim'][1])
    if parameters['ylim']:
        ax_train.set_ylim(parameters['ylim'][0], parameters['ylim'][1])
    ax_train.set_xlabel(parameters['label_horizontal_axis'], fontsize=parameters['fontsize'])
    ax_train.set_title("Train set = {:.1f} %".format((1-test_size)*100), fontsize=parameters['fontsize'])
    plt.text(text_position_r2[0], text_position_r2[1], "$R^2$"+" = {:.3f}".format(train_score), fontsize=parameters['fontsize'], transform=ax_train.transAxes)
    plt.text(text_position_mae[0], text_position_mae[1], "MAE = {0:.3f} {1:s}".format(mean_absolute_error(y_trainval, y_pred_trainval), parameters['unit_of_measure']), fontsize=parameters['fontsize'], 
             transform=ax_train.transAxes)

    # Test set
    ax_test = plt.subplot(gs[0,2])
    ax_test.scatter(y_test, y_pred_test, s=markersize_test, c=X_test[parameters['feature_color']], 
                    vmin=min(data['data'][parameters['feature_color']]), vmax=max(data['data'][parameters['feature_color']]), 
                    cmap=parameters['colormap'])
    ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='-', linewidth=1, color='black')
    ax_test.tick_params(axis='both', which='major', labelsize=parameters['tick_size'])
    ax_test.tick_params(axis='both', which='minor', labelsize=parameters['tick_size'])
    if parameters['xlim']:
        ax_test.set_xlim(parameters['xlim'][0], parameters['xlim'][1])
    if parameters['ylim']:
        ax_test.set_ylim(parameters['ylim'][0], parameters['ylim'][1])
    ax_test.set_xlabel(parameters['label_horizontal_axis'], fontsize=parameters['fontsize'])
    ax_test.set_title("Test set = {:.1f} %".format(test_size*100), fontsize=parameters['fontsize'])
    plt.text(text_position_r2[0], text_position_r2[1], "$R^2$ = {:.3f}".format(test_score), fontsize=parameters['fontsize'], transform=ax_test.transAxes)
    plt.text(text_position_mae[0], text_position_mae[1], "MAE = {0:.3f} {1:s}".format(mean_absolute_error(y_test, y_pred_test), parameters['unit_of_measure']), fontsize=parameters['fontsize'], 
             transform=ax_test.transAxes)

    # Colorbar
    ax_cbar = fig.add_axes(parameters['colorbar_position'])
    norm = matplotlib.colors.Normalize(vmin=min(data['data'][parameters['feature_color']]), vmax=max(data['data'][parameters['feature_color']]))
    cbar = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=parameters['colormap'], norm=norm, orientation='vertical')
    ax_cbar.tick_params(axis='both', which='major', labelsize=parameters['tick_size'])
    ax_cbar.tick_params(axis='both', which='minor', labelsize=parameters['tick_size'])
    cbar.set_label(parameters['colorbar_label'], fontsize=parameters['fontsize'], labelpad=parameters['colorbar_labelpad'])

    # Save figure
    fig.set_figheight(parameters['figure_size'][0])
    fig.set_figwidth(parameters['figure_size'][1])
    fig.savefig(parameters['folder_output']+'/'+parameters['name_output']+'_scatter_plot_predictions_'+parameters['model_name']+'.pdf', format='pdf', dpi=parameters['dpi'])
    plt.clf()




def model_regression(database_file = '',
                     database_file_type = '',
                     folder_output = '',
                     folder_output_renew = False,
                     name_output = '',

                     filters=[],
                     target='',
                     target_range=(),
                     select_target_range_type='',
                     features_select=[],
                     plot_correlation_parameters={},
                     plot_histogram_parameters={},

                     test_size = 0.2,
                     random_state = 0,
                     cross_validation_folds = 10,
                     unit_of_measure = '',

                     model_optimize_parameters = False,
                     model_type = [],
                     param_grid = {},
                     n_jobs = 1,
                     model_fit = True,
                     model_parameters = {},
                     model_name = '',

                     
                     model_plot_test = True,
                     plot_test_parameters = {}):
        
        from machine_learning.machine_learning import select_data, model_regression_grid_search, model_regression_fit, model_regression_test_plot
        from machine_learning.utilities import makedir
        import time

        # The output folder
        makedir(folder_output+'/', renew_folder = folder_output_renew)

        file_time = open(folder_output+'/time.txt', 'w')

        # Select data for machine learning
        start_time = time.perf_counter()
        data_full, data_X, data_y = select_data(database=(database_file, database_file_type),
                                                filters=filters,
                                                target=target, 
                                                target_range=target_range, 
                                                select_target_range_type=select_target_range_type,  
                                                features_select=features_select,
                                                plot_correlation_parameters=plot_correlation_parameters,
                                                plot_histogram_parameters=plot_histogram_parameters, 
                                                folder_output=folder_output,
                                                folder_output_renew=False,
                                                name_output=name_output)
        data = {'data': data_full,'data_X': data_X, 'data_y': data_y}
        end_time = time.perf_counter()
        duration = (end_time - start_time)/60.0
        print(f"'select_data' duration is {duration:.3f} minutes", file=file_time)
        print(f"len(data['data']) after select_data {len(data['data'])}")

        # Model optimize parameters
        if model_optimize_parameters:
            start_time = time.perf_counter()            
            model_optimized = model_regression_grid_search(data=data,
                                                           test_size=test_size,
                                                           random_state=random_state,
                                                           model_type=model_type,
                                                           model_name=model_name,
                                                           param_grid=param_grid,
                                                           n_jobs=n_jobs,
                                                           cross_validation_folds=cross_validation_folds,
                                                           unit_of_measure=unit_of_measure,
                                                           folder_output=folder_output,
                                                           folder_output_renew=False,
                                                           name_output=name_output)
            end_time = time.perf_counter()
            duration = (end_time - start_time)/60.0
            print(f"'model_regression_grid_search' duration is {duration:.3f} minutes", file=file_time)
        try:
            model_parameters = model_optimized.best_params_
        except NameError:
            pass

        # Model fit
        if model_fit:
            start_time = time.perf_counter()     
            model_regression_fit(data=data,
                                 test_size=test_size,
                                 random_state=random_state,
                                 model_type=model_type,
                                 model_name=model_name,
                                 model_parameters=model_parameters,
                                 cross_validation_folds=cross_validation_folds,
                                 unit_of_measure=unit_of_measure,
                                 folder_output=folder_output,
                                 folder_output_renew=False,
                                 name_output=name_output)
            end_time = time.perf_counter()
            duration = (end_time - start_time)/60.0
            print(f"'model_regression_fit' duration is {duration:.3f} minutes", file=file_time)

        # Plot results
        if model_plot_test:
            start_time = time.perf_counter()
            model_regression_test_plot(model_file=folder_output+'/'+name_output+'_model_'+model_name+'.pkl',
                                  data=data,
                                  test_size=test_size,
                                  random_state=random_state,
                                  feature_markersize=plot_test_parameters['feature_markersize'],
                                  markersize_multiply=plot_test_parameters['markersize_multiply'],                                  
                                  feature_color=plot_test_parameters['feature_color'],
                                  colorbar_label=plot_test_parameters['colorbar_label'],
                                  colorbar_labelpad=plot_test_parameters['colorbar_labelpad'],
                                  colorbar_position=plot_test_parameters['colorbar_position'],
                                  colormap=plot_test_parameters['colormap'],
                                  figure_size=plot_test_parameters['figure_size'],
                                  fontsize=plot_test_parameters['fontsize'],
                                  tick_size=plot_test_parameters['tick_size'],
                                  margins=plot_test_parameters['margins'],
                                  text_position_mae=plot_test_parameters['text_position_mae'],
                                  text_position_r2=plot_test_parameters['text_position_r2'],
                                  label_horizontal_axis=plot_test_parameters['label_horizontal_axis'],
                                  xlim=plot_test_parameters['xlim'],
                                  ylim=plot_test_parameters['ylim'],
                                  dpi=plot_test_parameters['dpi'],
                                  unit_of_measure=unit_of_measure,
                                  folder_output=folder_output,
                                  folder_output_renew=False,
                                  name_output=name_output,
                                  model_name=model_name)
            end_time = time.perf_counter()
            duration = (end_time - start_time)/60.0
            print(f"'model_regression_test_plot' duration is {duration:.3f} minutes", file=file_time)

        file_time.close()