import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier

title_fontsize = 24
fontsize = 24
legend_fontsize = 18
def convert_nn_layers_parameter(hidden_layer_sizes):
    temp = ""
    for i in hidden_layer_sizes:
        temp += (str(i) + " ,")
    temp = temp[:-2]
    return temp


def convert_nn_layers_parameter_list(hidden_layer_sizes_list):
    string_hidden_layer_size = []
    for i in hidden_layer_sizes_list:
        string_hidden_layer_size.append(convert_nn_layers_parameter(i))
    return string_hidden_layer_size


def convert_boosting_base_estimator(base_estimator_parameter: DecisionTreeClassifier) -> str:
    return "ccp_alpha: " + str(base_estimator_parameter.ccp_alpha) + " max_depth: " + str(base_estimator_parameter.max_depth)


def convert_boosting_base_estimator_parameters_list(base_estimator_parameters_list):
    converted_parameters = []
    for i in base_estimator_parameters_list:
        converted_parameters.append(convert_boosting_base_estimator(i))
    return converted_parameters

def plot_grid_search_model_complexity_1param(gs_results, plot_param, PLOT_SAVE_LOCATION, ALGO, DATASET, unused_params_value_dict=None, tick_spacing=1, text_wrap_len=30, ylim=None, figsize=(15,10)):
    """
    References:
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

    :param gs_results:
    :return:
    """
    cv_results = gs_results.cv_results_
    cv_results = pd.DataFrame(cv_results)
    # Get Test Scores Mean and std for each grid search
    all_test_scores_mean = cv_results['mean_test_score']
    all_test_scores_sd = cv_results['std_test_score']
    all_train_scores_mean = cv_results['mean_train_score']
    all_train_scores_sd = cv_results['std_train_score']
    all_parameters = gs_results.cv_results_['params']

    param_names = []
    # Dictionary of all the parameter names as keys with the values of the parameters
    # matched with the score and std arrays as the values
    param_values = {}
    for i in gs_results.cv_results_['params'][0].keys():
        param_names.append(i)
        param_values[i] = []

    test_scores_mean = []
    test_scores_std = []
    train_scores_mean = []
    train_scores_std = []
    # Get arrays of scores, standard deviations and the value of each parameter
    for train_mean, train_std, test_mean, test_std, params in \
            zip(all_train_scores_mean, all_train_scores_sd, all_test_scores_mean, all_test_scores_sd, all_parameters):
        train_scores_mean.append(train_mean)
        train_scores_std.append(train_std)
        test_scores_mean.append(test_mean)
        test_scores_std.append(test_std)
        for p in param_names:
            param_values[p].append(params[p])

    ## Ploting results
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='all',figsize=figsize)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    fig.suptitle('{} Model Complexity Curve for Parameter {}\n Data Set: {}'.format(ALGO, plot_param, DATASET), fontsize=title_fontsize, fontweight='bold')
    fig.text(-0.02, 0.5, 'Mean Score (Accuracy)', va='center', rotation='vertical', fontsize=fontsize)

    i=0
    mask = np.ones(np.array(train_scores_mean).shape, dtype=bool)
    title = ""
    for best_param, values in param_values.items():
        if isinstance(values[0], list):
            values = convert_nn_layers_parameter_list(values)
        elif isinstance(values[0], DecisionTreeClassifier):
            values = convert_boosting_base_estimator_parameters_list(values)
        if plot_param != best_param:
            if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                best_param_value = gs_results.best_params_[best_param]
            else:
                best_param_value = unused_params_value_dict[best_param]
            if isinstance(best_param_value, list):
                best_param_value = convert_nn_layers_parameter(best_param_value)
            elif isinstance(best_param_value, DecisionTreeClassifier):
                best_param_value = convert_boosting_base_estimator(best_param_value)
            mask = mask & np.where(np.array(values) == best_param_value,True,False)
            try:
                title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
            except TypeError:
                title += (best_param + " = " + best_param_value + " ")
    title = textwrap.fill(title, text_wrap_len)

    x = np.array(np.array(param_values[plot_param])[mask])
    plot_test_scores = np.array(test_scores_mean)[mask]
    plot_test_std = np.array(test_scores_std)[mask]
    plot_train_scores = np.array(train_scores_mean)[mask]
    plot_train_std = np.array(train_scores_std)[mask]

    if isinstance(x[0], list):
        x = convert_nn_layers_parameter_list(x)
        ax.xaxis.set_tick_params(rotation=90)
    elif isinstance(x[0], DecisionTreeClassifier):
        x = convert_boosting_base_estimator_parameters_list(x)
        ax.xaxis.set_tick_params(rotation=90)

    ax.plot(x, plot_test_scores, label="Cross-validation Score",
               color="navy", marker=".")
    ax.fill_between(x, plot_test_scores - plot_test_std,
                       plot_test_scores + plot_test_std, alpha=0.2,
                       color="navy", lw=2)
    ax.plot(x, plot_train_scores, label="Training Score",
               color="darkorange", marker=".")
    ax.fill_between(x, plot_train_scores - plot_train_std,
                       plot_train_scores + plot_train_std, alpha=0.2,
                       color="darkorange", lw=2)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax.xaxis.set_ticks(x[::tick_spacing])
    ax.set_xlabel(plot_param.upper(), fontsize=fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", fontsize=legend_fontsize)
    ax.yaxis.set_tick_params(labelbottom=True)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)

    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_SAVE_LOCATION + DATASET.replace(" ", "_") + "_" + ALGO + "_" + "GS_ModelComplexity" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    #plt.savefig(save_plot_name, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_grid_search_model_complexity(gs_results, PLOT_SAVE_LOCATION, ALGO, DATASET, unused_params_value_dict=None, tick_spacing=None, text_wrap_len=30, ylim=None):
    """
    References:
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

    :param gs_results:
    :return:
    """
    cv_results = gs_results.cv_results_
    cv_results = pd.DataFrame(cv_results)
    # Get Test Scores Mean and std for each grid search
    all_test_scores_mean = cv_results['mean_test_score']
    all_test_scores_sd = cv_results['std_test_score']
    all_train_scores_mean = cv_results['mean_train_score']
    all_train_scores_sd = cv_results['std_train_score']
    all_parameters = gs_results.cv_results_['params']

    param_names = []
    # Dictionary of all the parameter names as keys with the values of the parameters
    # matched with the score and std arrays as the values
    param_values = {}
    for i in gs_results.cv_results_['params'][0].keys():
        param_names.append(i)
        param_values[i] = []

    test_scores_mean = []
    test_scores_std = []
    train_scores_mean = []
    train_scores_std = []
    # Get arrays of scores, standard deviations and the value of each parameter
    for train_mean, train_std, test_mean, test_std, params in \
        zip(all_train_scores_mean, all_train_scores_sd, all_test_scores_mean, all_test_scores_sd, all_parameters):
        train_scores_mean.append(train_mean)
        train_scores_std.append(train_std)
        test_scores_mean.append(test_mean)
        test_scores_std.append(test_std)
        for p in param_names:
            param_values[p].append(params[p])

    ## Ploting results
    fig, ax = plt.subplots(1,len(param_names),sharex='none', sharey='all',figsize=(20,10))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.suptitle('{} Model Complexity Curves per Parameter\n Data Set: {}'.format(ALGO, DATASET), fontsize=title_fontsize, fontweight='bold')
    fig.text(-0.02, 0.5, 'Mean Score (Accuracy)', va='center', rotation='vertical', fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    for i, plot_param in enumerate(param_names):
        mask = np.ones(np.array(train_scores_mean).shape, dtype=bool)
        title = ""
        for best_param, values in param_values.items():
            if isinstance(values[0], list):
                values = convert_nn_layers_parameter_list(values)
            elif isinstance(values[0], DecisionTreeClassifier):
                values = convert_boosting_base_estimator_parameters_list(values)
            if plot_param != best_param:
                if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                    best_param_value = gs_results.best_params_[best_param]
                else:
                    best_param_value = unused_params_value_dict[best_param]
                if isinstance(best_param_value, list):
                    best_param_value = convert_nn_layers_parameter(best_param_value)
                elif isinstance(best_param_value, DecisionTreeClassifier):
                    best_param_value = convert_boosting_base_estimator(best_param_value)
                mask = mask & np.where(np.array(values) == best_param_value,True,False)
                try:
                    title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
                except TypeError:
                    title += (best_param + " = " + best_param_value + " ")
        title = textwrap.fill(title, text_wrap_len)

        x = np.array(np.array(param_values[plot_param])[mask])
        plot_test_scores = np.array(test_scores_mean)[mask]
        plot_test_std = np.array(test_scores_std)[mask]
        plot_train_scores = np.array(train_scores_mean)[mask]
        plot_train_std = np.array(train_scores_std)[mask]

        rotation=0
        if isinstance(x[0], list):
            x = convert_nn_layers_parameter_list(x)
            ax[i].xaxis.set_tick_params(rotation=90)
        elif isinstance(x[0], DecisionTreeClassifier):
            x = convert_boosting_base_estimator_parameters_list(x)
            ax[i].xaxis.set_tick_params(rotation=90)

        ax[i].plot(x, plot_test_scores, label="Cross-validation Score",
                 color="navy", marker=".")
        ax[i].fill_between(x, plot_test_scores - plot_test_std,
                         plot_test_scores + plot_test_std, alpha=0.2,
                         color="navy", lw=2)
        ax[i].plot(x, plot_train_scores, label="Training Score",
                 color="darkorange", marker=".")
        ax[i].fill_between(x, plot_train_scores - plot_train_std,
                         plot_train_scores + plot_train_std, alpha=0.2,
                         color="darkorange", lw=2)

        if tick_spacing is not None:
            ax[i].xaxis.set_ticks(x[::tick_spacing[i]])
        plt.setp(ax[i].get_xticklabels(), rotation=rotation, fontsize=fontsize)
        ax[i].set_xlabel(plot_param.upper(), fontsize=fontsize)
        if ylim is not None:
            ax[i].set_ylim(*ylim)
        ax[i].legend(loc="upper left", fontsize=legend_fontsize)
        ax[i].yaxis.set_tick_params(labelbottom=True)
        ax[i].set_title(title, fontsize=fontsize)
        ax[i].grid(True)

    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_SAVE_LOCATION + DATASET.replace(" ", "_") + "_" + ALGO + "_" + "GS_ModelComplexity" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_grid_search_training_times_1param(gs_results, plot_param, PLOT_SAVE_LOCATION, ALGO, DATASET, unused_params_value_dict=None, tick_spacing=None, text_wrap_len=30, ylim=None, figsize=(15,10)):
    """
    References:
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

    :param gs_results:
    :return:
    """
    cv_results = gs_results.cv_results_
    cv_results = pd.DataFrame(cv_results)
    # Get Test Scores Mean and std for each grid search
    all_fit_times_mean = cv_results['mean_fit_time']
    all_fit_times_sd = cv_results['std_fit_time']
    all_score_times_mean = cv_results['mean_score_time']
    all_score_times_sd = cv_results['std_score_time']
    all_parameters = gs_results.cv_results_['params']

    param_names = []
    # Dictionary of all the parameter names as keys with the values of the parameters
    # matched with the score and std arrays as the values
    param_values = {}
    for i in gs_results.cv_results_['params'][0].keys():
        param_names.append(i)
        param_values[i] = []

    fit_time_means = []
    fit_time_stds = []
    score_time_means = []
    score_time_stds = []
    # Get arrays of scores, standard deviations and the value of each parameter
    for fit_mean, fit_std, score_mean, score_std, params in \
        zip(all_fit_times_mean, all_fit_times_sd, all_score_times_mean, all_score_times_sd, all_parameters):
        fit_time_means.append(fit_mean)
        fit_time_stds.append(fit_std)
        score_time_means.append(score_mean)
        score_time_stds.append(score_std)
        for p in param_names:
            param_values[p].append(params[p])

    ## Ploting results
    fig, ax = plt.subplots(1, 1,sharex='none', sharey='all',figsize=figsize)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    fig.suptitle('{} Training and Prediction time for Parameter {}\n Data Set: {}'.format(ALGO, plot_param, DATASET), fontsize=title_fontsize, fontweight='bold')
    fig.text(-0.02, 0.5, 'Mean Time (s)', va='center', rotation='vertical', fontsize=fontsize)

    mask = np.ones(np.array(fit_time_means).shape, dtype=bool)
    title = ""
    for best_param, values in param_values.items():
        if isinstance(values[0], list):
            values = convert_nn_layers_parameter_list(values)
        elif isinstance(values[0], DecisionTreeClassifier):
            values = convert_boosting_base_estimator_parameters_list(values)
        if plot_param != best_param:
            if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                best_param_value = gs_results.best_params_[best_param]
            else:
                best_param_value = unused_params_value_dict[best_param]
            if isinstance(best_param_value, list):
                best_param_value = convert_nn_layers_parameter(best_param_value)
            elif isinstance(best_param_value, DecisionTreeClassifier):
                best_param_value = convert_boosting_base_estimator(best_param_value)
            mask = mask & np.where(np.array(values) == best_param_value,True,False)
            try:
                title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
            except TypeError:
                title += (best_param + " = " + best_param_value + " ")
    title = textwrap.fill(title, text_wrap_len)
    x = np.array(np.array(param_values[plot_param])[mask])
    rotation=0
    if isinstance(x[0], list):
        x = convert_nn_layers_parameter_list(x)
        ax.xaxis.set_tick_params(rotation=90)
    elif isinstance(x[0], DecisionTreeClassifier):
        x = convert_boosting_base_estimator_parameters_list(x)
        ax.xaxis.set_tick_params(rotation=90)

    train_time_mean = np.array(fit_time_means)[mask]
    train_time_std = np.array(fit_time_stds)[mask]
    predict_time_mean = np.array(score_time_means)[mask]
    predict_time_std = np.array(score_time_stds)[mask]

    ax.plot(x, train_time_mean, label="Training Time", marker=".")
    ax.fill_between(x, train_time_mean - train_time_std, train_time_mean + train_time_std, alpha=0.2, lw=2)
    ax.plot(x, predict_time_mean, label="Prediction Time", marker=".")
    ax.fill_between(x, predict_time_mean - predict_time_std, predict_time_mean + predict_time_std, alpha=0.2, lw=2)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax.set_xlabel(plot_param.upper(), fontsize=fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.yaxis.set_tick_params(labelbottom=True)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)
    if tick_spacing is not None:
        ax.xaxis.set_ticks(x[::tick_spacing])
    plt.setp(ax.get_xticklabels(), rotation=rotation)
    if ylim is not None:
        ax.set_ylim(*ylim)

    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_SAVE_LOCATION + DATASET.replace(" ", "_") + "_" + ALGO + "_" + "GS_Times" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_grid_search_training_times(gs_results, PLOT_SAVE_LOCATION, ALGO, DATASET, unused_params_value_dict=None, tick_spacing=None, text_wrap_len=30, ylim=None):
    """
    References:
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

    :param gs_results:
    :return:
    """
    cv_results = gs_results.cv_results_
    cv_results = pd.DataFrame(cv_results)
    # Get Test Scores Mean and std for each grid search
    all_fit_times_mean = cv_results['mean_fit_time']
    all_fit_times_sd = cv_results['std_fit_time']
    all_score_times_mean = cv_results['mean_score_time']
    all_score_times_sd = cv_results['std_score_time']
    all_parameters = gs_results.cv_results_['params']

    param_names = []
    # Dictionary of all the parameter names as keys with the values of the parameters
    # matched with the score and std arrays as the values
    param_values = {}
    for i in gs_results.cv_results_['params'][0].keys():
        param_names.append(i)
        param_values[i] = []

    fit_time_means = []
    fit_time_stds = []
    score_time_means = []
    score_time_stds = []
    # Get arrays of scores, standard deviations and the value of each parameter
    for fit_mean, fit_std, score_mean, score_std, params in \
        zip(all_fit_times_mean, all_fit_times_sd, all_score_times_mean, all_score_times_sd, all_parameters):
        fit_time_means.append(fit_mean)
        fit_time_stds.append(fit_std)
        score_time_means.append(score_mean)
        score_time_stds.append(score_std)
        for p in param_names:
            param_values[p].append(params[p])

    ## Ploting results
    fig, ax = plt.subplots(1,len(param_names),sharex='none', sharey='all',figsize=(20,10))
    fig.suptitle('{} Training and Prediction time per parameter\n Data Set: {}'.format(ALGO, DATASET), fontsize=14, fontweight='bold')
    fig.text(0.04, 0.5, 'Mean Time (s)', va='center', rotation='vertical', fontsize=14)

    for i, plot_param in enumerate(param_names):
        mask = np.ones(np.array(fit_time_means).shape, dtype=bool)
        title = ""
        for best_param, values in param_values.items():
            if isinstance(values[0], list):
                values = convert_nn_layers_parameter_list(values)
            elif isinstance(values[0], DecisionTreeClassifier):
                values = convert_boosting_base_estimator_parameters_list(values)
            if plot_param != best_param:
                if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                    best_param_value = gs_results.best_params_[best_param]
                else:
                    best_param_value = unused_params_value_dict[best_param]
                if isinstance(best_param_value, list):
                    best_param_value = convert_nn_layers_parameter(best_param_value)
                elif isinstance(best_param_value, DecisionTreeClassifier):
                    best_param_value = convert_boosting_base_estimator(best_param_value)
                mask = mask & np.where(np.array(values) == best_param_value,True,False)
                try:
                    title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
                except TypeError:
                    title += (best_param + " = " + best_param_value + " ")
        title = textwrap.fill(title, text_wrap_len)
        x = np.array(np.array(param_values[plot_param])[mask])
        rotation=0
        if isinstance(x[0], list):
            x = convert_nn_layers_parameter_list(x)
            ax[i].xaxis.set_tick_params(rotation=90)
        elif isinstance(x[0], DecisionTreeClassifier):
            x = convert_boosting_base_estimator_parameters_list(x)
            ax[i].xaxis.set_tick_params(rotation=90)

        train_time_mean = np.array(fit_time_means)[mask]
        train_time_std = np.array(fit_time_stds)[mask]
        predict_time_mean = np.array(score_time_means)[mask]
        predict_time_std = np.array(score_time_stds)[mask]
        # ax[i].errorbar(x, train_time_mean, train_time_std, linestyle='--', marker='o', label='fit time')
        # ax[i].errorbar(x, predict_time_mean, predict_time_std, linestyle='-', marker='^',label='score time' )

        ax[i].plot(x, train_time_mean, label="Training Time", marker=".")
        ax[i].fill_between(x, train_time_mean - train_time_std, train_time_mean + train_time_std, alpha=0.2, lw=2)
        ax[i].plot(x, predict_time_mean, label="Prediction Time", marker=".")
        ax[i].fill_between(x, predict_time_mean - predict_time_std, predict_time_mean + predict_time_std, alpha=0.2, lw=2)


        ax[i].set_xlabel(plot_param.upper(), fontsize=14)
        ax[i].legend(loc="upper right")
        if ylim is not None:
            ax[i].set_ylim(*ylim)
        ax[i].yaxis.set_tick_params(labelbottom=True)
        ax[i].set_title(title, fontsize=14)
        ax[i].grid(True)
        if tick_spacing is not None:
            ax[i].xaxis.set_ticks(x[::tick_spacing[i]])
        plt.setp(ax[i].get_xticklabels(), rotation=rotation)

    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_SAVE_LOCATION + DATASET.replace(" ", "_") + "_" + ALGO + "_" + "GS_Times" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")

def plot_grid_search_model_complexity_and_training(gs_results, PLOT_PREFIX, unused_params_value_dict=None, tick_spacing=1):
    """
    References:
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

    :param gs_results:
    :return:
    """
    cv_results = gs_results.cv_results_
    cv_results = pd.DataFrame(cv_results)
    # Get Test Scores Mean and std for each grid search
    all_test_scores_mean = cv_results['mean_test_score']
    all_test_scores_sd = cv_results['std_test_score']
    all_train_scores_mean = cv_results['mean_train_score']
    all_train_scores_sd = cv_results['std_train_score']

    all_fit_times_mean = cv_results['mean_fit_time']
    all_fit_times_sd = cv_results['std_fit_time']
    all_score_times_mean = cv_results['mean_score_time']
    all_score_times_sd = cv_results['std_score_time']

    all_parameters = gs_results.cv_results_['params']

    param_names = []
    # Dictionary of all the parameter names as keys with the values of the parameters
    # matched with the score and std arrays as the values
    param_values = {}
    for i in gs_results.cv_results_['params'][0].keys():
        param_names.append(i)
        param_values[i] = []

    test_scores_mean = []
    test_scores_std = []
    train_scores_mean = []
    train_scores_std = []

    fit_time_means = []
    fit_time_stds = []
    score_time_means = []
    score_time_stds = []

    # Get arrays of scores, standard deviations and the value of each parameter
    for train_mean, train_std, test_mean, test_std, params in \
        zip(all_train_scores_mean, all_train_scores_sd, all_test_scores_mean, all_test_scores_sd, all_parameters):
        train_scores_mean.append(train_mean)
        train_scores_std.append(train_std)
        test_scores_mean.append(test_mean)
        test_scores_std.append(test_std)
        for p in param_names:
            param_values[p].append(params[p])

    for fit_mean, fit_std, score_mean, score_std, params in \
        zip(all_fit_times_mean, all_fit_times_sd, all_score_times_mean, all_score_times_sd, all_parameters):
        fit_time_means.append(fit_mean)
        fit_time_stds.append(fit_std)
        score_time_means.append(score_mean)
        score_time_stds.append(score_std)

    ## Ploting results
    fig, ax = plt.subplots(2,len(param_names),sharex='none', sharey='row',figsize=(20,10))
    fig.suptitle('Accuracy and Time per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')

    for i, plot_param in enumerate(param_names):
        mask = np.ones(np.array(train_scores_mean).shape, dtype=bool)
        title = ""
        for best_param, values in param_values.items():
            if isinstance(values[0], list):
                values = convert_nn_layers_parameter_list(values)
            elif isinstance(values[0], DecisionTreeClassifier):
                values = convert_boosting_base_estimator_parameters_list(values)
            if plot_param != best_param:
                if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                    best_param_value = gs_results.best_params_[best_param]
                else:
                    best_param_value = unused_params_value_dict[best_param]
                if isinstance(best_param_value, list):
                    best_param_value = convert_nn_layers_parameter(best_param_value)
                elif isinstance(best_param_value, DecisionTreeClassifier):
                    best_param_value = convert_boosting_base_estimator(best_param_value)
                mask = mask & np.where(np.array(values) == best_param_value,True,False)
                try:
                    title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
                except TypeError:
                    title += (best_param + " = " + best_param_value + " ")
        title = textwrap.fill(title, 20)


        x = np.array(np.array(param_values[plot_param])[mask])
        if isinstance(x[0], list):
            x = convert_nn_layers_parameter_list(x)
            ax[0, i].xaxis.set_tick_params(rotation=90)
            ax[1, i].xaxis.set_tick_params(rotation=90)
        elif isinstance(x[0], DecisionTreeClassifier):
            x = convert_boosting_base_estimator_parameters_list(x)
            ax[0, i].xaxis.set_tick_params(rotation=90)
            ax[1, i].xaxis.set_tick_params(rotation=90)

        plot_test_scores = np.array(test_scores_mean)[mask]
        plot_test_std = np.array(test_scores_std)[mask]
        plot_train_scores = np.array(train_scores_mean)[mask]
        plot_train_std = np.array(train_scores_std)[mask]

        y_1 = np.array(fit_time_means)[mask]
        e_1 = np.array(fit_time_stds)[mask]
        y_2 = np.array(score_time_means)[mask]
        e_2 = np.array(score_time_stds)[mask]

        ax[0, i].plot(x, plot_test_scores, label="Cross-validation Score",
                 color="navy", marker=".")
        ax[0, i].fill_between(x, plot_test_scores - plot_test_std,
                         plot_test_scores + plot_test_std, alpha=0.2,
                         color="navy", lw=2)
        ax[0, i].plot(x, plot_train_scores, label="Training Score",
                 color="darkorange", marker=".")
        ax[0, i].fill_between(x, plot_train_scores - plot_train_std,
                         plot_train_scores + plot_train_std, alpha=0.2,
                         color="darkorange", lw=2)

        ax[0, i].legend(loc="upper left")
        ax[0, i].yaxis.set_tick_params(labelbottom=True)
        ax[0, i].set_title(title)
        ax[0, i].grid(True)

        ax[1, i].plot(x, y_1, label="Fit Time", marker=".")
        ax[1, i].fill_between(x, y_1 - e_1, y_1 + e_1, alpha=0.2, lw=2)
        ax[1, i].plot(x, y_2, label="Score Time", marker=".")
        ax[1, i].fill_between(x, y_2 - e_2, y_2 + e_2, alpha=0.2, lw=2)

        ax[1, i].set_xlabel(plot_param.upper())
        ax[1, i].legend(loc="upper left")
        ax[1, i].yaxis.set_tick_params(labelbottom=True)
        ax[1, i].grid(True)
        # plt.setp(ax[0, i].get_xticklabels(), rotation=rotation)
        # plt.setp(ax[1, i].get_xticklabels(), rotation=rotation)
        ax[0, i].xaxis.set_ticks(x[::tick_spacing])
        ax[1, i].xaxis.set_ticks(x[::tick_spacing])


    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_PREFIX + "GS_ModelComplexity" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")


# From https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def plot_grid_search_2_params(gs_results, name_param_1, name_param_2, score_limit, PLOT_SAVE_LOCATION, ALGO, DATASET, plot_counter=0, text_wrap_len=30, unused_params_value_dict=None, tick_spacing=None):
    cv_results = pd.DataFrame(gs_results.cv_results_)
    # Get Test Scores Mean and std for each grid search
    all_scores_mean = cv_results['mean_test_score']
    all_scores_sd = cv_results['std_test_score']
    all_parameters = gs_results.cv_results_['params']

    unused_params = {}
    unused_params_values = {}
    for i in gs_results.cv_results_['params'][0].keys():
        if i != name_param_2 and i != name_param_1:
            if unused_params_value_dict is not None:
                unused_params[i] = unused_params_value_dict[i]
            else:
                unused_params[i] = gs_results.best_params_[i]
            unused_params_values[i] = []

    scores_mean = []
    scores_std = []
    parameter_1_values = []
    parameter_2_values = []

    for mean, std, params in zip(all_scores_mean, all_scores_sd, all_parameters):
        scores_mean.append(mean)
        scores_std.append(std)
        parameter_1_values.append(params[name_param_1])
        parameter_2_values.append(params[name_param_2])
        for p, v in unused_params.items():
            unused_params_values[p].append(params[p])

    scores_mean = np.array(scores_mean)
    parameter_1_values = np.array(parameter_1_values)
    scores_matrix = []
    parameter_1_matrix = []
    grid_param_2 = []
    for i in np.unique(parameter_2_values):
        mask = np.ones(parameter_1_values.shape, dtype=bool)
        for p, v in unused_params_values.items():
            mask = mask & np.where(np.array(v) == unused_params[p],True,False)
        scores = scores_mean[mask & np.array(np.array(parameter_2_values) == i)]
        if np.mean(scores) > score_limit or np.median(scores) > score_limit or scores[0] > score_limit:
            scores_matrix.append(scores)
            parameter_1_matrix.append(parameter_1_values[mask & np.array(np.array(parameter_2_values) == i)])
            grid_param_2.append(i)

    # for i in range(len(scores_matrix)):
    #     scores_matrix[i] = [score for p,score in sorted(zip(parameter_1_matrix[i], scores_matrix[i]))]
    #     parameter_1_matrix[i] = [p for p,score in sorted(zip(parameter_1_matrix[i], scores_matrix[i]))]

    # Plot Grid search scores
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a "different curve (color line)
    for idx, val in enumerate(grid_param_2):
            ax.plot(parameter_1_matrix[idx], scores_matrix[idx], '-o', linestyle=None, label= name_param_2 + ': ' + str(round(val, 5)))

    title = "{} Grid Search Scores ".format(ALGO)
    for key, value in unused_params.items():
        try:
            title += "{} = {} ".format(key, round(value, 4))
        except TypeError:
            title += "{} = {} ".format(key, value)

    title += "\n Data Set: {}".format(DATASET)
    title = textwrap.fill(title, text_wrap_len)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0.95, 1.0), fontsize=15)
    ax.grid('on')
    plt.subplots_adjust(left=0, right=0.8)
    if tick_spacing is not None:
        ax.xaxis.set_ticks(parameter_1_matrix[0][::tick_spacing])

    save_plot_name = PLOT_SAVE_LOCATION + DATASET.replace(" ", "_") + "_" + ALGO + "_" + "GridScore_" + name_param_1 + "_" + name_param_2 + "_" + str(plot_counter) + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")