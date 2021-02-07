import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap


def plot_grid_search_model_complexity(gs_results, PLOT_PREFIX, unused_params_value_dict=None):
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
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')

    for i, plot_param in enumerate(param_names):
        mask = np.ones(np.array(train_scores_mean).shape, dtype=bool)
        title = ""
        for best_param, values in param_values.items():
            if plot_param != best_param:
                if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                    best_param_value = gs_results.best_params_[best_param]
                else:
                    best_param_value = unused_params_value_dict[best_param]
                mask = mask & np.where(np.array(values) == best_param_value,True,False)
                try:
                    title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
                except TypeError:
                    title += (best_param + " = " + best_param_value + " ")
        title = textwrap.fill(title, 20)


        x = np.array(np.array(param_values[plot_param])[mask])
        plot_test_scores = np.array(test_scores_mean)[mask]
        plot_test_std = np.array(test_scores_std)[mask]
        plot_train_scores = np.array(train_scores_mean)[mask]
        plot_train_std = np.array(train_scores_std)[mask]

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

        # ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        # ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(plot_param.upper())
        ax[i].legend(loc="upper left")
        ax[i].yaxis.set_tick_params(labelbottom=True)
        ax[i].set_title(title)
        ax[i].grid(True)

    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_PREFIX + "GS_ModelComplexity" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")


def plot_grid_search_training_times(gs_results, PLOT_PREFIX, unused_params_value_dict=None):
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
    fig.suptitle('Train / Score time per parameter')
    fig.text(0.04, 0.5, 'MEAN Time', va='center', rotation='vertical')

    for i, plot_param in enumerate(param_names):
        mask = np.ones(np.array(fit_time_means).shape, dtype=bool)
        title = ""
        for best_param, values in param_values.items():
            if plot_param != best_param:
                if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                    best_param_value = gs_results.best_params_[best_param]
                else:
                    best_param_value = unused_params_value_dict[best_param]
                mask = mask & np.where(np.array(values) == best_param_value,True,False)
                try:
                    title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
                except TypeError:
                    title += (best_param + " = " + best_param_value + " ")
        title = textwrap.fill(title, 20)
        x = np.array(np.array(param_values[plot_param])[mask])
        y_1 = np.array(fit_time_means)[mask]
        e_1 = np.array(fit_time_stds)[mask]
        y_2 = np.array(score_time_means)[mask]
        e_2 = np.array(score_time_stds)[mask]
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='fit time')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='score time' )
        ax[i].set_xlabel(plot_param.upper())
        ax[i].legend(loc="upper right")
        ax[i].yaxis.set_tick_params(labelbottom=True)
        ax[i].set_title(title)
        ax[i].grid(True)

    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_PREFIX + "GS_Times" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")

def plot_grid_search_model_complexity_and_training(gs_results, PLOT_PREFIX, unused_params_value_dict=None):
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
            if plot_param != best_param:
                if unused_params_value_dict is None or best_param not in unused_params_value_dict.keys():
                    best_param_value = gs_results.best_params_[best_param]
                else:
                    best_param_value = unused_params_value_dict[best_param]
                mask = mask & np.where(np.array(values) == best_param_value,True,False)
                try:
                    title += (best_param + " = " + str(round(best_param_value, 4)) + " ")
                except TypeError:
                    title += (best_param + " = " + best_param_value + " ")
        title = textwrap.fill(title, 20)


        x = np.array(np.array(param_values[plot_param])[mask])
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



    param_string = ""
    for param in param_names:
        param_string += ("_" + param)
    save_plot_name = PLOT_PREFIX + "GS_ModelComplexity" + param_string + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")


# From https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def plot_grid_search_2_params(gs_results, name_param_1, name_param_2, score_limit, PLOT_PREFIX, plot_counter=0, unused_params_value_dict=None):
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

    title = "Grid Search Scores "
    test_wrap_len = 21
    for key, value in unused_params.items():
        try:
            title += "{} = {} ".format(key, round(value, 4))
        except TypeError:
            title += "{} = {} ".format(key, value)

    title = textwrap.fill(title, test_wrap_len)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0.95, 1.0), fontsize=15)
    ax.grid('on')
    plt.subplots_adjust(left=0, right=0.8)

    save_plot_name = PLOT_PREFIX + "GridScore_" + name_param_1 + "_" + name_param_2 + "_" + str(plot_counter) + ".png"
    print("Plot saved as: ", save_plot_name)
    plt.savefig(save_plot_name, bbox_inches="tight")