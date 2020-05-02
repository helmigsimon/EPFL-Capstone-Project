import numbers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GridSearchPlotter:
    """
    Takes a GridSearchCV Object as input and plots the development of training loss and validation scores during cross-validation for a given hyperparameter
    Inspired by the function outlined in the following blogpost: https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results
    """
    def __init__(self,grid_search_cv):
        self.grid_search_cv = grid_search_cv
         
    def _create_slices(self,idx,parameter,base_parameter_idx):
        """
        Creates the index slices to be used to identify the relevant training and validation scores statistics 
        """
        if idx == base_parameter_idx:
            return slice(None)
        best_parameter_value = self.grid_search_cv.best_params_[parameter]
        best_parameter_idx = 0
        if isinstance(self.parameter_ranges[idx], np.ndarray):
            best_parameter_idx = self.parameter_ranges[idx].tolist().index(best_parameter_value)
        else:
            best_parameter_idx = self.parameter_ranges[idx].index(best_parameter_value)
        return best_parameter_idx

   
    def plot_validation_curve(self,parameter, title='Validation Curve', xlim=None, ylim=None, log=None):
        """Plots train and cross-validation scores from a GridSearchCV instance's best parameters while varying one of those parameters
        """

        df_cv_results = pd.DataFrame(self.grid_search_cv.cv_results_)

        train_scores_mean, valid_scores_mean = df_cv_results['mean_train_score'], df_cv_results['mean_test_score']
        train_scores_std, valid_scores_std = df_cv_results['std_train_score'], df_cv_results['std_test_score']
        
        self.parameter_columns = [column for column in df_cv_results.columns if 'param_' in column]
        self.parameter_ranges = [self.grid_search_cv.param_grid[p.replace('param_','')] for p in self.parameter_columns]
        self.parameter_ranges_lengths = [len(pr) for pr in self.parameter_ranges]

        train_scores_mean = np.array(train_scores_mean).reshape(*self.parameter_ranges_lengths)
        valid_scores_mean = np.array(valid_scores_mean).reshape(*self.parameter_ranges_lengths)
        train_scores_std = np.array(train_scores_std).reshape(*self.parameter_ranges_lengths)
        valid_scores_std = np.array(valid_scores_std).reshape(*self.parameter_ranges_lengths)

        base_parameter_idx = self.parameter_columns.index('param_{}'.format(parameter))

        slices = tuple(self._create_slices(idx,param,base_parameter_idx) for idx,param in enumerate(self.grid_search_cv.best_params_))

        train_scores_mean = train_scores_mean[slices]
        valid_scores_mean = valid_scores_mean[slices]
        train_scores_std = train_scores_std[slices]
        valid_scores_std = valid_scores_std[slices]
        
        plt.clf()

        plt.title(title)
        plt.xlabel(parameter)
        plt.ylabel('score')

        if not ylim:
            plt.ylim(0.0,1.1)
        else:
            plt.ylim(*ylim)

        if xlim:
            plt.xlim(*xlim)

        lw = 2
        
        plot_fn = plt.plot
        if log:
            plot_fn = plt.semilogx

        parameter_range = self.parameter_ranges[base_parameter_idx]

        if not isinstance(parameter_range[0],numbers.Number):
            parameter_range = [str(x) for x in parameter_range]

        plot_fn(parameter_range, train_scores_mean, label='Training score', color='r', lw=lw)
        plt.fill_between(parameter_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color='r', lw=lw)
        
        plot_fn(parameter_range,valid_scores_mean,label='Cross-validation score', color='b', lw=lw)
        plt.fill_between(parameter_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha =0.1, color='b', lw=lw)

        plt.legend(loc='lower right')

        plt.show()

