import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self):
        self.algorithms = {}
        self.data = {}
        self.errors = {}
        self.pair_plots = {}
        self.scatter_plots = {}
        self.scalers = {}
        self.predictions = {}

    def add_algorithm(self, algorithm, algo_name):
        """ Adds OBJECTS to the pipeline algorithms dict."""
        if str(algo_name) in self.algorithms:
            print('Name already referenced. Please try another.')
        else: 
            self.algorithms[str(algo_name)] = algorithm
    
    def add_data(self, name, data):
        """ Takes data in the form of a dataframe and stores it in a dict """
        if name in ['train', 'test', 'validation']:
            self.data[name] = data.copy()
            self.original_data = self.data.copy()
        else:
            print('Name must be: train, test or validation.')
    
    def view_pairlot_of_data(self, dataset_name, hue=None):
        """ Uses data from self.data and returns a pairplot"""
        fig = sns.pairplot(self.data[dataset_name], hue=hue, height=2)
        self.pair_plots[dataset_name] = fig.figure
    
    def check_num_uniques(self, dataset_name):
        """ Prints the number of unique values in a dataframe per column """
        print(self.data[dataset_name].nunique())

    def check_nans(self, dataset_name):
        """ Prints the number of nans in a dataframe per column """
        print(self.data[dataset_name].isna().sum())
    
    def assess_data(self, dataset_name):
        print("\t\t NUM UNIQUES:")
        self.check_num_uniques(dataset_name)
        print("\t\t NUM NANs:")
        self.check_nans(dataset_name)

    def adjust_data(self, data, dataset_name) -> None:
        """ Allows a readjustment of the data """
        self.data[dataset_name] = data
    
    def encode_non_numeric_values(self, dataset_name):
        """ Encodes columns of non-numeric (!= int or float) data by assigning a 
            uninque integer to each. Iterates from 0. It then overwrites the current
            data dict. """
        df = self.data[dataset_name]
        columns = df.columns.values
        for col in columns:
            if df[col].dtype != int and df[col].dtype != float:
                unique_vals = df[col].unique()
                reassign_dict = {unique_vals[i]: i for i in range(0, len(unique_vals), 1)}
                df = df.replace({col: reassign_dict})
        self.data[dataset_name] = df
    
    def scale_data(self, dataset_name, column_names, scaler_obj):
        """ Scales the data and overwrites self.data dict. Only works with sklearn
            methods that use .fit() then .transform(). Also stores the used scalers
            in a dict {column name: scaler object}"""
        data = self.data[dataset_name].copy()
        for col in column_names:
            X = np.array(data[col]).reshape(-1,1)
            scaler_obj.fit(X)
            data[col] = scaler_obj.transform(X)
            self.scalers[col] = scaler_obj
        self.data[dataset_name] = data    

    def set_labels(self, dataset_name, label_column):
        """ Returns the features and lable dataframes. Only used during supervised/
            reinforcement learning. """
        data = self.data[dataset_name].copy()
        self.labels = data[label_column]
        self.features = data.drop(columns=label_column)
        return self.features, self.labels

    def get_errors(self, predictions, estimator_name):
        """ Returns average errors & how the error changes. Only used during supervised/
            reinforcement learning. Saves to self.errors per estimator as
            (error, error_change per prediction)"""
        labels = self.labels.copy()
        errors = np.empty(shape=(1,1))
        errors = np.append(errors, np.array(predictions-labels))
        error = np.mean(errors)
        avg_error_over_time = np.empty(shape=(1,1))
        for i in range(0, len(errors)):
            total = np.sum(errors[0:i])
            avg_error_over_time = np.append(avg_error_over_time, total/(i+1))
        self.errors[estimator_name] = {'error': errors, 'delta_error': avg_error_over_time}
    
    def plot_scatter(self, y, name, x=None):
        """ Returns figure object of the scatter plot. """
        fig = plt.figure()
        if x is None:
            plt.scatter(np.arange(0, len(y)), (y), marker='.')
        else:
            plt.scatter((x), (y), marker='.')
        self.scatter_plots[name] = fig
    
    def remove_columns(self, dataset_name, columns_to_drop):
        """ Takes list of column names and drops them from the
            required dataset, overwriting it. """
        print("Data shape:", np.shape(self.data[dataset_name]))
        for col in columns_to_drop:
            self.data[dataset_name].drop(columns=[col], inplace=True)
        print("Data shape:", np.shape(self.data[dataset_name]))

    def remove_na_rows(self, dataset_name):
        print("Data shape:", np.shape(self.data[dataset_name]))
        self.data[dataset_name].dropna(inplace=True)
        print("Data shape:", np.shape(self.data[dataset_name]))

    def add_predictions(self, name, predictions):
        self.predictions[name] = predictions
