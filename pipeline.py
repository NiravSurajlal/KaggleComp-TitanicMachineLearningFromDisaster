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

    def add_algorithm(self, algorithm, algo_name):
        """ Adds OBJECTS to the pipeline algorithms dict."""
        if str(algo_name) in self.algorithms:
            print('Name already referenced. Please try another.')
        else: 
            self.algorithms[str(algo_name)] = algorithm
    
    def add_data(self, name, data):
        """ Takes data in the form of a dataframe and stores it in a dict """
        if name in ['train', 'test', 'validation']:
            self.data[name] = data
            self.original_data = self.data.copy()
        else:
            print('Name must be: train, test or validation.')
    
    def view_pairlot(self, data, hue=None):
        """ Uses data from self.data and returns a pairplot"""
        fig = plt.figure()
        sns.set_style("whitegrid");
        sns.pairplot(data, hue=hue, height=2);
        return fig
    
    def check_num_uniques(self, dataset_name):
        """ Prints the number of unique values in a dataframe per column """
        print(self.data[dataset_name].nunique())

    def check_nans(self, dataset_name):
        """ Prints the number of nans in a dataframe per column """
        print(self.data[dataset_name].isna().sum())
    
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
        self.data[col]=df
    
    def scale_data(self, dataset_name, column_names, scaler):
        """ Scales the data and overwrites self.data dict. Only works with sklearn
            methods that use .fit() then .transform(). Also stores the used scalers
            in a dict {column name: scaler object}"""
        self.scalers = {}
        data = self.data[dataset_name].copy()
        for col in column_names:
            scaler.fit(np.array(data[col]).reshape(-1,1))
            data[col] = scaler.transform()
            self.scalers[col] = scaler
        self.data[dataset_name] = data    

    def set_labels(self, dataset_name, label_column):
        """ Returns the features and lable dataframes. Only used during supervised/
            reinforcement learning. """
        data = self.data[dataset_name].copy()
        labels = data[label_column]
        features = data.drop(columns=label_column)
        return features, labels

    def get_errors(self, labels, predictions):
        """ Returns average errors & how the error changes. Only used during supervised/
            reinforcement learning. """
        errors = np.array()
        errors = np.append(errors, np.array(predictions-labels))
        error = np.mean(errors)
        avg_error_over_time = np.array()
        for i in range(0, len(errors)):
            total = np.sum(errors[0:i])
            avg_error_over_time = np.append(total/(i+1))
        return error, avg_error_over_time
    
    def plot_scatter(self, y, x=None):
        """ Returns figure object of the scatter plot. """
        fig = plt.figure()
        if x is None:
            plt.scatter(np.arange(0, len(y)), (y), marker='.')
        else:
            plt.scatter((x), (y), marker='.')
        return fig
        
