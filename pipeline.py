import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

class Pipeline:
    def __init__(self):
        self.algorithms = {}
        self.data = {}
        self.errors = {}
        self.pair_plots = {}
        self.scatter_plots = {}
        self.scalers = {}
        self.predictions = {}
        self.trained_estimators = {}

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
            self.__original_data = self.data.copy()
        else:
            print('Name must be: train, test or validation.')
    
    def view_pairlot_of_data(self, dataset_name, hue=None):
        """ Uses data from self.data and returns a pairplot"""
        fig = sns.pairplot(self.data[dataset_name], hue=hue, height=2)
        self.pair_plots[dataset_name] = fig.figure
        # plt.ion()
    
    def check_num_uniques(self, dataset_name):
        """ Prints the number of unique values in a dataframe per column """
        print(self.data[dataset_name].nunique())

    def check_nans(self, dataset_name):
        """ Prints the number of nans in a dataframe per column """
        print(self.data[dataset_name].isna().sum())
    
    def assess_data(self, dataset_name):
        print(f"\t\t NUM UNIQUES in {dataset_name}:")
        self.check_num_uniques(dataset_name)
        print(f"\t\t NUM NANs in  {dataset_name}:")
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
    
    def scale_data(self, dataset_name, column_names, scaler_objs):
        """ Scales the data . Only works with sklearn
            methods that use .fit() then .transform(). Also stores the used scalers
            in a dict {column name: scaler object}"""
        for scaler_obj_key in scaler_objs.keys():
            scaler_obj = scaler_objs[scaler_obj_key]
            scalers_to_add = {}
            data = self.data[dataset_name].copy()
            for col in column_names:
                X = np.array(data[col]).reshape(-1,1)
                scaler_obj.fit(X)
                # data[col] = scaler_obj.transform(X)
                scalers_to_add[col]=scaler_obj
            self.scalers[str(scaler_obj)] = scalers_to_add

    def perform_scaling(self, dataset_name='train'):
        data = self.data[dataset_name].copy()
        for scaler_obj_name in self.scalers.keys():
            scaler = self.scalers[scaler_obj_name]
            for col in data.columns.values:
                if col in scaler.keys():
                    X = np.array(data[col]).reshape(-1,1)
                    data[col] = scaler[col].transform(X)
            self.data[f"{scaler_obj_name} {dataset_name.upper()}"] = data

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
        """ Add predictions for estimator """
        self.predictions[name] = predictions
    
    def set_trained_estimator(self, algo_name, estimator):
        """ Add an estimator to the estimator dict for comparisons"""
        self.trained_estimators[algo_name] = estimator
    
    def load_and_view_data(self, dict_of_data, label_name, display_descriptions=True, display_pair_plot=False):
        for key in dict_of_data.keys():
            self.add_data(key, dict_of_data[key])
            if display_descriptions:
                print(f"\t\t Description of {key} data: ")
                display(dict_of_data[key].describe())
        try:
            self.assess_data('train')
        except KeyError as e:
            print(f"assess_data Error: {e} \n No key 'train' in data. ")
        
        try:
            self.view_pairlot_of_data('train', hue=label_name)
        except KeyError as e:
            print(f"assess_data Error: {e} \n No such (label) column in data. ")
        
        if display_pair_plot:
            self.pair_plots['train'] 
    
    def setup_training_data(self, 
                            dataset_name='train', 
                            columns_to_drop=[], 
                            remove_na_rows=True, 
                            encode_non_numeric=True,
                            columns_to_scale=[],
                            scaler_obj={},
                            label_name=None):
        if columns_to_drop:
            print('Dropping Columns ...')
            try:
                self.remove_columns(dataset_name, columns_to_drop)
            except KeyError as e:
                print(f"Incorrect key used to select dataset or column to drop. \n Error {e}")
        if remove_na_rows:
            print('Removing Na Rows ...')
            try:
                self.remove_na_rows(dataset_name)
            except KeyError as e:
                print(f"Incorrect key used to select dataset. \n Error {e}")
        if encode_non_numeric:
            print('Encoding non-numeric data ...')
            try:
                self.encode_non_numeric_values(dataset_name)
            except KeyError as e:
                print(f"Incorrect key used to select dataset. \n Error {e}")
        if columns_to_scale and scaler_obj:
            print('Generating scaler objects ...')
            try:
                self.scale_data(dataset_name, columns_to_scale, scaler_obj)
            except KeyError as e:
                print(f"Incorrect key used to select dataset or column to scale. \n Error {e}")
    
            print('Scaling training data ...')
            try:
                self.perform_scaling()
            except KeyError as e:
                print(f"Unable to scale. \n Error {e}")

        if label_name:
            print('Assigning lables ...')
            try:
                self.set_labels(dataset_name, label_name)  
            except KeyError as e:
                print(f"Incorrect key used to select label column. \n Error {e}")      

        print('Completed Setup.')
        

        

        
