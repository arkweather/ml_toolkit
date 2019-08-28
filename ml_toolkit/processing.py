"""
David Harrison
August 2019

This utility script contains functions to help prepare data for training
"""

from __future__ import print_function
from ml_toolkit import csv_io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import datetime


def combineData(data, date_index, names = []):
	"""Merge multiple pandas dataframes into one
	
	:param data: List of pandas dataframes with a common date index to merge along
	:param date_index: The name of the column containing the date of a given row
	:param names: List of names of each dataframe to append to the original column names
				  e.g. If the name of a model is "GFS", then the "precip" column becomes "GFS_precip"
				  Leave empty to use the original names of the columns in each dataframe
	:return: A new dataframe with the data merged along the date_index column (hour is ignored when merging)
	"""
	
	print('Combining dataframes...')
	if len(names) > 0 and len(names) != len(data): raise IndexError("The length of names must be 0 or the same length as data")
	
	# Iterate through each dataframe
	for i in range(len(data)):
		
		# Strip hour info from date index column
		data[i][date_index[i]] = pd.to_datetime(data[i][date_index[i]]).dt.date
		
		# Update column names
		if len(names) > 0: data[i].rename(columns = lambda x: names[i] + '_' + x, inplace = True)
		data[i].rename(columns = {names[i] + '_' + date_index[i]: 'date'}, inplace = True)
			
		
	# Merge dataframes on the date_index
	combined_data = data[0]
	for df in data[1:]:
		combined_data = pd.merge(combined_data, df, on='date', how = 'outer')
		
	return combined_data		
		
		
def fillNaNs(data):
	"""Iterates through each column in the dataframe and fills NaNs by averaging columns with similar names
	   e.g. If the NAM4km tf_max is missing on a given day, the value is filled by averaging the tf_max
	   fields of all other models on that day.  This may take a while if there are a lot of NaNs
	   
	:param data: Pandas dataframe with the data to process
	:return: A Pandas dataframe with the NaNs filled
	"""
	
	print('Filling NaN values...\n')
	
	for col in data.keys():
		
		if col == 'date': continue
		
		# Set bad values (strings) to NaN	
		pd.to_numeric(data[col], errors = 'coerce')
		
		# Get indices of all NaN values in the column
		nan_index = data[col].index[pd.isnull(data[col])]
		
		# Find all similar columns
		match_cols = [key for key in data.keys() if key.endswith('_'.join(col.split('_')[1:]))]
		print('Filling NaNs in ' + col)
		
		# Fill each NaN in the column
		for i in range(len(nan_index)):
			
			day_values = [data[mcol].iloc[nan_index[i]] for mcol in match_cols]
			day_values = [np.nan if type(value) == str else value for value in day_values] # Set bad values (strings) to NaN

			# Extra checks to make sure NaNs and strings are gone
			if np.isnan(np.nanmean(day_values)): data[col].iloc[nan_index[i]] = 0
			elif len(day_values) > 1: data[col].iloc[nan_index[i]] = np.nanmean(day_values)
			else: data[col].iloc[nan_index[i]] = 0.
			
	return data

	
def getTrainTest(data, frac = 0.75, random = False, date = None, sep = 1, seed = 1):
	"""Split the data into training and testing data sets	
	
	:param data: A Pandas object with the data to split
	:param frac: Fraction of the dataset to use for training
	:param random: Set to true to randomly sample the data when compiling the training set
				   If False, the data will be selected sequentially
	:param date: Datetime to use as the delineator between training and testing data.
				 All data prior to the specified date will be assigned to the training set
				 and all data after the specified date will be assigned to the test set.
				 This param overides the frac and random parameters.
	:param sep: Number of rows to skip between the training and testing sets to maintain
				test set independance
	:param seed: Seed to set the random state if random == True
	:return: A tuple of Pandas objects (training set, test set)
	"""
	
	# Delineate by date
	if date != None:
		train = data[data['date'] < date][:-sep].reset_index(drop=True)
		test = data[data['date'] >= date].reset_index(drop=True)
		
	# Otherwise, take fraction of dataset
	else:
		
		# Randomly select without replacement
		if random:
			train = data.sample(frac = frac, random_state = seed)
			test = data.drop(train.index)
			train.reset_index()
			test.reset_index()
			
		# Sequentially select
		else:
			train = data[:np.ceil(frac * data.shape[0])][:-sep].reset_index()
			test = data[np.ceil(frac * data.shape[0]):].reset_index()
			
	return (train, test)


def transformPCA(input_data, trained_model = None):
    """
    Transforms data by normalizing and applying principle component analysis.

    :param train: Pandas dataset of the input training data
    :param trained_model: PCA model trained on previous training dataset
    :return: Transformed input train and test data
    """
    
    input_data = StandardScaler().fit_transform(input_data)
    if trained_model is not None:
        output_data = trained_model.transform(input_data)   
        return output_data  
    else:
        pca = PCA(n_components = 0.997, svd_solver = 'full')
        output_data = pca.fit_transform(input_data)
        print('Number of components:', np.shape(output_data)[1])
        return pca, output_data
	
		
		
		
		
		
		
		
