"""
David Harrison
June 2019
"""

import pandas as pd
import os
import numpy as np

def new_csv(cols, rows):
	"""Create a new pandas object and insert the provided data
	
	:param cols: List of column names
	:param rows: Array of shape (N, C) where N is the number of rows and C is the number of columns
	:return: A pandas object with the populated data
	"""
	
	csv = pd.DataFrame(rows, columns = cols)
	
	return csv
	 
	
def load_csv(name, directory):
	"""Load an existing csv from disk
	
	:param name: The name of the file to load
	:param directory: Where the file is located
	:return: A pandas object with the loaded csv
	"""
	
	return pd.read_csv(directory + name, index_col = 0)
	
	
def add_rows(csv, rows):
	"""Add a row(s) of data to an existing csv without adding new columns
	
	:param csv: An existing pandas object with column names already defined
	:param rows: Array of shape (N, C) where N is the number of rows and C is the number of columns
	:return: The pandas object with the data added to the end of the file
	""" 
	
	for row in rows:
		csv.loc[0 if pd.isnull(csv.index.max()) else df.index.max() + 1] = row
	
	return csv
	
	
def add_cols(csv, cols, data):
	"""Add a new column(s) to the csv and then populate the existing rows
	
	:param csv: An existing pandas object
	:param cols: The new column name(s) to add
	:param data: A dictionary structured as {col_name: [data]} where the order of the data corresonds to the order of the rows
	:return: The pandas object with the new row added
	"""
	
	if type(cols) != list: cols = [cols]
	
	for i in range(len(cols)):
		csv.loc[:, cols[i]] = pd.Series(data[cols[i]], index = csv.index)	
	
	return csv
	

def save_csv(csv, name, directory):
	"""Save a pandas object to disk as a csv file
	
	:param csv: An existing pandas object
	:param name: What to save the file as
	:param directory: Where to save the file
	"""
	
	csv.to_csv(directory + name)
	
