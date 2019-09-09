"""
David Harrison
August 2019

This utility script contains various functions related to machine learning training and calibration.
"""

from __future__ import print_function
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from ml_toolkit import processing


def trainRegressionModel(model, train, target,  warm = False):
    """Train a regression model on a set of data
    
    :param model: An instantiated machine learning model to train
    :param train: A Pandas dataframe with the training data
    :param target: A Pandas dataframe with the training set target variable
    :param warm: Set to true to continue training a pre-trained model (not supported by all models)
    :return: The trained model
    """
    if warm == True: model.set_params(warm_start = True)
    model.fit(train, target)
    return model
	
	
def tuneParameters(model, param_dist, train, target, columns, n_iter = 10, train_pcs = True, metric = None):
	"""Use a randomized search to find the best hyperparameters of a model

	:param model: An instantiated machine learning model to tune
	:param param_dist: Dictionary of hyperparams and values to tune
		            Should be structured as {'parameter': [list of values to test]}
	:param train: A Pandas dataframe with the training data
	:param target: Name of variable to try to predict
	:param columns: List of variables to use as inputs
	:param n_iter: Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.
	:param train_pcs: Set to True to train the model on PCs
	:param metric: A callable method that takes the true values and predicted values as inputs and returns a single value
	"""
	
	def report(results, n_top=3):
		"""Nested utility function to report the best scores"""

		for i in range(1, n_top + 1):
		    candidates = np.flatnonzero(results['rank_test_score'] == i)
		    for candidate in candidates:
		        print("Model with rank: {0}".format(i))
		        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
		            results['mean_test_score'][candidate],
		            results['std_test_score'][candidate]))
		        print("Parameters: {0}".format(results['params'][candidate]))
		        print("")
	
	if metric == None: metric = 'neg_mean_absolute_error'
	
	# Apply PCA to the training data 
	if train_pcs:
		_, train_data  = processing.transformPCA(train[columns])
		print('Number of Principal Components explaining 99.7% of variance:', np.shape(train_data)[1]) 
		print()
	else: train_data = train[columns]

    # Use a cross-validated randomized search to find the best hyperparameters
	random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, scoring = metric, n_jobs = -1, cv=5)
	random_search.fit(train_data, train[target])
	report(random_search.cv_results_, n_top = 5)
	return
	
def biasCorrection(model, tune, target):
	"""Computes an isotonic regression to calibrated a pre-trained model
	
	:param model: The pre-trained model to calibrate
	:param tune: A Pandas dataframe with the data to use for calibration
	:param target: List of true values
	:return: The calibrated model
	"""
	inputValues = model.predict(tune)
	corrected_model = IsotonicRegression(out_of_bounds = 'clip')
	corrected_model.fit(inputValues, target.values)
		
	return corrected_model
	
	
def classifierCorrection(model, tune, target):
    """Calibrates the class probabilities of a classifier
    
    :param model: The pre-trained classifer to calibrate
    :param tune: A Pandas dataframe with the data to use for calibration
    :param target: List of true values
    :return: The calibrated model
    """
    
    calibrated_model = CalibratedClassifierCV(base_estimator = model, method = 'isotonic', cv = 'prefit')
    calibrated_model.fit(tune, target.values)
    
    return calibrated_model
	
	
def saveModel(model, name, outdir):
	"""Save a trained model to disk
	
	:param model: The model to save
	:param name: What to call the file
	:param outdir: Where to save the file
	"""
	
	with open(outdir + name, 'wb') as f:
		pickle.dump(model, f)
