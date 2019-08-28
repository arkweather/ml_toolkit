"""
David Harrison
August 2019

This utility script contains various functions used to compute the performance of a machine learning model.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def measurePerformance(model, test, target, calib = False, base_model = None):
    """Compute the mean squared error of a machine learning model
    
    :param model: The pre-trained model to verify
    :param test: A Pandas dataframe with the data to use to compute the performance
    :param target: Name of variable to try to predict
    :param calibrate: Set to True to train an Isotonic Regression using k-fold cross validation
    :param base_model: If calibrate == True, this is the pre-calibrated machine learning model
    :return: The MAE of the model
    """
    
    # TODO - Allow other metrics here
    if calib:
        base_pred = base_model.predict(test)
        predict = model.predict(base_pred)
    else:
        predict = model.predict(test)
    actual = target.values 
    return mean_absolute_error(actual, predict)
	
	
def getConfidenceInterval(predict, true, iterations = 1000, bootstrap = 5000):
	"""Computes the 95% confidence intervals of a distribution
	
	:param predict: List of predicted values
	:param true: List of true values
	:param iterations: Number of times to compute the interval (final interval is mean of all iterations)
	:param bootstrap: Sample size to use to compute the interval
	:return: (lower interval, upper interval)
	"""
	
	print('\nGetting confidence intervals...')
	
	lower = []
	upper = []
	
	indices = range(0, len(predict))
	for i in range(iterations):
		error = []
		choices = np.random.choice(indices, bootstrap, replace = True)
		for index in choices:
			error.append(abs(predict[index] - true[index]))
	
		lower.append(np.percentile(error, 2.5))
		upper.append(np.percentile(error, 97.5))
	
	return(np.mean(lower), np.mean(upper))

