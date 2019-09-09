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


def measurePerformance(model, test, target, calib = False, base_model = None, metric = None, classifier = False, proba = False):
    """Compute the mean squared error of a machine learning model
    
    :param model: The pre-trained model to verify
    :param test: A Pandas dataframe with the data to use to compute the performance
    :param target: Name of variable to try to predict
    :param calibrate: Set to True to train an Isotonic Regression using k-fold cross validation
    :param base_model: If calibrate == True, this is the pre-calibrated machine learning model
    :param metric: A callable method that takes the true values and predicted values as inputs and returns a single value
    :param classification: Set to True if measuring the performance of a classifier
    :param proba: If True, predict the positive class probability
    :return: The score of the model
    """
    
    if metric == None:  metric = mean_absolute_error

    if calib and not classifier:
        base_pred = base_model.predict(test)
        predict = model.predict(base_pred)
    elif proba:
        predict = model.predict_proba(test)[:,1]
    else:
        predict = model.predict(test)
    actual = target.values 
    return metric(actual, predict)
	
	
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
	
	
def CSI(true, pred, method = 'csi', scalar = True):
	"""Calculate the POD and FAR for a series of forecasts
	
	:param true: List of true binary values
	:param pred: List of predicted values
	:param method: Which performance metrics to return.
	               "perform" returns the PODs and SRs 
	               "roc" returns the TPR and FPR
	               "csi" returns the CSI
	:param scalar: Set True to return the max CSI
	:return: lists containing the specified metrics for each probability threshold
	"""
	
	pods = []
	srs = []
	csis = []
	tpr = [1]
	fpr = [1]
	probs = range(10, 101, 10)
	
	for prob in probs:
		tp = fp = tn = fn = 0
		this_prob = prob / 100.
		
		for i in range(len(pred)):
			if pred[i] >= this_prob and true[i] > 0: tp += 1
			elif pred[i] >= this_prob and not true[i] > 0: fp += 1
			elif pred[i] < this_prob and true[i] > 0 : fn += 1
			elif pred[i] < this_prob and not true[i] > 0: tn += 1
			
		if tp + fn == 0: pod = 0
		else: pod = tp / float(tp + fn)
		if tp + fp == 0: sr = 1
		else: sr = 1 - (fp / float(tp + fp))
		if fp + tn == 0: fpr0 = 0
		else: fpr0 = fp / float(fp + tn)
		if tp + fn + fp == 0: csi = 0
		else: csi = tp / float(tp + fn + fp)
		
		if pod == 0 and sr == 1: continue
		
		pods.append(pod)
		srs.append(sr)
		tpr.append(pod)
		fpr.append(fpr0)
		csis.append(csi)
	
	tpr.append(0)
	fpr.append(0)
	
	if scalar and method != 'csi': raise ValueError("Method must be CSI to return a scalar value")
	elif scalar: return max(csis)
		
	if method == 'perform': return pods, srs
	elif method == 'roc': return tpr, fpr
	elif method == 'csi': return csis
		
		
def reliability(pred, true):
	"""Calculate the reliability from predicted and true values
	
	:param pred: List of predicted values (flattened)
	:param true: List of true binary or probability values (flattened)
	:param how: Set to 'binary' to calculate based on lightning strikes
	:return: The reliability for each 10% threshold
	"""
	
	data = {0: {'hits':0, 'total':0},
			10:{'hits':0, 'total':0},
			20:{'hits':0, 'total':0},
			30:{'hits':0, 'total':0},
			40:{'hits':0, 'total':0},
			50:{'hits':0, 'total':0},
			60:{'hits':0, 'total':0},
			70:{'hits':0, 'total':0},
			80:{'hits':0, 'total':0},
			90:{'hits':0, 'total':0},
			100:{'hits':0, 'total':0}}
			
	for index in range(len(pred)):
				
		thispred = pred[index]
		if thispred < .10: thispred = 0
		elif thispred >= .10 and thispred < .20: thispred = 10
		elif thispred >= .20 and thispred < .30: thispred = 20
		elif thispred >= .30 and thispred < .40: thispred = 30
		elif thispred >= .40 and thispred < .50: thispred = 40
		elif thispred >= .50 and thispred < .60: thispred = 50
		elif thispred >= .60 and thispred < .70: thispred = 60
		elif thispred >= .70 and thispred < .80: thispred = 70
		elif thispred >= .80 and thispred < .90: thispred = 80
		elif thispred >= .90: thispred = 90
		
		if true[index]: data[thispred]['hits'] += 1
		data[thispred]['total'] += 1
		
	rel = []
	for value in range(0, 101, 10):
	
		try:
			rel.append(100 * (data[value]['hits'] / float(data[value]['total'])))
		except ZeroDivisionError:
			continue
				
	return rel

