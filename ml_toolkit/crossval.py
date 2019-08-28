"""
David Harrison
August 2019

This utility script contains various functions related to k-fold cross-validation.
"""

from __future__ import print_function
from waf_code.learning import train as t
from waf_code.learning import metrics
from waf_code.learning import processing
from sklearn.base import clone 
from copy import deepcopy
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd


def generateFolds(data, folds, holdout = 1):
	"""Breaks up training data in a specified number of folds
	
	:param data: Pandas dataframe with the training / validation data
	:param folds: Number of desired folds
	:param holdout: Number of rows to skip between folds to preserve data independance
	:return: A list containing one pandas dataframe for each fold
	"""
	
	train_folds = np.array_split(np.array(range(data.shape[0])), folds)
	train_folds = [array[:-holdout].tolist() for array in train_folds]
	train_folds = [data.loc[fold] for fold in train_folds]
	return train_folds


def trainCrossval(folds, target, columns, calibrate = False, base_model = None, train_pcs = False, p_model = None):
    """Perform train or calibrate a regression model using k-fold cross validation 
    :param folds: List of training and validation data already broken into the desired number of folds
    :param target: Name of the target variable to try to predict
    :param columns: List of variables to train on
    :param calibrate: Set to True to train an Isotonic Regression using k-fold cross validation
    :param base_model: An instantiated machine learning model object to train.
                    If calibrate == True, this is the pre-trained machine learning model to calibrate
    :param train_pcs: Set to True to train the model using best PCA model, return best base model and pca model
    :param p_model: Best PCA model found from pre-trained machine learning model
    :return: The trained model or calibration, and number of principal components, with the best validation set RMSE
    """
    
    scores = []
    models = []
    p_components = []
    predicts = []
    trues = []
    
    # k-fold cross-validation   
    for f, fold in enumerate(folds):
        print('Fold ' + str(f+1))
        
        # Create train and validation sets
        train_folds = [folds[i] for i in range(len(folds)) if i != f]
        train = pd.concat(train_folds, ignore_index = True)
        val = fold
        
        train_target = train[target]
        val_target = val[target]
        
        # Apply PCA to the training and validation datasets if desired
        if train_pcs:
            if p_model is None:
                pca_model, train = processing.transformPCA(train[columns])
                val = processing.transformPCA(val[columns], trained_model = pca_model)
                p_components.append(pca_model)
            else:
                train = processing.transformPCA(train[columns], trained_model = p_model)
                val = processing.transformPCA(val[columns], trained_model = p_model)
        
        # Otherwise train on all data    
        else: 
            train = train[columns]
            val = val[columns]
        
        if calibrate: model = t.biasCorrection(base_model, train, train_target)
        else:
            base_model = deepcopy(base_model)
            model = t.trainRegressionModel(base_model, train, train_target)
        models.append(model)
        
        # Measure peformance
        if calibrate: scores.append(metrics.measurePerformance(model, val, val_target, True, base_model))
        else: scores.append(metrics.measurePerformance(model, val, val_target))
     
        # Compute bootstrap confidence intervals
        if calibrate: 
            base_predict = base_model.predict(val)
            predicts.append(model.predict(base_predict))
        
        else: predicts.append(model.predict(val))
        trues.append(val_target.values)
    
    # Get best validation score
    predicts = [item for fold in predicts for item in fold]
    trues = [item for fold in trues for item in fold]
    val_interval = metrics.getConfidenceInterval(predicts, trues, 1000)
    	
    if calibrate: print('\nCalibrated model mean MAE & 95% CI: ' + str(np.mean(scores)) + ' (' + str(val_interval[0]) + ' - ' + str(val_interval[1]) + ')')
    else: print('\nModel mean MAE & 95% CI: ' + str(np.mean(scores)) + ' (' + str(val_interval[0]) + ' - ' + str(val_interval[1]) + ')')
    
    print('Best model Validation MAE: ' + str(min(scores)))
    
    # Save the best model
    best_model = models[np.argmin(scores)]
    if train_pcs: 
        if p_model is None:
            best_pca = p_components[np.argmin(scores)]
            return best_model, best_pca
        else:
            return best_model, p_model
    else: 
        return best_model, None	
