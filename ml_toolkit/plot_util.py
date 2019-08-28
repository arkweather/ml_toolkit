"""
David Harrison
August 2019

This utility script contains various functions to plot the results of the machine learning
"""

import matplotlib.pyplot as plt
from PIL import Image
from ml_toolkit import processing
try:    
    import cStringIO
except:
    from io import StringIO

import numpy as np


def plotTrainingDeviance(model, val, target, columns, plot = False, save = True, name = '', outdir = ''):
	"""Plot the training and validation set deviance
	
	:param model: The trained model to plot the results for
	:param val: A Pandas object with the validation dataset
	:param target: The name of the variable to predict
	:param columns: List of variables to use as inputs for the model
	:param plot: Set to True to display the plot (plt.show())
	:param save: Set to True to save the plot
	:param name: What to call the save file
	:param outdir: Where to save the file
	"""
	
	test_score = np.zeros((100,), dtype=np.float64)

	for i, y_pred in enumerate(model.staged_predict(val[columns])):
		test_score[i] = model.loss_(val[target], y_pred)

	plt.figure(figsize=(12,10))
	plt.title('Validation Set Deviance', fontsize = 14)
	plt.plot(np.arange(100) + 1, model.train_score_, 'b-', linewidth = 2, label='Training Set Deviance')
	plt.plot(np.arange(100) + 1, test_score, 'r-', linewidth = 2, label='Validation Set Deviance')
	plt.legend(loc='upper right', fontsize = 12)
	plt.ylim(0)
	plt.xlim(0, 100)
	plt.grid()
	plt.xlabel('Boosting Iterations', fontsize = 12)
	plt.ylabel('Deviance', fontsize = 12)
	
	if plot: plt.show()
	elif save:
		# Save image through PIL to reduce file size
		ram = cStringIO.StringIO()
		plt.savefig(ram, format='png')
		ram.seek(0)
		im = Image.open(ram)
		im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
		im2.save(outdir + name, format = 'PNG')
		plt.close()
	

def plotFeatureImportance(model, columns, plot = False, save = True, name = '', outdir = ''):
	"""Plot the permutation feature importance of a model
	
	:param model: The trained machine learning model
	:param columns: List of variables that the model was trained on
	:param plot: Set to True to display the plot (plt.show())
	:param save: Set to True to save the plot
	:param name: What to call the save file
	:param outdir: Where to save the file
	"""	
	
	feature_importance = model.feature_importances_
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5

	plt.figure(figsize=(12,10))
	plt.barh(pos, feature_importance[sorted_idx], align='center')
	plt.yticks(pos, np.array(columns)[sorted_idx])
	plt.xlabel('Relative Importance (%)', fontsize = 12)
	plt.title('Variable Importance', fontsize = 14)
	
	if plot: plt.show()
	elif save:
		# Save image through PIL to reduce file size
		ram = cStringIO.StringIO()
		plt.savefig(ram, format='png')
		ram.seek(0)
		im = Image.open(ram)
		im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
		im2.save(outdir + name, format = 'PNG')
		plt.close()
		
def plotTestPredictions(base_model, calib_model, test, target, columns, 
                        title = '', ylabel = '', plot = False, save = True, 
                        name = '', outdir = '', test_pcs = False, p_model = None):
	
    """Plot the predictions of the base regression and calibrated regression compared to the true values
	
	:param base_model: The uncalibrated machine learning model
	:param calib_model: The calibrated machine learning model
	:param test: Pandas object containing the test set data
	:param target: Name of the target variable to predict
	:param columns: List of input variables
	:param title: What to title the plot
	:param ylabel: What to label the y axis
	:param plot: Set to True to display the plot (plt.show())
	:param save: Set to True to save the plot
	:param name: What to call the save file
	:param outdir: Where to save the file
    :param test_pcs: Will apply PCA to testing data if True
    :param p_model: Best PCA model found from pre-trained machine learning model
	"""
    # True values	
    actual = test[target].values
    dates = test['date']

    #Apply PCA if desired
    if test_pcs is True:
        test = processing.transformPCA(test[columns],trained_model=p_model)
    else:
        test = test[columns]

    # Uncalibrated model
    base_predict = base_model.predict(test)

    # Calibrated model
    tmp_pred = base_model.predict(test)
    calib_predict = calib_model.predict(tmp_pred)
    
    # Plot the figure
    plt.figure(figsize =(12,6))
    plt.plot(dates, actual, color = 'k', linewidth = 1, label = 'True Values')
    plt.plot(dates, base_predict, color = 'r', linewidth = 1, label = 'Base Predicted Values')
    plt.plot(dates, calib_predict, color = 'b', linewidth = 1, label = 'Calibrated Predicted Values')
    plt.ylabel(ylabel)
    plt.title(title, fontsize = 16)
    plt.legend()
    plt.grid()

    if plot: plt.show()
    elif save:
        # Save image through PIL to reduce file size
        ram = cStringIO.StringIO()
        plt.savefig(ram, format='png')
        ram.seek(0)
        im = Image.open(ram)
        im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
        im2.save(outdir + name, format = 'PNG')
        plt.close()
	
	
	
		
