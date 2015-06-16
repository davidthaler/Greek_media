"""
The functions in this module compute the mean F1-score evaluation metric,
perform cross-validation, grid search and make predictions from inside a
cross-validation loop for model stacking.

project: Kaggle WISE 2014 Greek Media competition
author: David Thaler
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score as cvs
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.utils import as_float_array
import models

# TODO: explain in file comment why we re-implemented 
#        grid search and cross-validation

def dvByLabel(y, dv):
  """
  Find the median decision values for the positive and negative instances
  of each class.
  This function is used for exploratory data analysis/model evaluation.
  
  Params:
    y - the 0-1 label matrix; a numpy array
    dv - array of decision values returned from a model; a numpy array

  Returns:
    a pandas data frame of size (# classes) x 3 with the median
    decision value on negative instances for class k in position [k, 0],
    the median positive decision value in [k, 1], and the count 
    of class k positives in [k, 2]
  """
  dv = as_float_array(dv)
  result = np.zeros((y.shape[1],2))
  for k in range(y.shape[1]):
    result[k, 0] = np.median(dv[ y[:, k] == 0, k ])
    result[k, 1] = np.median(dv[ y[:, k] == 1, k ])
  result = pd.DataFrame(result)
  result['tot'] = y.sum(0)
  return result


def getScoreFrame(y, pred):
  """
  Find the F1-score and its components, as well as the count of total
  positives, for each class.
  This function is used for exploratory data analysis/model evaluation.
  
  Params:
    y - training labels; a 0-1 numpy array
    pred - predictions; a 0-1 numpy array of the same size as y
  
  Returns:
    A Pandas data frame of size (# of classes) x 5
  """
  fscores = np.zeros(y.shape[1])
  tot = y.sum(0)
  tp = (y * pred).sum(0)
  fp = (y < pred).sum(0)
  fn = (y > pred).sum(0)
  for k in range(y.shape[1]):
    fscores[k] = f1_score(y[:, k], pred[:, k], average='binary')
  out = pd.DataFrame({'tot':tot})
  out['tp'] = tp 
  out['fp'] = fp
  out['fn'] = fn
  out['f1'] = fscores
  return out


def grid(model, x, y, cvals, t1vals, t2vals, k=3):
  """
  Performs grid search for hyperparameters used in the uniformOVA model.
  
  NB: This was re-implemented (vs. using grid search from sklearn) because
  sklearn grid search would try every (C, t1, t2) tuple, which would retrain
  the SVC with the same C value t1 x t2 times. This trains the SVC once, 
  and then searches over the (t1, t2) 2-tuples.
  
  Params:
    model - a model that has parameter C, t1 and t2
    x - training features
    y - training labels
    cvals - list or array of C parameter values for linear SVC
    t1vals - list or array of t1 (global threshold) values
    t2vals - list or array of t2 (proximity to max dv threshold) values
    k - # of cross-validation folds
    
  Returns:
    an array with every tuple of C, t1 and t2, along with its mean F1-score
  """
  result = []
  for c in cvals:
    model.c = c
    (pred, dv) = predictCV(model, x, y, k)
    for t1val in t1vals:
      for t2val in t2vals:
        pred = models.repredict(dv, t1val, t2val)
        score = f1_score(y, pred, average='samples')
        result.append((c, t1val, t2val, score))
  return result


def predictCV(model, x, y, k=3):
  """
  Makes predictions over the training set using a cross-validation loop
  such that the predictions for each instance are from the model trained 
  when that instance was held out.
  
  Params:
    model - The model used. Model state will be changed by training.
    x - training features
    y - training labels
    k - # of CV folds
    
  Returns: a 2-tuple
    pred - a numpy array of 0-1 predictions, similar to what gets submitted.
    dvs - a numpy array of the decision values of the SVC
  """
  folds = KFold(y.shape[0], k)
  pred = 0*y
  dvs = 0*y
  for train, val in folds:
    model.fit(x[train], y[train])
    dvs[val] = model.decision_function(x[val])
    pred[val] = model.predict(x[val])
  return (pred, dvs)


def cvk(model, x, y, k=3):
  """
  Computes mean F1-score for a provided model and training data.
  
  Params:
    model - the model to use in cross-validation
    x - training features
    y - training labels
    k - number of cross-validation folds
    
  Returns:
    a numpy array of the mean F1-scores for each fold
  """
  return cvs(model, x, y, scoring=meanF1scorer, cv=k, n_jobs=3)


def meanF1array(gold, pred):
  row_f1 = np.zeros((gold.shape[0]))
  for k in range(gold.shape[0]):
    tp = (pred[k] * gold[k]).sum()
    fp = (pred[k] > gold[k]).sum()
    fn = (pred[k] < gold[k]).sum()
    precision = tp/(tp + fp + 1e-9)
    recall = tp/(tp + fn + 1e-9)
    row_f1[k] = 2 * precision * recall / (precision + recall + 1e-9)
  return np.mean(row_f1)


def meanF1list(gold, pred):
  row_f1 = np.zeros(len(gold))
  for k in range(len(gold)):
    tp = len(np.intersect1d(gold[k], pred[k]))
    fp = len(np.setdiff1d(pred[k], gold[k]))
    fn = len(np.setdiff1d(gold[k], pred[k])) 
    precision = tp/(tp + fp + 1e-9)
    recall = tp/(tp + fn + 1e-9)
    row_f1[k] = 2 * precision * recall / (precision + recall + 1e-9)
  return np.mean(row_f1)
  

def meanF1scorer(model, x, y):
  """
  A scoring function for mean F1-score with the right signature 
  to use as the 'scoring' parameter in sklearn.metric.cross_val_score.
  
  Params:
    model - the model to use for prediction
    x - features to use for prediction
    y - ground truth labels for the examples in x
        a numpy 0-1 array of size (# instances) x (# classes)
    
  Returns:
    the mean F1-score for the predictions on x
  """
  pred = model.predict(x)
  if type(y) is list:
    return meanF1list(y, pred)
  else:
    return meanF1array(y, pred)







