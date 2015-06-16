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
from sklearn.utils import safe_asarray
import models

# TODO: Figure out if sklearn f1-score was used...
#       Did we have to do reimplement F1?
# TODO: explain in file comment why we re-implemented 
#        grid search and cross-validation
# TODO: decide whether dead code (next 2 functions) is in, out, or moved

def dvByLabel(y, dv):
  dv = safe_asarray(dv)
  result = np.zeros((y.shape[1],2))
  for k in range(y.shape[1]):
    result[k, 0] = np.median(dv[ y[:, k] == 0, k ])
    result[k, 1] = np.median(dv[ y[:, k] == 1, k ])
  result = pd.DataFrame(result)
  result['tot'] = y.sum(0)
  return result


def getScoreFrame(y, pred):
  fscores = np.zeros(y.shape[1])
  tot = y.sum(0)
  tp = (y * pred).sum(0)
  fp = (y < pred).sum(0)
  fn = (y > pred).sum(0)
  for k in range(y.shape[1]):
    fscores[k] = f1_score(y[:, k], pred[:, k], average='micro')
  out = pd.DataFrame({'tot':tot})
  out['tp'] = tp 
  out['fp'] = fp
  out['fn'] = fn
  out['f1'] = fscores
  return out


def grid(model, x, y, cvals, t1vals, t2vals, k=3):
  """
  Performs grid search for hyperparameters used in the uniformOVA model.
  
  Params:
    model - 
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
  folds = KFold(y.shape[0], k)
  pred = 0*y
  dvs = 0*y
  for train, val in folds:
    model.fit(x[train], y[train])
    dvs[val] = model.decision_function(x[val])
    pred[val] = model.predict(x[val])
  return (pred, dvs)


def cvk(model, x, y, k=3):
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
  """
  pred = model.predict(x)
  if type(y) is list:
    return meanF1list(y, pred)
  else:
    return meanF1array(y, pred)


