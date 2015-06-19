"""
The functions in this module transform and read the input files, 
and write out the submission file.

project: Kaggle WISE 2014 Greek Media competition
author: David Thaler
"""

import numpy as np
import pandas as pd
import gzip
import cPickle
import re
from sklearn.datasets import load_svmlight_file
import pdb

# abs. path to project top-level directory
BASE = '/Users/davidthaler/Documents/Kaggle/GreekMedia/'
DATA = BASE + 'data/'
SUBMIT = BASE + 'submissions/'
SAMPLE =  DATA + 'sampleSubmission.csv'
SUBMISSION_PATH = SUBMIT + 'submission%d.csv.gz'
XTEST = DATA + 'xtest.pkl.gz'
XTRAIN = DATA + 'xtrain.pkl.gz'
YTRAIN = DATA + 'ytrain.pkl.gz'
NROW_TRAIN = 64857
NLABELS = 203
NFEATURES= 301561


def rewrite_train():
  """
  Rewrites train.libsvm into a gzipped, pickled sparse matrix for the features,
  and a gzipped, pickled numpy (0-1) array for the labels.
  Run this once.
  
  Params: none
  
  Returns: 
    nothing, but writes out the transformed input files at data/
  """
  inpath = DATA + 'train.libsvm'
  (x, ylist) = load_svmlight_file(inpath, 
                                  n_features=NFEATURES, 
                                  multilabel=True, 
                                  zero_based=False)
  with gzip.open(XTRAIN, 'wb') as fx:
    cPickle.dump(x, fx)
  y = list2matrix(ylist)
  with gzip.open(YTRAIN, 'wb') as fy:
    cPickle.dump(y, fy)


def list2matrix(ylist):
  """
  Rewrites a list-of-lists of labels for the multilabel case into a
  0-1 label matrix. The matrix is a numpy array (dense data type),
  but fairly sparse in practice.
  
  Params:
    ylist - a list of lists of integer labels for multilabel classification
  
  Returns:
    a 0-1 numpy array of size (# instances) x (# classes)
  """
  y = np.zeros((NROW_TRAIN, NLABELS))
  for k in range(len(ylist)):
    yl = ylist[k]
    for l in yl:
      y[k, l-1] = 1  
  return y

      
def rewrite_test():
  """
  Rewrites the test set features from test.libsvm, into a gzipped, 
  pickled sparse matrix.
  
  Params: none
  
  Returns: 
    nothing, but writes out the transformed input files at data/
  """
  inpath = DATA + 'test.libsvm'
  (x, y) = load_svmlight_file(inpath, n_features=NFEATURES, zero_based=False)
  with gzip.open(XTEST, 'wb') as f:
    cPickle.dump(x, f)
    
    
def loadTrain():
  """
  Function loads (uncompresses, unpickles) the training data and labels.
  
  Params: none
  
  Returns: 
    2-tuple of training set features and labels
  """
  with gzip.open(XTRAIN) as fx:
    x = cPickle.load(fx)
  with gzip.open(YTRAIN) as fy:
    y = cPickle.load(fy)
  return (x, y)
  
  
def loadTest():
  """
  Function loads (uncompresses, unpickles) the test data.
  
  Params: none
  
  Returns: 
    the test features
  """
  with gzip.open(XTEST) as f:
    x = cPickle.load(f)
  return x


def writeSubmission(submit_num, pred):
  """
  Writes out the predictions in the correct form for submission to Kaggle.
  
  Params:
    submit_num - the submission is named submission<submit_num>.csv.gz
    pred - a 0-1 numpy array of predictions of dimension 
            (# test instances) x (# classes)

  Returns:
    nothing, but writes submission file into submissions/
  """
  ss = pd.read_csv(SAMPLE)
  for k in range(pred.shape[0]):
    s = np.array_str(pred[k].nonzero()[0] + 1)
    s = s[1:-1].strip()
    s = re.sub(r"\W+", " ", s)
    ss.Labels[k] = s
  path = SUBMISSION_PATH % submit_num
  with gzip.open(path, 'wb') as f:
    ss.to_csv(f, index=False)






