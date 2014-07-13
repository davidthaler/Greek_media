import numpy as np
import pandas as pd
import gzip
import cPickle
import re
from sklearn.datasets import load_svmlight_file
import pdb


BASE = '../../'
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
  y = np.zeros((NROW_TRAIN, NLABELS))
  for k in range(len(ylist)):
    yl = ylist[k]
    for l in yl:
      y[k, l-1] = 1  
  return y


def matrix2list(y):
  ylist = []
  for k in range(y.shape[0]):
    ylist.append(y[k].nonzero()[0] + 1)
  return ylist

      
def rewrite_test():
  inpath = DATA + 'test.libsvm'
  (x, y) = load_svmlight_file(inpath, n_features=NFEATURES, zero_based=False)
  with gzip.open(XTEST, 'wb') as f:
    cPickle.dump(x, f)
    
    
def loadTrain():
  with gzip.open(XTRAIN) as fx:
    x = cPickle.load(fx)
  with gzip.open(YTRAIN) as fy:
    y = cPickle.load(fy)
  return (x, y)
  
  
def loadTest():
  with gzip.open(XTEST) as f:
    x = cPickle.load(f)
  return x


def writeSubmission(submit_num, pred):
  ss = pd.read_csv(SAMPLE)
  for k in range(pred.shape[0]):
    s = np.array_str(pred[k].nonzero()[0] + 1)
    s = s[1:-1].strip()
    s = re.sub(r"\W+", " ", s)
    ss.Labels[k] = s
  path = SUBMISSION_PATH % submit_num
  with gzip.open(path, 'wb') as f:
    ss.to_csv(f, index=False)






