"""
This module contains all of the models used in the Greek Media topic
classification competition on Kaggle.

project: Kaggle WISE 2014 Greek Media competition
author: David Thaler
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import pdb


class RidgePCA(BaseEstimator):
  '''
  RidgePCA fits a ridge regression to 'topics' exacted 
  from the label matrix, y, using PCA.
  '''
  
  def __init__(self, c=1, n_components=100, t1=0, t2=0):
    '''
    Constructor only sets parameter values.
    
    Params:
      c - L2 regularization parameter for ridge regression.
          Usually called lambda, calling it 'c' lets us use eval.grid.
      n_components - the # of components to use in the representation 
          of the multilabel y matrix. This is like an LSA topic count.
      t1 - either a scalar threshold, or a vector of length 
         (# of classes); all predictions > t1 are positive
      t2 - a scalar threshold for closeness to the maximum row value
         Predictions >= row_max - t2 are positive.
         
    Returns: 
      an initialized RidgePCA model
    '''
    self.c = c
    self.n_components = n_components
    self.t1 = t1
    self.t2 = t2
    
  def fit(self, x, y):
    '''
    Fit RidgePCA model.
    
    Params:
      x - input features
      y - 0-1 label matrix
      
    Returns:
      nothing, but model is fitted.
    '''
    self.pca = PCA(n_components=self.n_components)
    self.lm = Ridge(alpha=self.c)
    z = self.pca.fit_transform(y)
    self.lm.fit(x, z)
    
  def decision_function(self, x):
    '''
    Computes a matrix of real-valued scores reflecting the strength 
    of belief that an instance is in a particular class.
    
    Params:
      x - input features
    
    Returns:
      a real-valued matrix of dimension (# instances) x (# classes)
    '''
    zdv = self.lm.decision_function(x)
    return self.pca.inverse_transform(zdv)

  def predict(self, x):
    '''
    Computes a 0-1 matrix of predicted labels for each instance.
    
    Params:
      x - input features
    
    Returns:
      A 0-1 matrix of predicted labels.
    '''
    dv = self.decision_function(x)
    return repredict(dv, self.t1, self.t2)
    

class UniformOVA(BaseEstimator):
  '''
  UniformOVA estimator fits a linear SVC model for each class that has
  data, and a NullModel for any classes with no positive instances.
  It predicts class membership whenever the decision value is either above
  one threshold or within a second threshold of the highest value 
  for that instance.
  '''

  # NB: best known values are c=1 (default), t1=-0.3, t2=0.1
  def __init__(self, c=1, t1=0, t2=0, null_dv=-99):
    '''
    Constructor for UniformOVA model. Just stores field values.
    
    Params:
      t1 - either a scalar threshold, or a vector of length(dv.shape[1])
         all instances with dvs > t1 are positive
      t2 - all instances with dvs >= row_max - t2 are positive
      c - L2 loss parameter for the SVC's
      null_dv - the decision value for classes with no positive instances.
      
    Returns:
      an initialized UniformOVA model
    '''
    self.t1 = t1
    self.t2 = t2
    self.c = c
    self.null_dv = null_dv
    
  def fit(self, x, y):
    '''
    Fit the UniformOVA model.
    
    Params:
      x - input features
      y - 0-1 label matrix
  
    Returns:
      nothing, but model is fitted.
    '''
    self.models = []
    for k in range(y.shape[1]):
      if (y[:, k]).any():
        model = LinearSVC(C = self.c)
        model.fit(x, y[:, k])
      else:
        model = NullModel(self.null_dv)
      self.models.append(model)
    
  def predict(self, x):
    '''
    Prediction method predicts class membership of instances with decision 
    values above threshold t1 or within t2 of the highest decision value 
    on that instance.
    
    Params:
      x - input features, not used
      y - 0-1 label matrix, not used
  
    Returns:
      A 0-1 matrix of predicted labels of size (# instances) x (# classes).
    '''
    dvs = self.decision_function(x)
    pred = (dvs > self.t1).astype(float)
    max_dv = dvs.max(1)
    for k in range(pred.shape[0]):
      cut = max_dv[k] - self.t2
      idx = (dvs[k, :] >= cut) 
      pred[k, idx] = 1
    return pred
    
  def decision_function(self, x):
    '''
    Finds the decision value for each instance under each per-class model.
    
    Params:
      x - input features, not used
    
    Returns:
      a real-valued matrix of dimension (# instances) x (# classes)
    '''
    dvs = np.zeros((x.shape[0], len(self.models)))
    for k in range(len(self.models)):
      dvs[:, k] = self.models[k].decision_function(x)
    return dvs
    
    
class NullModel(BaseEstimator):
  '''
  NullModel returns a decision value that results in a negative prediction.
  It is used for the 3 classes that do not appear in the training data.
  This model allows us to just keep a list of models for all of the classes.
  Normal models can't be fitted on classes with only one label. Unlike the 
  other models, NullModel is for only one class.
  '''
  
  def __init__  (self, null_dv=-99):
    '''
    Constructor stores the constant decision value to use.
    
    Params:
      null_dv - the decision value to return
      
    Returns: a NullModel
    '''
    self.null_dv = null_dv
    
  def fit(self, x, y):
    '''
    Fit is a no-op for the NullModel
    
    Params:
      x - input features, not used
      y - 0-1 label vector
    
    Returns: nothing
    '''
    pass
  
  def predict(self, x):
    '''
    For NullModel, predict() always returns 0 (non-membership).
    
    Params:
      x - input features, not used
      
    Returns:
      0, always
    '''
    return 0
    
  def decision_function(self, x):
    '''
    Returns the null_dv for all instances.
    
    Params:
      x - input features, not used
      
    Returns:
     the null_dv, always
    '''
    return self.null_dv * np.ones(x.shape[0])


class ThresholdOVA(BaseEstimator):
  '''
  ThresholdOVA uses a UniformOVA model, but chooses a per-class threshold
  on the decision values for predicting class membership. These per-class
  thresholds are found on a set of cross-validation predictions. The
  per-class threshold adjustment is only sought for the top k classes.
  '''
  
  def __init__(self, c=1, t1=0, t2=0, tstep=0.1, k=10):
    '''
    Constructor for ThresholdOVA model. Just stores field values.
    
    Params:
      c - L2 loss parameter for the SVC's
      t1 - either a scalar threshold, or a vector of length(dv.shape[1])
         all instances with dvs > t1 are positive
      t2 - all instances with dvs >= row_max - t2 are positive
      tstep - the thresholds tried are t1 +- tstep and 2*tstep
      k - thresholds are adjusted for the most-frequent k classes 
      
    Returns:
      an initialized ThresholdOVA model
    '''
    self.c = c
    self.t1 = t1
    self.t2 = t2
    self.tstep = tstep
    self.k = k
    
  def fit(self, x, y):
    '''
    Fit the ThresholdOVA model.
    
    Params:
      x - input features
      y - 0-1 label matrix
  
    Returns:
      nothing, but model is fitted.
    '''
    self.model = UniformOVA(c=self.c)
    dv = cvdv(self.model, x, y)
    self.dv = dv
    self.fit_thr(y)
    self.model.fit(x,y)
  
  def fit_thr(self, y):
    '''
    Fits per-class thresholds for the top self.k classes.
    
    Params:
      y - 0-1 label matrix
  
    Returns:
      nothing, but thresholds for the top self.k classes are adjusted.
    '''
    ysum = np.array(y.sum(0))
    self.thr = self.t1 * np.ones(len(ysum))
    idx = np.argsort(ysum)
    idx = idx[::-1]
    idx = idx[:self.k]
    steps = self.tstep * np.array([-2., -1., 0., 1., 2.])
    for i in idx:
      f1s = np.zeros(len(steps))
      for (k, step) in enumerate(steps):
        thr = self.thr.copy()
        thr[i] += step
        pred = repredict(self.dv, thr, self.t2)
        f1s[k] = f1_score(y, pred, average='samples')
      best_idx = np.argmax(f1s)
      self.thr[i] += steps[best_idx]

  def decision_function(self, x):
    '''
    Finds the decision value for each instance under each per-class model.
    
    Params:
      x - input features, not used
    
    Returns:
      a real-valued matrix of dimension (# instances) x (# classes)
    '''
    return self.model.decision_function(x)
    
  def predict(self, x):
    '''
    Computes a 0-1 matrix of predicted labels for each instance.
    
    Params:
      x - input features
    
    Returns:
      A 0-1 matrix of predicted labels.
    '''
    dv = self.decision_function(x)
    return repredict(dv, self.thr, self.t2)
  
  
class StackModel(BaseEstimator):
  '''
  StackModel trains a gradient boosting classifier (GBC) on features made by
  training other models inside of cross-validation loops and predicting 
  on the held-out fold. The predictions are the aggregated into a feature
  matrix with the same number of rows as the data. The first level models are 
  the UniformOVA model, the RidgePCA model, and a regular ridge regression 
  trained on the count of positive labels per row.
  '''
  
  def __init__(self, 
               c=1, 
               t1=2, 
               t2=0.75, 
               n_estimators=100, 
               max_depth=2,
               n_stack = 3000,
               folds=3):
    '''
    Constructor for StackModel. This only stores field values.
    
    Params:
      c - L2 loss parameter for the SVC's
      t1 - either a scalar threshold, or a vector of length(dv.shape[1])
         all instances with dvs > t1 are positive
      t2 - all instances with dvs >= row_max - t2 are positive
      n_estimators - # of trees used in GBC
      max_depth - max depth of trees in GBC
      n_stack - # examples used to train the StackModel. These are
           individual predictions for an (instance, class) 2-tuple, that
           is, for a single element in the label matrix, y
      folds - # of folds used in the CV-loops used to generate the features
           for the StackModel's GBC
          
    Returns: 
      an initialized StackModel
    '''
    self.c = c
    self.t1 = t1
    self.t2 = t2
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.n_stack = n_stack
    self.folds = folds
    
  def fit(self, x, y):
    '''
    Fit the StackModel.
    
    Params:
      x - input features
      y - 0-1 label matrix
  
    Returns:
      nothing, but model is fitted.
    '''
    self.model1 = UniformOVA(c=self.c)
    dv = cvdv(self.model1, x, y, self.folds)
    self.model0 = RidgePCA(n_components=125)
    dv0 = cvdv(self.model0, x, y, self.folds)
    idx = np.random.choice(dv.shape[0], self.n_stack, replace=False)
    nidx = np.setdiff1d(np.arange(y.shape[0]), idx)
    self.yrate = y[nidx].mean(0)
    
    self.posdv = np.zeros(dv.shape[1])
    self.negdv = np.zeros(dv.shape[1])
    self.posdv0 = np.zeros(dv.shape[1])
    #subset to data not used in stack model
    yout = y[nidx]
    dvout = dv[nidx]
    dvout0 = dv0[nidx]
    max_dv = dvout.max()
    max_dv0 = dvout0.max()
    #for each column, get median dv of the pos and neg examples
    for k in range(dv.shape[1]):
      self.posdv[k] = np.median(dvout[yout[:, k]==1, k])
      self.posdv0[k] = np.median(dvout0[yout[:, k]==1, k])
      if np.isnan(self.posdv[k]):
        self.posdv[k] = max_dv
        self.posdv0[k] = max_dv0
      self.negdv[k] = np.median(dvout[yout[:, k]==0, k])
    
    self.count_model = Ridge()
    self.count_model.fit(x[nidx], y[nidx].sum(1))
    self.model2 = GradientBoostingClassifier(n_estimators = self.n_estimators,
                                             max_depth = self.max_depth)
    f, yf = self.dv2ftr(dv[idx], dv0[idx], x[idx], y[idx])
    self.model2.fit(f, yf)
    self.model1.fit(x,y)
    self.model0.fit(x,y)
    
  def predict(self, x):
    '''
    Computes a 0-1 matrix of predicted labels for each instance.
    
    Params:
      x - input features
    
    Returns:
      A 0-1 matrix of predicted labels.
    '''
    dvs = self.decision_function(x)
    pred = (dvs > self.t1).astype(float)
    max_dv = dvs.max(1)
    for k in range(pred.shape[0]):
      cut = max_dv[k] - self.t2
      idx = (dvs[k, :] >= cut) 
      pred[k, idx] = 1
    return pred
    
  def decision_function(self, x):
    '''
    Finds the decision value for each instance under each per-class model.
    
    Params:
      x - input features, not used
    
    Returns:
      a real-valued matrix of dimension (# instances) x (# classes)
    '''
    dv0 = self.model0.decision_function(x)
    dv1 = self.model1.decision_function(x)
    f = self.dv2f(dv1, dv0, x)
    dv2 = self.model2.decision_function(f)
    dv2 = dv2.reshape( (x.shape[0], len(self.yrate) ) )
    return dv2

  def dv2ftr(self, dv, dv0, x, y):
    '''
    Computes the features and labels for use in training the StackModel.
    StackModel is trained on a subset of the data.
    '''
    f = self.dv2f(dv, dv0, x)
    yf = y.ravel()
    pos_idx = np.where(yf==1)[0]
    neg_idx = np.where(yf==0)[0]
    neg_idx = np.random.choice(neg_idx, 10 * len(pos_idx), replace=False)
    idx = np.union1d(pos_idx, neg_idx)
    return f[idx], yf[idx]

  def dv2f(self, dv, dv0, x):
    '''
    Computes the features used in the StackModel. These are:
    the decision value (DV) and max DV by row under the UniformOVA
    model; the DV and max DV by row under RidgePCA model; the 
    median dv for positive and negative instances of each class;
    the estimated counts under the count model. 
    '''
    rowmax = dv.max(1)
    row_dv = (dv.transpose() - rowmax).transpose()
    rowmax0 = dv0.max(1)
    row_dv0 = (dv0.transpose() - rowmax0).transpose()
    rates = np.tile(self.yrate, (dv.shape[0], 1) )
    posdv = np.tile(self.posdv, (dv.shape[0], 1) )
    posdv0 = np.tile(self.posdv0, (dv.shape[0], 1) )
    negdv = np.tile(self.negdv, (dv.shape[0], 1) )
    cts = self.count_model.predict(x)
    cts = np.tile(cts, (dv.shape[1], 1)).transpose()
    f = np.column_stack( (dv.ravel(),
                          row_dv.ravel(),
                          dv0.ravel(),
                          row_dv0.ravel(),
                          rates.ravel(),
                          posdv.ravel(),
                          posdv0.ravel(),
                          negdv.ravel(),
                          cts.ravel()) )
    return f
  

def getRowDV(dv):
  '''
  Adjusts the decision values by subtracting off the row-wise maximum
  value from each entry. This tells us how close a prediction was 
  to being the strongest response for that instance.
  
  Params:
    dv - matrix of decision values
  
  Returns:
    a matrix containing the decision values, with the row-wise 
    maximum value subtracted from each value
  '''
  rowmax = dv.max(1)
  row_dv = (dv.transpose() - rowmax).transpose()
  return row_dv
  
  
def cvdv(model, x, y, k=3, use_predict=False):
  """
  Get dvs for all of x by training on k folds, predicting on 1,
  and aggregating the predictions into an object the same shape as y.
  
  Params:
    x - input features
    y - 0-1 label matrix
    k - # of cross-validation folds
    use_predict - If true, use predict() instead of decision_value()
        to get the return values. Default False. 
  
  Returns:
    a real matrix of predictions made within cross-validation
  """
  folds = KFold(y.shape[0], k)
  dv = 0*y
  for train, val in folds:
    model.fit(x[train], y[train])
    if use_predict:
      dv[val] = model.predict(x[val])
    else:
      dv[val] = model.decision_function(x[val])
  return dv


def repredict(dv, t1, t2):
  """
  Takes decision values and returns predictions, given thresholds 
  for the overall level and for distance to the row-wise maximum.
  
  Params:
    dv - 2d array of decision values
    t1 - either a scalar threshold, or a vector of length(dv.shape[1])
         all dvs > t1 are positive
    t2 - all dvs >= row_max - t2 are positive
    
  Returns:
    predictions (0-1) from these dvs with the given threshold.
  """
  pred = ((dv - t1) > 0).astype(float)
  max_dv = dv.max(1)
  for k in range(pred.shape[0]):
    cut = max_dv[k] - t2
    idx = (dv[k, :] >= cut) 
    pred[k, idx] = 1
  return pred
  


