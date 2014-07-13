import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import pdb


class LinearStackModel(BaseEstimator):

  def __init__(self, 
               c1=1, 
               c2=1, 
               l2=1, 
               t1=0, 
               t2=0, 
               ycut=300, 
               folds=3, 
               pairs_cutoff=300):
    self.c1 = c1
    self.c2 = c2
    self.l2 = l2
    self.t1 = t1
    self.t2 = t2
    self.ycut=ycut
    self.folds = folds
    self.pairs_cutoff = pairs_cutoff
    
  def fit(self, x, y):
    self.ycount = np.array(y.sum(0))
    self.model1 = UniformOVA(c=self.c1)
    pf = pairFeatures(y, self.pairs_cutoff)
    yf = np.column_stack((y, pf))
    dv = cvdv(self.model1, x, yf, self.folds)
    dvf = dv[:, y.shape[1]:]
    dv = dv[:, :y.shape[1]]
    self.count_model = Ridge(alpha=self.l2)
    ys = y.sum(1)
    cts = cvdv(self.count_model, x, ys, k=self.folds, use_predict=True)
    row_dv = getRowDV(dv)
    self.stack_models = []
    for k in range(y.shape[1]):
      if (y[:, k]).any():
        k_idx = (self.ycount >= self.ycut)
        k_idx[k] = True
        stack_model = LinearSVC(C=self.c2)
        f = np.column_stack((dv[:, k_idx], row_dv[:, k_idx], dvf, cts))
        stack_model.fit(f, y[:, k])
      else:
        stack_model = NullModel()
      self.stack_models.append(stack_model)
    self.model1.fit(x, yf)
    self.count_model.fit(x, ys)
    
  def predict(self, x):
    dvs = self.decision_function(x)
    self.dv = dvs
    pred = (dvs > self.t1).astype(float)
    max_dv = dvs.max(1)
    for k in range(pred.shape[0]):
      cut = max_dv[k] - self.t2
      idx = (dvs[k, :] >= cut) 
      pred[k, idx] = 1
    return pred
    
  def decision_function(self, x):
    n = len(self.stack_models)
    dv_out = np.zeros( (x.shape[0], n) )
    dv1 = self.model1.decision_function(x)
    dvf = dv1[:, n:]
    dv1 = dv1[:, :n]
    cts = self.count_model.predict(x)
    row_dv = getRowDV(dv1)
    for k in range(n):
      k_idx = (self.ycount >= self.ycut)
      k_idx[k] = True
      f = np.column_stack( (dv1[:, k_idx], row_dv[:, k_idx], dvf, cts) )
      dv_k = self.stack_models[k].decision_function(f)
      dv_k = np.squeeze(dv_k)
      dv_out[:, k] = dv_k
    return dv_out

  def repredict(self, t1, t2):
    return repredict(self.dv, t1, t2)
    
    
#Not instance methods
def getRowDV(dv):
  rowmax = dv.max(1)
  row_dv = (dv.transpose() - rowmax).transpose()
  return row_dv
  
def pairFeatures(y, cutoff):
  pairs = [(j,k) for j in range(y.shape[1]) for k in range(y.shape[1]) if j < k]
  f = [(y[:, j]*y[:, k]) for j,k in pairs if (y[:, j] * y[:, k]).sum() >= cutoff]
  return np.array(f).transpose()


class RidgePCA(BaseEstimator):

  def __init__(self, c=1, n_components=100, t1=0, t2=0):
    self.c = c                    # calling it c allows us to use eval.grid
    self.n_components = n_components
    self.t1 = t1
    self.t2 = t2
    
  def fit(self, x, y):
    self.pca = PCA(n_components=self.n_components)
    self.lm = Ridge(alpha=self.c)
    z = self.pca.fit_transform(y)
    self.lm.fit(x, z)
    
  def decision_function(self, x):
    zdv = self.lm.decision_function(x)
    return self.pca.inverse_transform(zdv)

  def predict(self, x):
    dv = self.decision_function(x)
    return repredict(dv, self.t1, self.t2)
    

class StackModel2(BaseEstimator):

  def __init__(self, 
               c=1, 
               t1=2, 
               t2=0.75, 
               n_estimators=100, 
               max_depth=2,
               n_stack = 3000,
               folds=3):
    self.c = c
    self.t1 = t1
    self.t2 = t2
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.n_stack = n_stack
    self.folds = folds
    
  def fit(self, x, y):
    self.model1 = UniformOVA(c=self.c)
    dv = cvdv(self.model1, x, y, self.folds)
    idx = np.random.choice(dv.shape[0], self.n_stack, replace=False)
    nidx = np.setdiff1d(np.arange(y.shape[0]), idx)
    self.yrate = y[nidx].mean(0)
    
    #addition
    self.posdv = np.zeros(dv.shape[1])
    self.negdv = np.zeros(dv.shape[1])
    #subset to data not used in stack model
    yout = y[nidx]
    dvout = dv[nidx]
    max_dv = dvout.max()
    #for each column, get median dv of the pos and neg examples
    for k in range(dv.shape[1]):
      self.posdv[k] = np.median(dvout[yout[:, k]==1, k])
      if np.isnan(self.posdv[k]):
        self.posdv[k] = max_dv
      self.negdv[k] = np.median(dvout[yout[:, k]==0, k])
    #end addition
    
    self.count_model = Ridge()
    self.count_model.fit(x[nidx], y[nidx].sum(1))
    self.model2 = GradientBoostingClassifier(n_estimators = self.n_estimators,
                                             max_depth = self.max_depth)
    f, yf = self.dv2ftr(dv[idx], x[idx], y[idx])
    self.model2.fit(f, yf)
    self.model1.fit(x,y)
    
  def predict(self, x):
    dvs = self.decision_function(x)
    pred = (dvs > self.t1).astype(float)
    max_dv = dvs.max(1)
    for k in range(pred.shape[0]):
      cut = max_dv[k] - self.t2
      idx = (dvs[k, :] >= cut) 
      pred[k, idx] = 1
    return pred
    
  def decision_function(self, x):
    dv1 = self.model1.decision_function(x)
    f = self.dv2f(dv1, x)
    dv2 = self.model2.decision_function(f)
    dv2 = dv2.reshape( (x.shape[0], len(self.yrate) ) )
    return dv2
    

  def dv2ftr(self, dv, x, y):
    f = self.dv2f(dv, x)
    yf = y.ravel()
    pos_idx = np.where(yf==1)[0]
    neg_idx = np.where(yf==0)[0]
    neg_idx = np.random.choice(neg_idx, 10 * len(pos_idx), replace=False)
    idx = np.union1d(pos_idx, neg_idx)
    return f[idx], yf[idx]

  def dv2f(self, dv, x):
    rowmax = dv.max(1)
    row_dv = (dv.transpose() - rowmax).transpose()
    rates = np.tile(self.yrate, (dv.shape[0], 1) )
    posdv = np.tile(self.posdv, (dv.shape[0], 1) )
    negdv = np.tile(self.negdv, (dv.shape[0], 1) )
    cts = self.count_model.predict(x)
    cts = np.tile(cts, (dv.shape[1], 1)).transpose()
    f = np.column_stack( (dv.ravel(),
                          row_dv.ravel(),
                          rates.ravel(),
                          posdv.ravel(),
                          negdv.ravel(),
                          cts.ravel()) )
    return f
    
    
class UniformOVA(BaseEstimator):

  def __init__(self, c=1, t1=0, t2=0, null_dv=-99):
    self.t1 = t1
    self.t2 = t2
    self.c = c
    self.null_dv = null_dv
    
  def fit(self, x, y):
    self.models = []
    for k in range(y.shape[1]):
      if (y[:, k]).any():
        model = LinearSVC(C = self.c)
        model.fit(x, y[:, k])
      else:
        model = NullModel(self.null_dv)
      self.models.append(model)
    
  def predict(self, x):
    dvs = self.decision_function(x)
    pred = (dvs > self.t1).astype(float)
    max_dv = dvs.max(1)
    for k in range(pred.shape[0]):
      cut = max_dv[k] - self.t2
      idx = (dvs[k, :] >= cut) 
      pred[k, idx] = 1
    return pred
    
  def decision_function(self, x):
    dvs = np.zeros((x.shape[0], len(self.models)))
    for k in range(len(self.models)):
      dvs[:, k] = self.models[k].decision_function(x)
    return dvs
    
    
class NullModel(BaseEstimator):

  def __init__  (self, null_dv=-99):
    self.null_dv = null_dv
    
  def fit(self, x, y):
    pass
  
  def predict(self, x):
    return self.decision_function(x)
    
  def decision_function(self, x):
    return self.null_dv * np.ones(x.shape[0])


def repredict(dv, t1, t2):
  """
  Takes decision values and returns predictions, given a threshold.
  
  Args:
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

def cvdv(model, x, y, k=3, use_predict=False):
  """
  Get dvs for all of x by training on k folds, predicting on 1,
  and aggregating the predictions into an object the same shape as y.
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

  



  