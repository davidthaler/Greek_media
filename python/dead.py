import numpy as np

#This was ok, but not great

class StackModel3(BaseEstimator):

  def __init__(self, 
               c=1, 
               t1=0, 
               t2=0, 
               n_estimators=100, 
               max_depth=2, 
               folds=3):
    self.c = c
    self.t1 = t1
    self.t2 = t2
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.folds = folds
    
  def fit(self, x, y):
    self.model1 = UniformOVA(c=self.c)
    dv = cvdv(self.model1, x, y, self.folds)
    self.count_model = Ridge()
    ys = y.sum(1)
    cts = cvdv(self.count_model, x, ys, k=self.folds, use_predict=True)
    row_dv = self.getRowDV(dv)
    self.stack_models = []
    for k in range(y.shape[1]):
      if (y[:, k]).any():
        stack_model = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                 max_depth=self.max_depth)
        f = np.column_stack((dv[:, k], row_dv[:, k], cts))
        stack_model.fit(f, y[:,k])
      else:
        stack_model = NullModel()
      self.stack_models.append(stack_model)
    self.model1.fit(x, y)
    self.count_model.fit(x, ys)
    
  def decision_function(self, x):
    n = len(self.stack_models)
    dv_out = np.zeros( (x.shape[0], n) )
    dv1 = self.model1.decision_function(x)
    cts = self.count_model.predict(x)
    row_dv = self.getRowDV(dv1)
    for k in range(n):
      f = np.column_stack( (dv1[:, k], row_dv[:, k], cts) )
      dv_k = self.stack_models[k].decision_function(f)
      dv_k = np.squeeze(dv_k)
      dv_out[:, k] = dv_k
    return dv_out
    
  def predict(self, x):
    dvs = self.decision_function(x)
    pred = (dvs > self.t1).astype(float)
    max_dv = dvs.max(1)
    for k in range(pred.shape[0]):
      cut = max_dv[k] - self.t2
      idx = (dvs[k, :] >= cut) 
      pred[k, idx] = 1
    return pred

  def getRowDV(self, dv):
    rowmax = dv.max(1)
    row_dv = (dv.transpose() - rowmax).transpose()
    return row_dv



# I just gave up on this...I don't know what I'm doing :(
class myKNN(BaseEstimator):

  def __init__(self, k, c):
    self.k = k
    self.c = c
    
  def fit(self, x, y):
    self.x = x
    self.y = y
    
  def predict(self, x):
    pred = np.zeros( (x.shape[0], self.y.shape[1]) )
    for k in range(x.shape[0]):
      idx = x[k].nonzero()[1]
      xk = x[k, idx]
      base = normalize(self.x[:, idx])
      score = base.dot(xk.transpose())
      # Numpy type purgatory
      score = score.todense()
      score = np.array(score)
      score = score.squeeze()
      # end purgatory
      score_idx = np.argsort(score)[-self.k:]
      yk = y[score_idx]
      lbl_ct = yk.sum(0)
      


#first try at features for a stack model
# didn't really work
def dv2f(dv):
  """
  Get basic dv-based features for a stack model.
  """
  rowmax = dv.max(1)
  rowmax.shape = (len(rowmax), 1)
  row_dv = rowmax - dv
  rowmax = np.tile(rowmax, (1, dv.shape[1]) )
  colmax = dv.max(0)
  col_dv = colmax - dv
  colmax = np.tile(colmax, (dv.shape[0], 1) )
  f = np.column_stack((dv.ravel(), 
                       row_dv.ravel(), 
                       rowmax.ravel(),
                       col_dv.ravel(),
                       colmax.ravel() ))
  return f

# Kernel SVM is unusably slow on the full data
class StackedModel(BaseEstimator):

  def __init__(self, c1=1, c2=1, gamma=0.1, w0=5, sample_rate=5, k=5):
    self.c1 = c1
    self.c2 = c2
    self.gamma = gamma
    self.w0 = w0
    self.sample_rate = sample_rate
    self.k = k
    
  def fit(self, x, y):
    self.lms = UniformOVA(c=self.c1)
    dv = cvdv(self.lms, x, y, self.k)
    self.lms.fit(x, y)
    f = dv2f(dv)
    self.stackmodel = SVC(C = self.c2, 
                          gamma = self.gamma, 
                          class_weight = {0:self.w0})
    # downsample the negatives to self.sample_rate * # positives
    fpos = f[y.ravel() == 1]
    fneg = f[y.ravel() == 0]
    neg_idx = np.random.choice(fneg.shape[0],
                               size=self.sample_rate * fpos.shape[0],
                               replace=False)
    fneg = fneg[neg_idx]
    f = np.concatenate( (fpos, fneg) )
    yf = np.concatenate( ( np.ones(fpos.shape[0]), np.zeros(fneg.shape[0]) ) )
    self.stackmodel.fit(f, yf)

  def predict(self, x):
    dv = self.decision_function(x)
    f = dv2f(dv)
    pred = self.stackmodel.predict(f)
    return pred
    
  def decision_function(self, x):
    return self.lms.decision_function(x)

class VariableOVA(BaseEstimator):

  def __init__(self, c=1, t1=1, t2=0, neg_mult=5, null_dv=-99):
    self.c = c
    self.t1=t1
    self.t2=t2
    self.neg_mult = neg_mult
    self.null_dv = null_dv
    
  def fit(self, x, y):
    self.models = []
    self.idx = []
    n = Normalizer()
    for k in range(y.shape[1]):
      if (y[:, k]).any():
        model = LinearSVC(C = self.c, class_weight='auto')
        xpos = x[y[:, k]==1]
        ypos = np.ones(xpos.shape[0])
        xneg = x[y[:, k]==0]
        neg_idx = choice(xneg.shape[0], 
                         size=self.neg_mult * xpos.shape[0],
                         replace=False)
        xneg = xneg[neg_idx]
        yneg = np.zeros(xneg.shape[0])
        #Get the indices of non-empty columns in xpos
        col_idx = np.asarray(xpos.sum(0).nonzero()[1]).squeeze()
        xk = sp.vstack((xpos, xneg), format='csr')
        yk = np.concatenate((ypos, yneg))
        xk = xk[:, col_idx]
        xk = n.fit_transform(xk)
        model.fit(xk, yk)
      else:
        model = NullModel(self.null_dv)
        #NullModel requires x to have at least one column for prediction
        col_idx = np.array([0])  
      self.models.append(model)
      self.idx.append(col_idx)
  
  def predict(self, x):
    dv = self.decision_function(x)
    return repredict(dv, self.t1, self.t2)
    
  def decision_function(self, x):
    dv = np.zeros((x.shape[0], len(self.models)))
    n = Normalizer()
    for k in range(len(self.models)):
      xk = n.fit_transform(x[:, self.idx[k]])
      dv[:, k] = self.models[k].decision_function(xk)
    return dv


def optimizeThreshold(y, dv, thresholds, default_t=0):
  """
  Choose best threshold out of a set based on micro f1 (per class f1)
  
  Args:
    y - 2d array of 0-1 labels
    dv - 2d array of decision values
    thresholds - sequence of candidate thresholds
    default_t - the default threshold to use if labels are all the same,
      or if all f1 scores are 0.
    
  Returns:
    a threshold vector with of length #classes with the best threshold
    (in a micro-f1 sense) for each class.
  """
  result = default_t * np.ones(dv.shape[1])
  for k in range(dv.shape[1]):
    best_score = 0.0
    for t in thresholds:
      score = f1_score(y[:, k], dv[:, k] > t)
      if score > best_score:
        best_score = score
        result[k] = t
  return result
  
  
class OptimalThreshold(BaseEstimator):

  def __init__(self, model=None, thresholds=None, default_t=0, k=3):
    self.model = model
    self.thresholds = thresholds
    self.default_t = default_t
    self.k = k
    
  def fit(self, x, y):
    folds = KFold(x.shape[0], self.k)
    tvals = np.zeros((y.shape[1], self.k))
    j = 0
    for train, val in folds:
      self.model.fit(x[train], y[train])
      dv = self.model.decision_function(x[val])
      tvals[:, j] = optimizeThreshold(y[val], 
                                       dv, 
                                       self.thresholds, 
                                       self.default_t)
      j = j + 1
    self.fitted_t = np.mean(tvals, axis=1)
    self.model.fit(x, y)
    
  def predict(self, x):
    dv = self.model.decision_function(x)
    return repredict(dv, self.fitted_t)
