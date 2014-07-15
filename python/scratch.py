import numpy as np
import pandas as pd

def pairsCount(y, cutoff=100):
  n = y.shape[1]
  pairs = [(j,k) for j in range(n) for k in range(n) if j < k]
  totals = [(y[:, j] * y[:, k]).sum() for (j,k) in pairs]
  totals = np.array(totals)
  pairs = np.array(pairs)
  idx = (totals >= cutoff)
  out = pd.DataFrame({'j':pairs[idx,0], 'k':pairs[idx,1], 'count':totals[idx]})
  return out

def modifiedMax(pred, dv, thr):
  max_dv = dv.max(1)
  repred = pred.copy()
  for k in range(repred.shape[0]):
    cut = max_dv[k] - thr
    idx = (dv[k, :] > cut)
    repred[k, idx] = 1
  return repred
  
