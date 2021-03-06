"""
This script creates a submission for the Greek Media Multi-label Classification
Challenge (WISE 2014) hosted on Kaggle. Before running this script, the data from
Kaggle has to be transformed by running (one time) the script rewriteData.py.
The submission generated here is created by one of the models used in the final
10th place entry. This model scores 0.769 by itself. The actual 10th place entry, 
which takes a couple of hours to run, can be regenerated by uncommenting the line:

  model = StackModel(...

and it should score 0.775.

project: Kaggle WISE 2014 Greek Media competition
author: David Thaler
"""

import util
import models

SUBMIT_NUM = 1

print 'loading training data...'
xtr, ytr = util.loadTrain()

# This entry uses UniformOVA, which is much faster and almost as good (also 10th place).
model = models.UniformOVA(c=1, t1=-0.3, t2=0.1)

# The actual 10th place entry, which takes a couple of hours, was made with this model:
# model = StackModel(c=1, folds=3, max_depth=2, n_estimators=100, n_stack=10000, t1=2.25, t2=0.9)

print 'fitting model...'
model.fit(xtr, ytr)
print 'loading test data...'
xtest = util.loadTest()
print 'predicting...'
pred = model.predict(xtest)
print 'writing submission...'
util.writeSubmission(SUBMIT_NUM, pred)