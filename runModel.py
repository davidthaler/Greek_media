import util
import models

SUBMIT_NUM = 1

print 'loading training data...'
xtr, ytr = util.loadTrain()
model = models.UniformOVA(c=1, t1=-0.3, t2=0.1)
print 'fitting model...'
model.fit(xtr, ytr)
print 'loading test data...'
xtest = util.loadTest()
print 'predicting...'
pred = model.predict(xtest)
print 'writing submission...'
util.writeSubmission(SUBMIT_NUM, pred)