import numpy as np
import random
from dbn_tf import DBN
from rbm_tf import BBRBM
from util import normalization
import time
from sklearn import linear_model, metrics


logreg = linear_model.LogisticRegression(C=600.0)

hidt = 512

X_train, X_test = {}, {}
"""--------------------"""


#save the Data
X_train['ppi'] = np.load('X_train_ppi.npy')
X_train['go'] = np.load('X_train_go.npy')
y_train = np.load('y_train.npy')


top_dbn = DBN(hidden_layers_structure=[hidt,hidt,hidt],weight_cost=0.001,batch_size=4,n_epoches=30,
           learning_rate_rbm=[1e-2,1e-2,1e-2])

X_train_joint = np.concatenate((X_train['ppi'],X_train['go']),axis=1)
top_dbn.fit(X_train_joint)
X_train_joint = top_dbn.transform(X_train_joint)

logreg.fit(X_train_joint,y_train)


#predict
X_test_joint = np.load('X_test_joint.npy')

"""--------------------
This batch_size is also used due to the limited amount of RAM,
the program can only analyze 10000 samples at a time.
"""
batch_size = 10000
Total = X_test_joint.shape[0]
n_batches = Total // batch_size + (0 if Total % batch_size == 0 else 1)

y_score = np.zeros(Total)

for b in range(n_batches):
    X_test_joint[b * batch_size:(b + 1) * batch_size] = top_dbn.transform(X_test_joint[b * batch_size:(b + 1) * batch_size])

    y_score[b * batch_size:(b + 1) * batch_size] = logreg.predict_proba(X_test_joint[b * batch_size:(b + 1) * batch_size])[:,1]

np.save('top10_score',y_score)
