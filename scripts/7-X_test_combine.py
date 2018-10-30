import numpy as np


X_train, X_test = {}, {}
"""--------------------"""

#predict
X_test['ppi'] = np.load('X_test_ppi.npy')
X_test['go'] = np.load('X_test_go.npy')
X_test_joint = np.concatenate((X_test['ppi'],X_test['go']),axis=1)

np.save('X_test_joint',X_test_joint)
