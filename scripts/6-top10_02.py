import numpy as np
import random
from dbn_tf import DBN
from rbm_tf import BBRBM
from util import normalization
import time
from sklearn import linear_model, metrics

import pickle
name = '5foldIdx_shuffle_RN'
with open (name, 'rb') as fp:
    idxDic = pickle.load(fp)
#features of diseases and genes
dm, gm = {}, {}
gm['ppi'] = np.load('gene_ppi.npy')
dm['ppi'] = np.load('dis_ppi.npy')

with open('InBioList.txt') as fr:
    GeneList = []
    for line in fr:
        GeneList.append(line[:-1])
with open('GeneList_v1.1.txt') as fr:
    dgList = []
    for line in fr:
        dgList.append(GeneList.index(line[:-1]))

gm['ppi'] = gm['ppi'][dgList]

hidt = 512
hidb = 256

X_train, y_train, dbn = {}, {}, {}
omics = ['ppi']

train_index, test_index = idxDic[0]['train'], idxDic[0]['test']
n_sample = train_index.shape[0]+test_index.shape[0]

train_index = np.concatenate((train_index,test_index))
# print(train_index.shape)

#------------------------#
#obtain the test set
adj = np.load('Adj_v1.1.npy')
m, n = adj.shape
X_test = {}
# print(m,n)
Total = m*n
for key in omics:
    X_test[key] = np.zeros((Total,256))

k = 0
for i in range(m):
    for j in range(n):
        for key in omics:
            X_test[key][k] = np.concatenate((dm[key][int(i)],gm[key][int(j)]))
            k+=1

"""--------------------
The following batch_size is only used for predicting the unknown disease-gene associations.
Due to the limited amount of RAM, the program can only analyze 10000 samples at a time.
"""
batch_size = 10000
n_batches = Total // batch_size + (0 if Total % batch_size == 0 else 1)
start = time.time()

for key in omics:
    X_train[key] = np.zeros((n_sample,256))
    y_train[key] = np.zeros(n_sample)

    for i in range(n_sample):
        X_train[key][i] = np.concatenate((dm[key][int(train_index[i,0])],gm[key][int(train_index[i,1])]))
        y_train[key][i] = int(train_index[i,2])

    X_train[key] = normalization(X_train[key])

    for b in range(n_batches):
        X_test[key][b * batch_size:(b + 1) * batch_size] = normalization(X_test[key][b * batch_size:(b + 1) * batch_size])

    dbn[key] = DBN(hidden_layers_structure=[hidb,hidb,hidb],weight_cost=0.001,batch_size=4,n_epoches=30,
        learning_rate_rbm=[0.0005,1e-2,1e-2],rbm_gauss_visible=True)

    dbn[key].fit(X_train[key])
    X_train[key] = dbn[key].transform(X_train[key])

    for b in range(n_batches):
        X_test[key][b * batch_size:(b + 1) * batch_size] = dbn[key].transform(X_test[key][b * batch_size:(b + 1) * batch_size])

#save the Data
np.save('X_train_ppi',X_train[key])
np.save('X_test_ppi',X_test[key])

end = time.time()
print('time:',end-start,'s')
