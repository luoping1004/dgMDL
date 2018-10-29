import numpy as np
import random
import sys
sys.path.insert(0, 'D:/Python/DL_DG/Model/')
from dbn_tf import DBN
from rbm_tf import BBRBM
from util import normalization
import time
from sklearn import linear_model, metrics

import pickle
name = 'D:/Python/DL_DG/10-fold/5foldIdx_shuffle_RN'
with open (name, 'rb') as fp:
    idxDic = pickle.load(fp)
#features of diseases and genes
dm, gm = {}, {}
gm['go'] = np.load('D:/Python/DL_DG/DGI-GO/npy_files/gene_128_80.npy')
gm['ppi'] = np.load('D:/Python/DL_DG/DGI/npy_files/ppi_128.npy')
dm['go'] = np.load('D:/Python/DL_DG/DGI-GO/npy_files/dis_128.npy')
dm['ppi'] = np.load('D:/Python/DL_DG/DGI/npy_files/dis_128.npy')

with open('D:/Python/data/PPI/InBioList.txt') as fr:
    GeneList = []
    for line in fr:
        GeneList.append(line[:-1])
with open('D:/Python/Data/OMIM/clustering/GeneList_v1.1.txt') as fr:
    dgList = []
    for line in fr:
        dgList.append(GeneList.index(line[:-1]))
gm['go'] = gm['go'][dgList]
gm['ppi'] = gm['ppi'][dgList]

logreg = linear_model.LogisticRegression(C=600.0)

hidt = 512
hidb = 256
top_dbn = DBN(hidden_layers_structure=[hidt,hidt,hidt],weight_cost=0.001,batch_size=4,n_epoches=30,
           learning_rate_rbm=[0.01,1e-2,1e-2])

X_train, y_train, dbn = {}, {}, {}
omics = ['go','ppi']

train_index, test_index = idxDic[0]['train'], idxDic[0]['test']
n_sample = train_index.shape[0]+test_index.shape[0]

train_index = np.concatenate((train_index,test_index))
print(train_index.shape)

#------------------------#
#obtain the test set
adj = np.load('D:/Python/Data/OMIM/clustering/Adj_v1.1.npy')
m, n = adj.shape
X_test = {}
print(m,n)
Total = m*n
y_test = np.zeros(Total)
for key in omics:
    X_test[key] = np.zeros((Total,256))

k = 0
for i in range(m):
    for j in range(n):
        for key in omics:
            X_test[key][k] = np.concatenate((dm[key][int(i)],gm[key][int(j)]))
        if adj[i,j] == 1:
            y_test[k] == 1
        k+=1

"""--------------------"""

start = time.time()

for key in omics:
    X_train[key] = np.zeros((n_sample,256))
    y_train[key] = np.zeros(n_sample)

    for i in range(n_sample):
        X_train[key][i] = np.concatenate((dm[key][int(train_index[i,0])],gm[key][int(train_index[i,1])]))
        y_train[key][i] = int(train_index[i,2])

    X_train[key] = normalization(X_train[key])
    X_test[key] = normalization(X_test[key])

    dbn[key] = DBN(hidden_layers_structure=[hidb,hidb,hidb],weight_cost=0.001,batch_size=4,n_epoches=30,
        learning_rate_rbm=[0.0005,1e-2,1e-2],rbm_gauss_visible=True)

    dbn[key].fit(X_train[key])
    X_train[key] = dbn[key].transform(X_train[key])
    X_test[key] = dbn[key].transform(X_test[key])

X_train_joint = np.concatenate((X_train['ppi'],X_train['go']),axis=1)
top_dbn.fit(X_train_joint)
X_train_joint = top_dbn.transform(X_train_joint)

logreg.fit(X_train_joint,y_train['ppi'])

X_test_joint = np.concatenate((X_test['ppi'],X_test['go']),axis=1)
X_test_joint = top_dbn.transform(X_test_joint)

y_score = logreg.predict_proba(X_test_joint)[:,1]


end = time.time()
print('time:',end-start,'s')#2000s

np.save('top10',y_score)
