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
name = 'D:/Python/DL_DG/10-fold/5foldIdx_shuffle_10RN'
with open (name, 'rb') as fp:
    idxDic = pickle.load(fp)
#features of diseases and genes
dm, gm = {}, {}
gm['go'] = np.load('D:/Python/DL_DG/DGI-GO/npy_files/gene_128_80.npy')
gm['ppi'] = np.load('D:/Python/DL_DG/DGI/npy_files/ppi_128.npy')


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


lr, dbn = {}, {}
lr['go'] = linear_model.LogisticRegression(C=600.0)
lr['ppi'] = linear_model.LogisticRegression(C=600.0)
lr['top'] = linear_model.LogisticRegression(C=600.0)

hidt = 512
hidb = 256
top_dbn = DBN(hidden_layers_structure=[hidt,hidt,hidt],weight_cost=0.001,batch_size=4,n_epoches=30,
           learning_rate_rbm=[0.01,1e-2,1e-2])

AUC = {}
AUC['top'] = np.zeros(5)
AUC['go'] = np.zeros(5)
AUC['ppi'] = np.zeros(5)

X_train, y_train, X_test, y_test, dbn = {}, {}, {}, {}, {}
omics = ['go','ppi']

train_index, test_index = idxDic[0]['train'], idxDic[0]['test']
n_sample = train_index.shape[0]+test_index.shape[0]
Y = {}
Y_score = {}

start = time.time()
for k in range(5):
    # print(k)
    dm['ppi'] = np.load('D:/Python/DL_DG/DGI/npy_files/dis_{0}_128.npy'.format(k))
    dm['go'] = np.load('D:/Python/DL_DG/DGI-GO/npy_files/dis_128_{0}.npy'.format(k))

    train_index, test_index = idxDic[k]['train'], idxDic[k]['test']
    for key in omics:
        X_train[key] = np.zeros((train_index.shape[0],dm[key].shape[1]+gm[key].shape[1]))
        y_train[key] = np.zeros(train_index.shape[0])
        X_test[key] = np.zeros((test_index.shape[0],dm[key].shape[1]+gm[key].shape[1]))
        y_test[key] = np.zeros(test_index.shape[0])

        for i in range(X_train[key].shape[0]):
            X_train[key][i] = np.concatenate((dm[key][int(train_index[i,0])],gm[key][int(train_index[i,1])]))
            y_train[key][i] = int(train_index[i,2])
        for i in range(X_test[key].shape[0]):
            X_test[key][i] = np.concatenate((dm[key][int(test_index[i,0])],gm[key][int(test_index[i,1])]))
            y_test[key][i] = int(test_index[i,2])


        X_train[key] = normalization(X_train[key])
        X_test[key] = normalization(X_test[key])

        dbn[key] = DBN(hidden_layers_structure=[hidb,hidb,hidb],weight_cost=0.001,batch_size=4,n_epoches=30,
            learning_rate_rbm=[0.0005,1e-2,1e-2],rbm_gauss_visible=True)

        dbn[key].fit(X_train[key])
        X_train[key] = dbn[key].transform(X_train[key])
        X_test[key] = dbn[key].transform(X_test[key])
        lr[key].fit(X_train[key],y_train[key])
        scores = lr[key].predict_proba(X_test[key])[:,1]
        AUC[key][k] = metrics.roc_auc_score(y_test[key],scores)

    X_train_joint = np.concatenate((X_train['ppi'],X_train['go']),axis=1)
    X_test_joint = np.concatenate((X_test['ppi'],X_test['go']),axis=1)
    top_dbn.fit(X_train_joint)
    X_train_joint = top_dbn.transform(X_train_joint)
    X_test_joint = top_dbn.transform(X_test_joint)

    lr['top'].fit(X_train_joint,y_train['ppi'])
    y_score = lr['top'].predict_proba(X_test_joint)[:,1]

    AUC['top'][k] = metrics.roc_auc_score(y_test['ppi'],y_score)
    Y[k] = y_test['ppi']
    Y_score[k] = y_score

print('go:',np.mean(AUC['go']))
print('ppi:',np.mean(AUC['ppi']))
print('top:',np.mean(AUC['top']))

end = time.time()
print('time:',end-start,'s')#2000s

temp_label = Y[0]
for k in range(1,5):
    temp_label = np.concatenate((temp_label,Y[k]))

temp_score = Y_score[0]
for k in range(1,5):
    temp_score = np.concatenate((temp_score,Y_score[k]))

np.save('dgMDL_score_10RN',temp_score)
np.save('dgMDL_true_10RN',temp_label)
