import numpy as np

#adj = np.load('Adj_v1.1.npy')
m, n = 1154, 2909

#print(m,n)

y_scores = np.load('top10_score.npy')

with open('InBioList.txt') as fr:
    GeneList = []
    for line in fr:
        GeneList.append(line[:-1])
with open('GeneList_v1.1.txt') as fr:
    dgList = []
    for line in fr:
        dgList.append(line[:-1])

scoreDic = {}

k = 0
for i in range(m):
    for j in range(n):
        if adj[i,j] != 1:
            scoreDic[str(i)+'-'+str(j)] = y_scores[k]
            k+=1
        else:
            k+=1

from operator import itemgetter

result = sorted(scoreDic.items(), key=itemgetter(1), reverse=True)

with open('rank_100.txt', 'w') as fw:
    for i in range(100):
        fw.write(result[i][0])
        fw.write('\t')
        fw.write(str(result[i][1]))
        fw.write('\n')
