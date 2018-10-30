from __future__ import division
import numpy as np
import networkx as nx
import time

G = nx.read_gml('C:/Python/Data/PPI/InBio_Map.gml.gz')
adj = np.load('C:/Python/Data/OMIM/clustering/Adj_v1.1.npy')
dNum,gNum = adj.shape

with open('C:/Python/Data/OMIM/clustering/GeneList_v1.1.txt') as fr:
    dgList = []
    for line in fr:
        dgList.append(line[:-1])
# with open('C:/Python/Data/PPI/InBioList.txt') as fr:
#     GeneList = []
#     for line in fr:
#         GeneList.append(line[:-1])

dsim = np.zeros((dNum,dNum))
spsim = np.zeros((dNum,dNum))

def FG(d, g):
    dg = [dgList[s] for s in np.nonzero(adj[d])[0]]
    total = 0
    for gene in dg:
        if gene == g:
            total+=1
        elif nx.has_path(G, g, gene):
            total+=np.exp(-nx.shortest_path_length(G, g, gene))
    return total/len(dg)

def calSim(i,j):
    A = [dgList[s] for s in np.nonzero(adj[i])[0]]
    B = [dgList[s] for s in np.nonzero(adj[j])[0]]
    FA, FB = 0, 0
    for gene in A:
        FA += FG(j,gene)
    for gene in B:
        FB += FG(i,gene)
    return (FA+FB)/(len(A)+len(B))

start = time.time()
for i in range(dNum):
    for j in range(i,dNum):
        spsim[i,j] = calSim(i,j)
        spsim[j,i] = spsim[i,j]

for i in range(dNum):
    for j in range(i+1,dNum):
        dsim[i,j] = 2*spsim[i,j]/(spsim[i,i]+spsim[j,j])
        dsim[j,i] = dsim[i,j]

end = time.time()
print('time:',end-start,'s')

np.save('C:/Python/DL_DG/DGI/disease_similarity/ModuleSim',dsim)
