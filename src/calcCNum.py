"""
HCM2 algorithm with numpy matrix accelerate
"""

from index import *
from transfer import *
import numpy as np
import pandas as pd

#write the cluster algorithm                                          
def Kmeans(nodes,cNum,times):
    xNum=len(nodes)
    np.random.shuffle(nodes)
    means=nodes[:cNum].copy()
    xMatrix=nodes[:,np.newaxis,:].repeat(cNum,axis=1)
    for t in range(times):
        dMatrix=np.sqrt(((xMatrix-means[np.newaxis,:,:])**2).sum(2))
        results=dMatrix.argmin(1)
        nodesHave=np.zeros((xNum,cNum,3))
        nodesHave[range(xNum),dMatrix.argmin(1)]=nodes[range(xNum)]
        nodesCount=np.zeros((xNum,cNum))
        nodesCount[range(xNum),dMatrix.argmin(1)]=1
        nodesCount=nodesCount.sum(0)
        for i in range(cNum):
            if nodesCount[i]>0:
                means[i]=(100*means[i]+nodesHave[:,i].sum(0))/nodesCount[i] 
    #get the the sum of clusters
    return len(np.unique(results))    

#read the data file 
data, shape = loadImageData('../2017_05_08-165016.clustering.png', False, False)
#get the nodes data and answer
nodes = normalization(data)
#using the cluster method   
cNum=Kmeans(nodes,10,50) 
print 'the sum of clusters:',cNum
cNum=Kmeans(nodes,50,50) 
print 'the sum of clusters:',cNum
cNum=Kmeans(nodes,150,50) 
print 'the sum of clusters:',cNum
