# -*- coding: utf-8 -*-   
import numpy as np
import random 
import matplotlib.pyplot as plt
from optmize import *
import time
#----------------------PSO param---------------------------------
class PSO():
    def __init__(self,pN,max_iter,dataSet,c, classes):
        (n,s) = dataSet.shape
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1= 0.6
        self.r2=0.3
        self.pN = pN                #num
        self.max_iter = max_iter    
        self.X = np.zeros((self.pN,c,s))       
        self.V = np.zeros((self.pN,c,s))
        self.pbest = np.zeros((self.pN,c,s))   
        self.gbest = np.zeros((1,c,s))
        self.p_fit = np.zeros(self.pN)              
        self.fit = 1e10
        self.dataSet = dataSet
        self.c = c
        self.classes = classes
        
#---------------------OBJECT FUNCTION Sphere-----------------------------
    def function(self,V):
        U = calcMembership(V,self.dataSet,m)
        U,V,J = fcmIteration(U,V,self.dataSet,m,self.c)
        # J = calcObjective(U,V,self.dataSet,m)
        return 1.0/evaluate(U,self.classes,self.dataSet)
        # return J
#---------------------INIT----------------------------------
    def init_Population(self):
        (n,s) = self.dataSet.shape
        for i in xrange(self.pN):
            U,V,J = fcm(self.dataSet,m,self.c)
            self.X[i] = V
            self.V[i] = np.random.uniform(0,1,[c,s])

            self.pbest[i] = self.X[i]
            tmp = evaluate(U,self.classes,self.dataSet)
            self.p_fit[i] = tmp
            if(tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]
    
#----------------------update positon----------------------------------
    def iterator(self):  
        fitness = []  
        tem = self.p_fit.copy()
        for t in range(self.max_iter):
            for i in xrange(self.pN):         #update gbest\pbest
               temp = tem[i]
               if(temp>self.p_fit[i]):      #update pbest
                   self.p_fit[i] = temp  
                   self.pbest[i] = self.X[i]  
                   if(self.p_fit[i] > self.fit):  #update gbest
                       self.gbest = self.X[i]  
                       self.fit = self.p_fit[i]  
            for i in range(self.pN):  
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i]) + self.c2*self.r2*(self.gbest - self.X[i])  
                V = self.X[i] + self.V[i]
                U = calcMembership(V,self.dataSet,m)
                U,V,J = fcmIteration(U,V,self.dataSet,m,self.c)
                self.X[i] = V
                tem[i] = evaluate(U,self.classes,self.dataSet)
            fitness.append(self.fit)  
            print(self.p_fit)                   #print best
        return fitness

if __name__ == '__main__':
    dataFilePath = '../data/pendigits.tra.csv'
    dataSet = loadCsv(dataFilePath)

    global classes
    classes = dataSet[:, -1]
    dataSet = dataSet[:, 0:-1]
    dataSet = normalization(dataSet)
    c = int(len(set(classes)))
    m = int(2)

    #----------------------RUN-----------------------
    my_pso = PSO(pN=5,max_iter=10,dataSet = dataSet, c=c,classes = classes)
    start = time.clock()
    my_pso.init_Population()
    fitness = my_pso.iterator()
    print time.clock() - start
    #-------------------DRAW--------------------
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0,10)])
    fitness = np.array(fitness)
    print fitness
    plt.plot(t,fitness, color='b',linewidth=3)
    plt.show()