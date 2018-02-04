# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
from optmize import *
import time
#----------------------ABC param---------------------------------


class ABC():
    def __init__(self, pN, max_iter, dataSet, c, classes):
        (n, s) = dataSet.shape
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # employ bee num
        self.max_iter = max_iter
        self.X = np.zeros((self.pN, c, s))
        self.gbest = np.zeros((1, c, s))
        self.p_fit = np.zeros(self.pN)
        self.p_fit = np.zeros(self.pN)
        self.fit = 1e-10
        self.limit = int(max_iter * pN / 10)
        self.dataSet = dataSet
        self.c = c
        self.classes = classes

#---------------------OBJECT FUNCTION Sphere-----------------------------
    def function(self, V):
        U = calcMembership(V, self.dataSet, m)
        # U, V, J = fcmIteration(U, V, self.dataSet, m, self.c)
        # J = calcObjective(U,V,self.dataSet,m)
        return evaluate(U, self.classes, self.dataSet)
        # return J

#---------------------RANDOM-----------------------------
    def random(self, exlude):
        index = int(np.random.rand() * (self.pN - 1))
        if index < exlude:
            return index
        else:
            return index + 1

#---------------------phi-----------------------------
    def phi(self):
        return np.random.rand() * 2 - 1

#---------------------Psi-----------------------------
    def psi(self):
        return np.random.uniform(0, 1.5)

#---------------------CalculateProbabilities----------------
    def calculateProbabilities(self):
        rand = np.random.rand()
        for i in xrange(self.pN):
            rand = rand - self.p_fit[i]
            if rand <= 0:
                return i

#---------------------INIT----------------------------------
    def init_Population(self):
        (n, s) = self.dataSet.shape
        for i in xrange(self.pN):
            # U,V,J = fcm(self.dataSet,m,self.c)
            self.X[i] = initCentroid(self.dataSet, self.c)
            U = calcMembership(self.X[i], self.dataSet, 2)
            tmp = evaluate(U, self.classes, self.dataSet)
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = self.X[i]

#----------------------update positon----------------------------------
    def iterator(self):
        fitness = []
        tem = self.p_fit.copy()
        for t in range(self.max_iter):
            for i in xrange(self.pN):
                k = self.random(i)
                j = int(np.random.rand() * self.c)
                V = self.X.cpoy()
                # φij · (xij − xkj ) + ψij · (gbestj − xij )
                V[j] = self.X[i, j] + self.phi() * (self.X[i, j] - self.X[k, j]) + \
                    self.psi() * (self.gbest[j] - self.X[i, j])
                fit = self.function(V)
                if fit > self.p_fit[i]:
                    self.X = V
                    self.p_fit[i] = fit
                    self.trail[i] = 0
                    if fit > self.fit:
                        self.gbest = V
                        self.fit = fit
                else:
                    self.trail[i] = self.trail[i] + 1

            for ite in xrange(self.pN):
                i = self.calculateProbabilities()
                k = self.random(i)
                j = int(np.random.rand() * self.c)
                V = self.X.cpoy()
                # φij · (xij − xkj ) + ψij · (gbestj − xij )
                V[j] = self.X[i, j] + self.phi() * (self.X[i, j] - self.X[k, j]) + \
                    self.psi() * (self.gbest[j] - self.X[i, j])
                fit = self.function(V)
                if fit > self.p_fit[i]:
                    self.X = V
                    self.p_fit[i] = fit
                    self.trail[i] = 0
                    if fit > self.fit:
                        self.gbest = V
                        self.fit = fit
                else:
                    self.trail[i] = self.trail[i] + 1

            for i in xrange(self.pN):
                if self.trail[i] > self.limit:
                    self.X[i] = initCentroid(self.dataSet, self.c)
                    U = calcMembership(self.X[i], self.dataSet, 2)
                    tmp = evaluate(U, self.classes, self.dataSet)
                    self.p_fit[i] = tmp
                        if tmp > self.fit:
                            self.fit = tmp
                            self.gbest = self.X[i]

            U = calcMembership(self.gbest, self.dataSet, m)
            U, self.gbest, J = fcmIteration(
                U, self.gbest, self.dataSet, m, self.c)
            self.fit = evaluate(U, self.classes, self.dataSet)
            fitness.push(self.fit)
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
    my_abc = ABC(pN=5, max_iter=10, dataSet=dataSet, c=c, classes=classes)
    start = time.clock()
    my_abc.init_Population()
    fitness = my_abc.iterator()
    print time.clock() - start
    #-------------------DRAW--------------------
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 10)])
    fitness = np.array(fitness)
    print fitness
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()
