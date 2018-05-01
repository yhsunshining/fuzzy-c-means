# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import random
import matplotlib.pyplot as plt
from optmize import *
import time


def b2d(b):  # 将二进制转化为十进制 x∈[0,10]
    t = 0
    for j in range(len(b)):
        t += b[j] * (math.pow(2, j))
    t = t * 10 / 1023
    return t


def calfitvalue(objvalue):  # 转化为适应值，目标函数值越大越好，负值淘汰。
    fitvalue = []
    temp = 0.0
    Cmin = 0
    for i in range(len(objvalue)):
        if(objvalue[i] + Cmin > 0):
            temp = Cmin + objvalue[i]
        else:
            temp = 0.0
        fitvalue.append(temp)
    return fitvalue


def decodechrom(pop):  # 将种群的二进制基因转化为十进制（0,1023）
    temp = []
    for i in range(len(pop)):
        t = 0
        for j in range(10):
            t += pop[i][j] * (math.pow(2, j))
        temp.append(t)
    return temp


def calobjvalue(pop):  # 计算目标函数值
    temp1 = []
    objvalue = []
    temp1 = decodechrom(pop)
    for i in range(len(temp1)):
        x = temp1[i] * 10 / 1023  # （0,1023）转化为 （0,10）
        objvalue.append(10 * math.sin(5 * x) + 7 * math.cos(4 * x))
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


def best(pop, fitvalue):  # 找出适应函数值中最大值，和对应的个体
    px = len(pop)
    bestindividual = []
    bestfit = fitvalue[0]
    for i in range(1, px):
        if(fitvalue[i] > bestfit):
            bestfit = fitvalue[i]
            bestindividual = pop[i]
    return [bestindividual, bestfit]


def sum(fitvalue):
    total = 0
    for i in range(len(fitvalue)):
        total += fitvalue[i]
    return total


def cumsum(fitvalue):
    values = np.zeros(20)
    for i in range(len(fitvalue)):
        t = 0
        j = 0
        while(j <= i):
            t += fitvalue[j]
            j = j + 1
        values[i] = t
    return values


def selection(pop, fitvalue):  # 自然选择（轮盘赌算法）
    newfitvalue = []
    totalfit = sum(fitvalue)
    for i in range(len(fitvalue)):
        newfitvalue.append(fitvalue[i] / totalfit)
    newfitvalue = cumsum(newfitvalue)
    ms = []
    poplen = len(pop)
    for i in range(poplen):
        ms.append(random.random())  # random float list ms
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    while newin < poplen:
        if(ms[newin] < newfitvalue[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if(random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0: cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2


def mutation(pop, pm):  # 基因突变
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if(random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if(pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


class GA():
    def __init__(self, pN, max_iter, dataSet, c, classes):
        (n, s) = dataSet.shape
        self.s = s
        self.pN = pN
        self.pc = 0.9
        self.pm = 0.001
        self.max_iter = max_iter
        self.chromlength = s * c
        self.X = np.zeros((self.pN, c * s))
        self.pbest = np.zeros((self.pN, c * s))
        self.gbest = np.zeros(c * s)
        self.p_fit = np.zeros(self.pN)
        self.fit = -1e-10
        self.dataSet = dataSet
        self.c = c
        self.classes = classes

    def init_Population(self):
        (n, s) = self.dataSet.shape
        for i in xrange(self.pN):
            U, V, J = fcm(self.dataSet, m, self.c)
            self.X[i] = V.reshape(s * self.c)

            self.pbest[i] = self.X[i]
            tmp = evaluate(U, self.classes, self.dataSet)
            self.p_fit[i] = tmp
            if(tmp > self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

    def calculateProbabilities(self, max):
        rand = np.random.rand() * max
        for i in xrange(self.pN):
            rand = rand - self.p_fit[i]
            if rand <= 0:
                return i

    def random(self, exlude):
        index = int(np.random.rand() * (self.pN - 1))
        if index < exlude:
            return index
        else:
            return index + 1

    def crossover(self):  # 个体间交叉，实现基因交换
        newPop = np.zeros((self.pN, self.c * self.s))
        newPop[:] = self.X[:]
        for i in xrange(self.pN):
            if(np.random.rand() < self.pc):
                j = self.random(i)
                cpoint = random.randint(0, self.chromlength)
                newPop[i, 0:j + 1] = self.X[j, 0:j + 1]
                newPop[j, 0:j + 1] = self.X[i, 0:j + 1]
        self.X = newPop

    def mutation(self):  # 基因突变
        for i in range(self.pN):
            if(np.random.rand() < self.pm):
                mpoint = random.randint(0, self.chromlength - 1)
                self.X[i, mpoint] = np.random.rand()

    def iterator(self):
        (n, s) = self.dataSet.shape
        fitness = []
        fitness.append(self.fit)
        for i in xrange(self.max_iter):
            # update pop
            newPop = np.zeros((self.pN, self.c * s))
            maxRandom = np.sum(self.p_fit)
            for j in xrange(self.pN):
                newPop[self.calculateProbabilities(maxRandom)] = self.X[i]
            self.X = newPop
            self.crossover()
            self.mutation()
            for j in xrange(self.pN):
                V = self.X[j].reshape(self.c, self.s)
                U = calcMembership(V, self.dataSet, m)
                U, V, J = fcmIteration(U, V, self.dataSet, m, self.c)
                self.X[i] = V.reshape(self.c * self.s)
                tmp = evaluate(U, self.classes, self.dataSet)
                self.p_fit[i] = tmp
                if(tmp > self.fit):
                    self.fit = tmp
                    self.gbest = self.X[i]

            minIndex = np.argmin(self.p_fit)
            maxIndex = np.argmax(self.p_fit)
            self.p_fit[minIndex] = self.p_fit(maxIndex)
            self.X[minIndex] = self.X[maxIndex]

            fitness.append(self.fit)
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

    for i in xrange(10):
        ga = GA(pN=20, max_iter=5, dataSet=dataSet, c=c, classes=classes)
        start = time.clock()
        ga.init_Population()
        fitness = ga.iterator()
        print (time.clock() - start, fitness[-1])
