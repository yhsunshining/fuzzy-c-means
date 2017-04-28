"""these are some tools about fcm"""
# -*- coding: utf-8 -*-
import csv
import os
import random
import shutil
import time
from math import exp

import matplotlib
# matplotlib.use('Agg')
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def saveUV(U, V, name):
    f = open('./tem/' + name + '_U.csv', 'w')
    for i in U:
        k = ','.join([str(j) for j in i])
        f.write(k + "\n")
    f.close()
    print 'save U success'
    f = open('./tem/' + name + '_V.csv', 'w')
    for i in V:
        k = ','.join([str(j) for j in i])
        f.write(k + "\n")
    f.close()
    print 'save V success'


def loadCsv(filename):
    """ loas data set """
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return np.array(dataset)


def normalization(dataSet, axis=0):
    dataSet = np.float32(dataSet)
    colMax = np.max(dataSet, axis=axis)
    colMin = np.min(dataSet, axis=axis)
    colRange = colMax - colMin
    return (dataSet - colMin) / colRange


def initMembership(n, c):
    membership = np.random.uniform(0.001, 1, [n, c])
    return membership / np.sum(membership, axis=1).reshape(n, 1)


def initCentroid(dataSet, c):
    dimension = np.shape(dataSet)[1]
    return np.random.rand(dimension * c).reshape(c, dimension)


def calcMembership(centriod, dataSet, m):
    n = dataSet.shape[0]
    c = centriod.shape[0]
    dist = distanceMat(centriod, dataSet)
    if dist[dist == 0].shape[0]:
        print '-------------- dist == 0 ------------------'
    distPower = np.power(dist, -2.0 / (m - 1))
    return distPower / np.dot(
        np.sum(distPower, axis=1).reshape((n, 1)), np.ones((1, c)))


def calcCentriod(membership, dataSet, m):
    n, c = membership.shape
    dimension = dataSet.shape[1]
    centriod = np.zeros((c, dimension))
    membershipPower = np.power(membership, m)
    denominator = np.dot(
        np.sum(membershipPower, axis=0).reshape((c, 1)),
        np.ones((1, dimension)))
    return np.dot(membershipPower.T, dataSet) / denominator


def calcCentroidHessian(centriod, dataSet, m):
    n = dataSet.shape[0]
    c, dimension = centriod.shape
    U = calcMembership(centriod, dataSet, m)
    membershipPower = np.power(U, m)
    sumByn = np.sum(membershipPower, axis=0)
    A = 2 * sumByn.reshape(c, 1, 1) * np.eye(dimension) - (4 * m / (
        m - 1) * sumByn).reshape(c, 1, 1)
    distMat = distanceMat(centriod, dataSet)
    distPower = np.power(distMat, -2.0 / (m - 1))
    gk = np.sum(distPower, axis=1)
    gkPower = np.power(gk, m - 1)
    h = np.repeat(dataSet, c, axis=0).reshape(n, c, dimension) - V
    h = h * membershipPower.reshape(n, c, 1)
    H = np.zeros((c * dimension, c * dimension))
    for i in range(c):
        for j in range(c):
            Bij = 4 * m / (m - 1) * np.sum(
                np.sum(h[:, i] * h[:, j], axis=1) * gkPower)
            if i == j:
                tem = A[i] + Bij
            else:
                tem = Bij
            H[i * dimension:(i + 1) * dimension, j * dimension:(j + 1) *
              dimension] = tem
    return H


def calcMembershipHessian(membership, dataSet, m):
    dimension = dataSet.shape[1]
    n, c = membership.shape
    centriod = calcCentriod(membership, dataSet, m)
    membershipPower = np.power(membership, m)
    denominator = np.sum(membershipPower, axis=0)
    h = np.repeat(dataSet, c, axis=0).reshape(n, c, dimension) - V
    h = h * (membershipPower / membership).reshape(n, c, 1)
    H = np.zeros((c * n, c * n))
    for i in range(c):
        mat = np.dot(h[:, i], h[:, i].T)
        diagD = np.diag(mat) / membershipPower[:, i]
        D = np.diag(diagD)
        G = mat / denominator[i]
        tem = D - 2 * m / (m - 1) * G
        H[i * n:(i + 1) * n, i * n:(i + 1) * n] = m * (m - 1) * tem
        print i
    return H


def calcObjective(membership, centriod, dataSet, m):
    membershipPower = np.power(membership, m)
    dist = np.power(distanceMat(centriod, dataSet),2)
    return np.sum(membershipPower * dist)


def distance(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.linalg.norm(np_x - np_y)


def distanceMat(centriod, dataSet):
    c, dimension = centriod.shape
    n = dataSet.shape[0]
    mat = np.zeros((n, c))
    for i in range(c):
        mat[:, i] = np.linalg.norm(dataSet - centriod[i], axis=1)
    return mat


def drawImage(dataSet, exp, c, figName="figure", V=None):
    """ draw image in 2-d dataset """
    global figIndex
    contact = np.column_stack((dataSet, exp))
    colors = pltColors.cnames.keys()
    fig = plt.figure()
    for i in range(c):
        mask = contact[:, -1] == i
        select = contact[mask]
        x, y = select[:, 0], select[:, 1]
        plt.scatter(
            x,
            y,
            c=colors[i],
            label=str(i),
            s=100,
            marker="${}$".format(i),
            alpha=1,
            edgecolors='none')
        if V <> None:
            plt.scatter(
                V[i][0],
                V[i][1],
                c=colors[i],
                label=str(i),
                s=100,
                marker="o",
                alpha=1,
                edgecolors='white')
    plt.title(str(figName))
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend()
    plt.grid(True)
    fig.savefig(
        './images/R15/' + str(figIndex) + '.' + str(figName) + '.png',
        dpi=fig.dpi)
    figIndex += 1
    # plt.show()


def getExpResult(membership):
    return np.array([np.argmax(item) for item in membership])


def evaluate(membership, std, dataSet):
    n = len(std)
    classNum = membership.shape[1]
    exp = getExpResult(membership)
    a = b = c = d = 0
    expMat = np.repeat(exp,n).reshape(n,n)
    expFlag = expMat == expMat.T
    stdMat = np.repeat(std,n).reshape(n,n)
    stdFlag = stdMat == stdMat.T
    a = (np.sum(expFlag * stdFlag) - n)/2.0
    b = np.sum(expFlag * -stdFlag) / 2.0
    c = np.sum(expFlag * -stdFlag) /2.0
    d = np.sum(-expFlag * -stdFlag) /2.0
    JC = a / (a + b + c)
    FMI = (a**2 / ((a + b) * (a + c)))**(1.0 / 2)
    RI = 2 * (a + d) / (n * (n - 1))
    # print JC, FMI, RI
    # drawImage(dataSet,  exp, classNum, str(FMI))
    return FMI


def fcmIteration(U, V, dataSet, m, c):
    MAX_ITERATION = 50
    epsilon = 1e-8
    delta = float('inf')
    while delta > epsilon and MAX_ITERATION > 0:
        U = calcMembership(V, dataSet, m)
        # J = calcObjective(U, V, dataSet, m)
        #drawImage(dataSet,getExpResult(U),c,J,V )
        #print('{0},{1}').format(J, evaluate(U, classes, dataSet))
        _V = calcCentriod(U, dataSet, m)
        # J = calcObjective(U, _V, dataSet, m)
        #drawImage(dataSet,getExpResult(U),c,J,_V )
        # print('{0},{1}').format(J, evaluate(U, classes, dataSet))
        delta = distance(V, _V)**2
        V = _V
        MAX_ITERATION -= 1
    J = calcObjective(U, V, dataSet, m)
    return U, V, J


def fcm(dataSet, m, c):
    n = len(dataSet)
    U = initMembership(n, c)
    V = initCentroid(dataSet, c)
    return fcmIteration(U, V, dataSet, m, c)


def sortByCol(ndarray):
    return ndarray[np.argsort(ndarray[:, 0])]


class TabuSearch:
    def __init__(self,
                 tabuList,
                 tabuLength=5,
                 maxSearchNum=5,
                 MAX_ITERATION=20,
                 neighbourhoodUnit=0.01,
                 neighbourhoodTimes=5):
        self.tabuList = tabuList
        self.tabuLength = tabuLength
        self.maxSearchNum = maxSearchNum
        self.MAX_ITERATION = MAX_ITERATION
        self.neighbourhoodUnit = neighbourhoodUnit
        self.neighbourhoodTimes = neighbourhoodTimes

    def neighbourhoodV(self, V):
        shape = V.shape
        _V = (np.random.rand(*shape) - 0.5
              ) * self.neighbourhoodUnit * self.neighbourhoodTimes + V.copy()
        return _V

    def neighbourhood(self, neighbour):
        return self.neighbourhoodV(neighbour)

    def tabuJudge(self, obj):
        listLength, c, dimension = self.tabuList.shape
        if not listLength:
            return False
        for tabuIndex in range(listLength):
            sortObj = sortByCol(obj)
            absMat = np.fabs(self.tabuList[tabuIndex] - sortObj)
            # H = calcCentroidHessian(self.tabuList[tabuIndex] , dataSet, m)
            # w = np.linalg.eigvalsh(H)
            # print np.min(np.fabs(w))
            # print '---------'
            # print distance(self.tabuList[tabuIndex],sortObj)
            # if np.min(np.fabs(w)) < distance(self.tabuList[tabuIndex],sortObj):
            #     print '2222'
            if not absMat[absMat > self.neighbourhoodUnit].shape[0]:
                print '-------------- tabu hint ------------------'
                return True

        return False

    def addTabuObj(self, tabuObj):
        sortObj = sortByCol(tabuObj)
        self.tabuList = np.row_stack(
            (self.tabuList, sortObj.reshape(1, *sortObj.shape)))

    def updateList(self, tabuObj):
        if self.tabuList.shape[0]:
            self.tabuList = np.delete(self.tabuList, 0, axis=0)
        self.addTabuObj(tabuObj)

    def start(self, U, V, J, accuracy, dataSet, m, c):
        _U, _V, _J, _accuracy = U, V, J, accuracy
        curTimes = 0
        _tabuLength = 0
        epsilon = 1e-6
        lastlocationJ = _J
        lastA = _accuracy
        while (curTimes < self.MAX_ITERATION):
            searchNum = 0
            locationJ = float('inf')
            locationA = 0
            locationU = locationV = None
            while (searchNum < self.maxSearchNum):
                neighbourV = self.neighbourhood(V)
                if not self.tabuJudge(neighbourV):
                    temU, temV, temJ = fcmIteration(U, neighbourV, dataSet, m,
                                                    c)
                    temA = evaluate(temU, classes, dataSet)
                    # if temJ < locationJ:
                    if temA > locationA:
                        locationU = temU
                        locationV = temV
                        locationJ = temJ
                        locationA = temA
                    searchNum += 1
            print('{0},{1}').format(locationJ, locationA)
            # if locationJ < _J:
            if locationA > _accuracy:
                _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA
                self.neighbourhoodTimes = max(5, self.neighbourhoodTimes - 5)
            else:
                self.neighbourhoodTimes = min(50, self.neighbourhoodTimes + 5)
            U, V = locationU, locationV
            if _tabuLength < self.tabuLength:
                self.addTabuObj(locationV)
                _tabuLength += 1
            else:
                self.updateList(locationV)
            if abs(lastlocationJ - locationJ) <= epsilon:
                break
            else:
                lastlocationJ = locationJ
            curTimes += 1

        return _U, _V, _J, _accuracy


def SA(U, V, J, accuracy):
    T0 = 200
    T = TMAX = 500
    k = 0.99
    MAX_ITERATION = 120
    p = 1 - 1e-4
    curIndex = 0
    _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA = U, V, J, accuracy
    vShape = V.shape
    while (curIndex < MAX_ITERATION):
        locationU = calcMembership(locationV, dataSet, m)
        locationV = calcCentriod(locationU, dataSet, m)
        locationJ = calcObjective(locationU, locationV, dataSet, m)
        locationA = evaluate(_U, classes, dataSet)
        temV = (np.random.rand(*vShape) - 0.5) * 0.01 + locationV.copy()
        temU = calcMembership(temV, dataSet, m)
        temJ = calcObjective(temU, temV, dataSet, m)
        temA = evaluate(temU, classes, dataSet)
        _p = exp(float(locationJ - temJ) / (k * T))
        if (temJ <= locationJ) or _p > p:
            # _p = exp(float(temA - locationA) / (k * T))
            # if (temA >= locationA) or _p > p:
            locationU, locationV, locationJ, locationA = temU, temV, temJ, temA
        if (locationJ < _J):
            # if (locationA > _accuracy):
            _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA
        T = k * T
        # print T
        # if (T < T0):
        #     break
        curIndex += 1
        print("{0},{1}").format(locationJ, locationA)
    return _U, _V, _J, _accuracy


def printResult(accuracy, J):
    # print('Accuracy: {0}%').format(accuracy * 100)
    # print('J: {0}').format(J)
    print('{0},{1}').format(J, accuracy)


if __name__ == '__main__':
    """ clean up dir"""
    # shutil.rmtree('./images/R15')
    # os.mkdir('./images/R15')
    timeString = time.strftime('%Y_%m_%d-%H%M%S')
    """ clean end """
    """ figIndex init """
    global figIndex
    figIndex = 1
    """ figIndex end """
    dataFilePath = './data/iris.csv'
    dataSet = loadCsv(dataFilePath)
    global classes
    classes = dataSet[:, -1]
    dataSet = normalization(dataSet[:, 0:-1])
    c = int(3)
    m = int(2)
    """ calc the time of run more times of iteration """
    start = time.clock()
    for i in range(0,200):
        U, V, J = fcm(dataSet, m, c)
        accuracy = evaluate(U, classes, dataSet)
        printResult(accuracy, J)
    end = time.clock()
    print end-start
    # U = loadCsv('./tem/R15_U.csv')
    # V = loadCsv('./tem/R15_V.csv')
    # H = calcMembershipHessian(U, dataSet, m)
    # H = calcCentroidHessian(V, dataSet, m)
    # w = np.linalg.eigvalsh(H)
    # print np.average(w)
    # print np.max(w)
    # print len(w[w < 0])
    # print w[w < 0]
    # J = 11.9517360284
    # J = calcObjective(U,V,dataSet,m)
    # accuracy = evaluate(U, classes, dataSet)
    # printResult(accuracy, J)
    # exp= getExpResult(U)
    # drawImage(dataSet,exp,c,'init',V)
    """ tabu search start """
    start = time.clock()
    ts = TabuSearch(
        tabuList=np.array([]).reshape(0, *V.shape), MAX_ITERATION=40)
    U, V, J, accuracy = ts.start(U, V, J, accuracy, dataSet, m, c)
    print time.clock() - start
    printResult(accuracy, J)
    # exp = getExpResult(U)
    # H = calcCentroidHessian(V, dataSet, m)
    # w= np.linalg.eigvalsh(H)
    # print w
    # drawImage(dataSet, exp, c, timeString, V)
    """ tabu search end """
    """ SA start """
    # U, V, J, accuracy = SA(U, V, J, accuracy)
    # printResult(accuracy, J)
    """ SA end """
