"""these are some tools about fcm"""
# -*- coding: utf-8 -*-
import csv
import os
import random
import shutil
import time
from math import exp

import matplotlib
matplotlib.use('Agg')
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
    # return np.random.rand(dimension * c).reshape(c, dimension)
    a = [[0.8574887422413442, 0.10109825224006086], [0.5923083391988512, 0.6658565400716234], [0.2596003660024273, 0.9460051412697439], [0.38962953675882217, 0.4343171489811012], [0.8861786088438488, 0.19190001553322], [0.01934210082889254, 0.20756480700274793], [0.6518368004862588, 0.46769177019926855], [0.2839022819999658, 0.678064652095539], [0.11509072190305047, 0.2984879366372082], [0.6668203272358659, 0.44678854258405565], [0.8909449296247398, 0.8238401480001747], [0.0810878583913075, 0.6886102962333017], [0.3056707358868215, 0.18807918927107725], [0.09641123403222229, 0.9659351509486931], [0.5780858988991402, 0.26787160927748965], [0.7033033675298799, 0.3821575872063153], [0.12546595035596353, 0.6992275366498454], [0.7699690193674422, 0.7388862185105358], [0.14502901067131913, 0.01252988016861345], [0.8291408653052127, 0.5055766696789465], [0.10010879814783802, 0.3277440189356433], [0.4745397266144854, 0.8586158496838048], [0.7254165664857137, 0.919428434087701], [0.12307715088095761, 0.5688056835235162], [0.9690572245387058, 0.08237680627267352], [0.4990706591983708, 0.3905115827781064], [0.45070143955456576, 0.7782498007330346], [0.3591903699479282, 0.22460948217120036], [0.7155802899781109, 0.8536592753752054],[0.35062023601491976, 0.7514458686231675], [0.6289584454881498, 0.6646431337118539]]
    return np.array(a)


def calcMembership(centriod, dataSet, m):
    n = dataSet.shape[0]
    c = centriod.shape[0]
    dist = distanceMat(centriod, dataSet)
    if dist[dist == 0].shape[0]:
        print '-------------- dist == 0 ------------------'
        print centriod.tolist()
        exit(0)
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
    return H


def calcObjective(membership, centriod, dataSet, m):
    membershipPower = np.power(membership, m)
    dist = np.power(distanceMat(centriod, dataSet), 2)
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
        './images/d31/' + str(figIndex) + '.' + str(figName) + '.png',
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
    expMat = np.repeat(exp, n).reshape(n, n)
    expFlag = expMat == expMat.T
    stdMat = np.repeat(std, n).reshape(n, n)
    stdFlag = stdMat == stdMat.T
    a = (np.sum(expFlag * stdFlag) - n) / 2.0
    b = np.sum(expFlag * -stdFlag) / 2.0
    c = np.sum(expFlag * -stdFlag) / 2.0
    d = np.sum(-expFlag * -stdFlag) / 2.0
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
        # print('{0},{1}').format(J, evaluate(U, classes, dataSet))
        _V = calcCentriod(U, dataSet, m)
        # J = calcObjective(U, _V, dataSet, m)
        #drawImage(dataSet,getExpResult(U),c,J,_V )
        # print('{0},{1}').format(J, evaluate(U, classes, dataSet))
        # dis = np.linalg.norm(testU - U)**2 + np.linalg.norm(testV - _V)**2
        # if dis <= np.min(np.sum(np.power(U, 2), axis=0)):
        #     print True
        # else :
        #     print False

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
                 tabuList=[],
                 tabuLength=None,
                 maxSearchNum=5,
                 MAX_ITERATION=20,
                 neighbourhoodUnit=0.01,
                 neighbourhoodTimes=5,
                 extra={}):
        self.tabuList = tabuList
        self.tabuLength = tabuLength or int(0.25 * MAX_ITERATION)
        self.maxSearchNum = maxSearchNum
        self.MAX_ITERATION = MAX_ITERATION
        self.neighbourhoodUnit = neighbourhoodUnit
        self.neighbourhoodTimes = neighbourhoodTimes
        for key in extra:
            setattr(self, key, extra[key])

    def neighbourhoodV(self, V):
        shape = V.shape
        _V = (np.random.rand(*shape) - 0.5
              ) * self.neighbourhoodUnit * self.neighbourhoodTimes + V.copy()
        return _V

    def neighbourhood(self, neighbour):
        return self.neighbourhoodV(neighbour)

    def tabuJudge(self, obj):
        listLength = len(self.tabuList)
        c, dimension = obj.shape
        if not listLength:
            return False
        for tabuIndex in range(listLength):
            sortObj = sortByCol(obj)
            absMat = np.fabs(self.tabuList[tabuIndex]['value'] - sortObj)
            tabuU = self.tabuList[tabuIndex]['extra']
            sortU = calcMembership(sortObj, self.dataSet, self.m)
            dis = np.linalg.norm(tabuU - sortU)**2 + np.linalg.norm(absMat)**2
            if dis <= np.min(np.sum(np.power(sortU, 2), axis=0)):
                print '-------------- tabu hint ------------------'
                return True
            # H = calcMembershipHessian(self.tabuList[tabuIndex]['extra'] , dataSet, m)
            # w = np.linalg.eigvalsh(H)
            # if not absMat[absMat > self.neighbourhoodUnit].shape[0]:
            #     print '-------------- tabu hint ------------------'
            #     return True

        return False

    def addTabuObj(self, tabuObj, extra=None):
        sortObj = sortByCol(tabuObj)
        obj = {
            'value': sortObj,
            'extra': extra or calcMembership(sortObj, self.dataSet, self.m)
        }
        self.tabuList.append(obj)

    def updateList(self, tabuObj):
        if len(self.tabuList):
            del self.tabuList[0]
        self.addTabuObj(tabuObj)

    def start(self, U, V, J, accuracy, dataSet, m, c):
        _U, _V, _J, _accuracy = U, V, J, accuracy
        _tabuLength = 0
        epsilon = 1e-6
        lastlocationJ = _J
        lastA = _accuracy
        for iterationRound in xrange(self.MAX_ITERATION):
            locationJ = float('inf')
            locationA = 0
            locationU = locationV = None
            """ genarator twice loop """
            # neighbourhoodVs = np.array(
            #     [self.neighbourhood(V) for i in xrange(self.maxSearchNum)])
            # judge = np.array(
            #     [self.tabuJudge(neighbourV) for neighbourV in neighbourhoodVs])
            """ once loop """
            neighbourhoodVs = []
            judge = []
            for i in xrange(self.maxSearchNum):
                neighbourV = self.neighbourhood(V)
                neighbourhoodVs.append(neighbourV)
                judge.append(self.tabuJudge(neighbourV))
            neighbourhoodVs = np.array(neighbourhoodVs)
            judge = np.array(judge)

            if not judge.all():
                neighbourhoodVs = neighbourhoodVs[judge == False]
            for neighbourV in neighbourhoodVs:
                temU, temV, temJ = fcmIteration(U, neighbourV, dataSet, m, c)
                temA = evaluate(temU, classes, dataSet)
                # if temJ < locationJ:
                if temA > locationA:
                    locationU = temU
                    locationV = temV
                    locationJ = temJ
                    locationA = temA

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
            # if -epsilon <= lastlocationJ - locationJ <= epsilon:
            #     break
            # else:
            #     lastlocationJ = locationJ

        return _U, _V, _J, _accuracy


def SA(U, V, J, accuracy):
    T0 = 0.2
    T = TMAX = 500
    k = 0.9
    MAX_ITERATION = 100
    inertia = 0.7
    epsilon = 1e-8
    curIndex = 0
    _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA = U, V, J, accuracy
    vShape = V.shape
    for i in xrange(MAX_ITERATION)
        locationU = calcMembership(locationV, dataSet, m)
        lastV = locationV
        locationV = calcCentriod(locationU, dataSet, m)
        locationJ = calcObjective(locationU, locationV, dataSet, m)
        locationA = evaluate(locationU, classes, dataSet)
        temV = locationV + inertia * (locationV - lastV)
        # temV = (np.random.rand(*vShape) - 0.5) * 0.01 + locationV.copy()
        temU = calcMembership(temV, dataSet, m)
        temJ = calcObjective(temU, temV, dataSet, m)
        temA = evaluate(temU, classes, dataSet)
        # p = exp(float(locationJ - temJ) / (k * T))
        # if (temJ <= locationJ) or p > np.random.rand():
        p = exp(float(temA - locationA)*500 / T)
        print p
        if (temA >= locationA) or p > np.random.rand():
            locationU, locationV, locationJ, locationA = temU, temV, temJ, temA
        # if (locationJ < _J):
        if (locationA > _accuracy):
            _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA
        T = k * T
        if (T < T0) or (distance(lastV, locationV)**2 < epsilon):
            break
        print("{0},{1}").format(locationJ, locationA)
    return _U, _V, _J, _accuracy


def printResult(accuracy, J):
    # print('Accuracy: {0}%').format(accuracy * 100)
    # print('J: {0}').format(J)
    print('{0},{1}').format(J, accuracy)


if __name__ == '__main__':
    """ clean up dir"""
    # shutil.rmtree('./images/d31')
    # os.mkdir('./images/d31')
    timeString = time.strftime('%Y_%m_%d-%H%M%S')
    """ clean end """
    """ figIndex init """
    global figIndex
    figIndex = 1
    """ figIndex end """
    dataFilePath = './data/d31.csv'
    dataSet = loadCsv(dataFilePath)
    global classes
    classes = dataSet[:, -1]
    dataSet = normalization(dataSet[:, 0:-1])
    c = int(31)
    m = int(2)
    """ calc the time of run more times of iteration """
    # start = time.clock()
    # for i in range(0,200):
    #     U, V, J = fcm(dataSet, m, c)
    #     accuracy = evaluate(U, classes, dataSet)
    #     printResult(accuracy, J)
    # end = time.clock()
    # print end-start
    # U = loadCsv('./tem/d31_U.csv')
    # V = loadCsv('./tem/d31_V.csv')
    # J = 2.90098815891/0.895706457478
    # J = calcObjective(U, V, dataSet, m)
    # accuracy = evaluate(U, classes, dataSet)
    # printResult(accuracy, J)
    # exp= getExpResult(U)
    # drawImage(dataSet,exp,c,'init',V)
    """ tabu search start """
    # start = time.clock()
    # ts = TabuSearch(MAX_ITERATION=40,extra={
    #     'dataSet':dataSet,
    #     'm':m,
    #     'c':c
    # })
    # U, V, J, accuracy = ts.start(U, V, J, accuracy, dataSet, m, c)
    # print time.clock() - start
    # printResult(accuracy, J)
    # exp = getExpResult(U)
    # H = calcCentroidHessian(V, dataSet, m)
    # w= np.linalg.eigvalsh(H)
    # print w
    # drawImage(dataSet, exp, c, timeString, V)
    """ tabu search end """
    """ SA start """
    start = time.clock()
    V = initCentroid(dataSet, c)
    U = J = accuracy = None
    U, V, J, accuracy = SA(U, V, J, accuracy)
    printResult(accuracy, J)
    print time.clock() - start
    """ SA end """
