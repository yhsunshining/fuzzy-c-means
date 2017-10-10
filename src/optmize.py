""" some tools about fcm"""
# -*- coding: utf-8 -*-
import csv
import os
import random
import time
from math import exp

import matplotlib
# matplotlib.use('Agg')
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque


def saveUV(U, V, name):
    """ save membership and centriod mat as csv files

    Args:
        U: membership mat of n*c
        V: centriod mat of c*s
        name: file name to save

    """
    f = open('../tem/' + name + '_U.csv', 'w')
    for i in U:
        k = ','.join([str(j) for j in i])
        f.write(k + "\n")
    f.close()
    print 'save U success'
    f = open('../tem/' + name + '_V.csv', 'w')
    for i in V:
        k = ','.join([str(j) for j in i])
        f.write(k + "\n")
    f.close()
    print 'save V success'


def loadUV(path):
    """ load membership and centriod mat from csv files

    Args:
        path: file name to load without _U.csv and _V.csv

    Returns:
        The tuple(U,V) consist of membership and centriod mat,
        the datatype of each mat is ndarray
    """
    U = loadCsv(path + '_U.csv')
    V = loadCsv(path + '_V.csv')
    return U, V


def loadCsv(path):
    """ load data set from csv file

    Args:
        path: file path to load

    Returns:
        a 2-d ndarray
    """
    lines = csv.reader(open(path, "rb"))
    dataset = list(lines)
    for i in xrange(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return np.array(dataset)


def normalization(dataSet, axis=0):
    """ normaliza the mat by axis

    Args:
        dataSet: a 2-d ndarray to normaliza
        axis: the axis of the ndarray to normaliza,
              default is 0

    Returns:
        the ndarray be normalized
    """
    dataSet = np.float32(dataSet)
    colMax = np.max(dataSet, axis=axis)
    colMin = np.min(dataSet, axis=axis)
    colRange = colMax - colMin
    return (dataSet - colMin) / colRange


def initMembership(n, c):
    """ init membership mat of n*c by random """
    membership = np.random.uniform(0.001, 1, [n, c])
    return membership / np.sum(membership, axis=1).reshape(n, 1)


def initCentroid(dataSet, c):
    """ init Centroid mat of n*c by random """
    dimension = np.shape(dataSet)[1]
    return np.random.rand(dimension * c).reshape(c, dimension)


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
    """ caculate the Hessian mat at a centriod mat """
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
    for i in xrange(c):
        for j in xrange(c):
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
    """ caculate the Hessian mat at a Membership mat """
    dimension = dataSet.shape[1]
    n, c = membership.shape
    centriod = calcCentriod(membership, dataSet, m)
    membershipPower = np.power(membership, m)
    denominator = np.sum(membershipPower, axis=0)
    h = np.repeat(dataSet, c, axis=0).reshape(n, c, dimension) - V
    h = h * (membershipPower / membership).reshape(n, c, 1)
    H = np.zeros((c * n, c * n))
    for i in xrange(c):
        mat = np.dot(h[:, i], h[:, i].T)
        diagD = np.diag(mat) / membershipPower[:, i]
        D = np.diag(diagD)
        G = mat / denominator[i]
        tem = D - 2 * m / (m - 1) * G
        H[i * n:(i + 1) * n, i * n:(i + 1) * n] = m * (m - 1) * tem
    return H


def calcObjective(membership, centriod, dataSet, m):
    """ caculate the value of objective function (J)"""
    membershipPower = np.power(membership, m)
    dist = np.power(distanceMat(centriod, dataSet), 2)
    return np.sum(membershipPower * dist)


def distance(x, y):
    """ the Euclidean distance of dot x and dot y """
    np_x = np.array(x)
    np_y = np.array(y)
    return np.linalg.norm(np_x - np_y)


def distanceMat(centriod, dataSet):
    """ the Euclidean distance mat of two 2-d mat """
    c, dimension = centriod.shape
    n = dataSet.shape[0]
    mat = np.zeros((n, c))
    for i in xrange(c):
        mat[:, i] = np.linalg.norm(dataSet - centriod[i], axis=1)
    return mat


def drawImage(dataSet, exp, c, figName="figure", V=None):
    """ draw image in 2-d dataset

    Args:
        dataSet: the dataSet mat
        exp: the Clustering results of each dot in dataSet, a 1-d adarray
        c: Number of clusters
        figName: figName to save
        V: calcCentriod mat, if the arg is given, calcCentriod will be drawn on the figure

    Returns:
        None, save the scatter in path ./images/d31/
    """
    global figIndex
    contact = np.column_stack((dataSet, exp))
    colors = pltColors.cnames.keys()
    fig = plt.figure()
    for i in xrange(c):
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
    """ get the Clustering results,
    item belong to the cluster which the menbership is max
    """
    return np.array([np.argmax(item) for item in membership])


def evaluate(membership, std, dataSet):
    """ calc the external indicators

    Args:
        membership: membership mat of n*c (ndarray)
        std: the classification of each item (1-d ndarray)
        dataSet: data mat of n*s
    """
    n = len(std)
    classNum = membership.shape[1]
    exp = getExpResult(membership)
    a = b = c = d = 0
    expMat = np.repeat(exp, n).reshape(n, n)
    expFlag = expMat == expMat.T
    stdMat = np.repeat(std, n).reshape(n, n)
    stdFlag = stdMat == stdMat.T
    a = (np.sum(expFlag * stdFlag) - n) / 2.0
    b = np.sum(expFlag * ~stdFlag) / 2.0
    c = np.sum(~expFlag * stdFlag) / 2.0
    d = np.sum(~expFlag * ~stdFlag) / 2.0
    JC = a / (a + b + c)
    FMI = (a**2 / ((a + b) * (a + c)))**(1.0 / 2)
    RI = 2 * (a + d) / (n * (n - 1))
    # print JC, FMI, RI
    return FMI


def fcmIteration(U, V, dataSet, m, c, returnType=0):
    """ fcm iteration start from the init value

    MAX_ITERATION = 50
    epsilon = 1e-8

    Args:
        U: Membership mat of n*c (ndarray)
        V: Centriod mat of c*s (ndarray)
        dataSet: data mat of n*s
        m: m in fcm
        c: numbers of cluster

    Returns:
        The tuple(U,V,J) consist of membership, centriod mat, and the value of objective function
        the mats are all 2-d ndarray
    """
    MAX_ITERATION = 50
    epsilon = 1e-8
    delta = float('inf')
    VQue = deque([V])
    while delta > epsilon and MAX_ITERATION > 0:
        U = calcMembership(V, dataSet, m)
        # J = calcObjective(U, V, dataSet, m)
        # print('{0},{1}').format(J, evaluate(U, classes, dataSet))
        _V = calcCentriod(U, dataSet, m)
        # J = calcObjective(U, _V, dataSet, m)
        # print('{0},{1}').format(J, evaluate(U, classes, dataSet))
        delta = distance(V, _V)**2
        V = _V
        VQue.append(V)
        MAX_ITERATION -= 1
    J = calcObjective(U, V, dataSet, m)
    if returnType == 0:
        return U, V, J
    else:
        return U, V, J, VQue


def fcm(dataSet, m, c, returnType=0):
    """ the Entrance of fcm alg. """
    n = len(dataSet)
    U = initMembership(n, c)
    V = initCentroid(dataSet, c)
    return fcmIteration(U, V, dataSet, m, c, returnType)


def sortByCol(ndarray):
    """ sort 2d ndarray by col val. """
    return ndarray[np.argsort(ndarray[:, 0])]


class TabuSearch:
    """ the Class of Tabu Search
    include the structure and the tool functions

    Attributes:
        tabuList: a list of tabu object
        tabuLength: the tabu length, int, default = 0.25 * MAX_ITERATION
        maxSearchNum: the number of neighborhood samples
        MAX_ITERATION: max iteration number of ts
        neighbourhoodUnit: step length of each move
        neighbourhoodTimes: step number of each move
        extra: a dict of extra attributes,default: None, but usually include dataSet, m and c
    """

    def __init__(self,
                 tabuList=[],
                 tabuLength=None,
                 maxSearchNum=5,
                 MAX_ITERATION=20,
                 neighbourhoodUnit=0.05,
                 neighbourhoodTimes=1,
                 extra={}):
        """Inits TabuSearch with blah."""
        self.tabuList = deque(tabuList[:])
        self.tabuLength = tabuLength or int(0.25 * MAX_ITERATION)
        self.maxSearchNum = maxSearchNum
        self.MAX_ITERATION = MAX_ITERATION
        self.neighbourhoodUnit = neighbourhoodUnit
        self.neighbourhoodTimes = neighbourhoodTimes
        for key in extra.copy():
            setattr(self, key, extra[key])

    def neighbourhoodV(self, V):
        """ get a random sample from the ring neighborhood of a centroid mat
        the raduis is the neighborhood length(move length)
        """
        c, s = V.shape

        """ get a sample from the rect neighborhood of a centroid mat"""
        # _V = np.random.rand(c, s) - 0.5
        # return _V * self.neighbourhoodUnit * self.neighbourhoodTimes + V.copy()

        _V = np.random.randn(c, s)
        tem = np.linalg.norm(_V, axis=1)
        r = (np.random.rand(c, 1) * 0.05 +
             self.neighbourhoodUnit * self.neighbourhoodTimes)
        return _V / tem.reshape(c, 1) * r + V

    def _neighbourhoodV(self, V, lastV):
        c, s = V.shape
        vector = V - lastV
        inertia = (self.neighbourhoodUnit * self.neighbourhoodTimes)
        flagMat = (np.random.rand(c, 1) > 0.5) * 2 - 1

        vector /= np.power(np.sum(np.power(vector, 2),
                                  axis=1), 0.5).reshape(c, 1)
        ineritiaV = V + flagMat * inertia * vector
        return ineritiaV

    def neighbourhood(self, neighbour, extra):
        """ get a sample from the ring neighborhood of a mat
        Args:
            neighbour: the mat
        """
        return self._neighbourhoodV(neighbour, extra)
        # return self.neighbourhoodV(neighbour)

    def tabuJudge(self, obj):
        """ tabu judge by local Convergence of FCM
        Retuens:
            a boolean value
            True: the object is taboo
            False: the object can be selected
        """
        listLength = len(self.tabuList)
        c, dimension = obj.shape
        if not listLength:
            return False
        for tabuIndex in xrange(listLength):
            # sort to eliminate centroid mat sequential interference
            sortObj = sortByCol(obj)
            absMat = np.fabs(self.tabuList[tabuIndex]['value'] - sortObj)
            tabuU = self.tabuList[tabuIndex]['extra']
            sortU = calcMembership(sortObj, self.dataSet, self.m)
            dis = np.linalg.norm(tabuU - sortU)**2 + np.linalg.norm(absMat)**2
            tabuRadius = self.tabuList[tabuIndex]['radius']
            if dis <= tabuRadius:
                # print '-------------- tabu hint ------------------'
                return True

            """ tabu judge by the spectral radius of Heisen matrix"""
            # H = calcMembershipHessian(self.tabuList[tabuIndex]['extra'] , dataSet, m)
            # w = np.linalg.eigvalsh(H)
            """ tabu judge by local areas """
            # if not absMat[absMat > 0.01].shape[0]:
            #     print '-------------- tabu hint ------------------'
            #     return True

        return False

    def addTabuObj(self, tabuObj, extra=None):
        """ add the object into the tabu list

        Args:
            tabuObj: the object to be added, deuault: centroid mat
            extra: extra value to be added, deuault: membership mat
            redius: local convence redius
        """
        sortObj = sortByCol(
            tabuObj)    # sort to eliminate sequential interference
        tabuU = calcMembership(sortObj, self.dataSet, self.m)
        obj = {
            'value': sortObj,
            'extra': extra or tabuU,
            'radius': np.min(np.sum(np.power(tabuU, self.m), axis=0))
        }
        self.tabuList.append(obj)

    def updateList(self, tabuObj):
        if self.tabuLength:
            self.tabuList.popleft()
        self.addTabuObj(tabuObj)

    def start(self, U, V, J, accuracy, VQue):
        """ the Entrance of ts alg.

        epsilon = 1e-6

        use external indicators as evaluation index,
        need to calculate the accuracy, so the classes be a global var.
        """
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
            # Get the candidate solution from the neighborhood
            # and judge every solution
            neighbourhoodVs = deque([])
            judge = deque([])
            for i in xrange(self.maxSearchNum):
                neighbourV = self.neighbourhood(V, VQue[-2])
                neighbourhoodVs.append(neighbourV)
                judge.append(self.tabuJudge(neighbourV))
            neighbourhoodVs = np.array(neighbourhoodVs)
            judge = np.array(judge)

            if not judge.all():
                # All taboo, amnesty
                neighbourhoodVs = neighbourhoodVs[judge == False]
            for neighbourV in neighbourhoodVs:
                temU, temV, temJ, temVQue = fcmIteration(
                    U, neighbourV, self.dataSet, self.m, self.c, 1)
                temA = evaluate(temU, self.classes, self.dataSet)
                """ use J as the evaluation index """
                # if temJ < locationJ:
                if temA > locationA:
                    locationU = temU
                    locationV = temV
                    locationJ = temJ
                    locationA = temA
                    locationVQue = temVQue

            # print('{0},{1}').format(locationJ, locationA)
            ''' Modify the step and move the current solution '''
            # if locationJ < _J:
            if locationA <= accuracy:
                self.neighbourhoodTimes = min(10, self.neighbourhoodTimes + 1)
            else:
                if locationA > _accuracy:
                    _U, _V, _J, _accuracy, _VQue = locationU, locationV, locationJ, locationA, locationVQue
                self.neighbourhoodTimes = max(1, self.neighbourhoodTimes - 1)
            U, V, J, accuracy, VQue = locationU, locationV, locationJ, locationA, locationVQue
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


class SA:
    def __init__(self, param):
        for key in param.copy():
            setattr(self, key, param[key])
        self.V = initCentroid(self.dataSet, self.c)
        self.U = self.J = self.accuracy = None

    def start(self):
        T0 = 0.2
        T = TMAX = 500
        k = 0.9
        MAX_ITERATION = 100
        inertia = 0.7
        epsilon = 1e-8
        curIndex = 0
        _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA = self.U, self.V, self.J, self.accuracy
        vShape = self.V.shape
        for i in xrange(MAX_ITERATION):
            lastV = locationV
            locationU, locationV, locationJ = fcmIteration(
                locationU, locationV, self.dataSet, self.m, self.c)
            locationA = evaluate(locationU, self.classes, self.dataSet)
            flagMat = (np.random.rand(self.c, 1) > 0.5) * 2 - 1
            ineritiaV = locationV + flagMat * inertia * (locationV - lastV)
            temU, temV, temJ = fcmIteration(
                self.U, ineritiaV, self.dataSet, self.m, self.c)
            temA = evaluate(temU, self.classes, self.dataSet)
            # p = exp(float(locationJ - temJ) / (k * T))
            # if (temJ <= locationJ) or p > np.random.rand():
            p = exp(float(temA - locationA) * 500 / T)
            # print p
            if (temA >= locationA) or p > np.random.rand():
                locationU, locationV, locationJ, locationA = temU, temV, temJ, temA
            # if (locationJ < _J):
            if (locationA > _accuracy):
                _U, _V, _J, _accuracy = locationU, locationV, locationJ, locationA
            T = k * T
            if (T < T0) or (distance(lastV, locationV)**2 < epsilon):
                break
            # print("{0},{1}").format(locationJ, locationA)
        return _U, _V, _J, _accuracy


def printResult(accuracy, J):
    # print('Accuracy: {0}%').format(accuracy * 100)
    # print('J: {0}').format(J)
    print('{0},{1}').format(J, accuracy)


if __name__ == '__main__':
    timeString = time.strftime('%Y_%m_%d-%H%M%S')
    """ figIndex init """
    global figIndex
    figIndex = 1
    """ figIndex end """
    dataFilePath = '../data/d31.csv'
    dataSet = loadCsv(dataFilePath)

    global classes
    classes = dataSet[:, -1]
    dataSet = dataSet[:, 0:-1]
    dataSet = normalization(dataSet)
    c = int(len(set(classes)))
    m = int(2)
    """ calc the time of run more times of iteration """
    # start = time.clock()
    # for i in range(0, 100):
    #     U, V, J = fcm(dataSet, m, c)
    #     accuracy = evaluate(U, classes, dataSet)
    #     printResult(accuracy, J)
    # end = time.clock()
    # print end - start
    start = time.clock()
    U, V, J, VQue = fcm(dataSet, m, c, 1)
    accuracy = evaluate(U, classes, dataSet)
    printResult(accuracy, J)
    end = time.clock()
    print end - start
    saveUV(U,V,'d31')
    np.savetxt('../tem/d31_VQue',np.array(VQue[-2]),'%.8f',',')
    # exp= getExpResult(U)
    # drawImage(dataSet,exp,c,'init',V)
    """ tabu search start """
    # start = time.clock()
    # ts = TabuSearch(MAX_ITERATION=5,maxSearchNum=10, extra={
    #     'dataSet': dataSet,
    #     'classes': classes,
    #     'm': m,
    #     'c': c
    # })
    # _U, _V, _J, _accuracy = ts.start(U, V, J, accuracy, VQue)
    # print time.clock() - start
    # printResult(_accuracy, _J)
    # exp = getExpResult(U)
    # H = calcCentroidHessian(V, dataSet, m)
    # w= np.linalg.eigvalsh(H)
    # print w
    # drawImage(dataSet, exp, c, timeString, V)
    """ tabu search end """
    """ SA start """
    # start = time.clock()
    # sa = SA({
    #     'dataSet': dataSet,
    #     'classes': classes,
    #     'm': m,
    #     'c': c
    # })
    # U, V, J, accuracy = sa.start()
    # printResult(accuracy, J)
    # print time.clock() - start
    """ SA end """
