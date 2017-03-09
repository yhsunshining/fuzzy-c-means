"""these are some tools about fcm"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.colors as pltColors
import numpy as np
import pandas as pd
import csv
import random


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


def normalization(dataSet):
    colNum = np.shape(dataSet)[1]
    for index in range(colNum):
        col = dataSet[:, index]
        colMax = np.max(col)
        colMin = np.min(col)
        dataSet[:, index] = (col - colMin) / (colMax - colMin)
    return dataSet


def initMembership(n, c):
    membership = np.random.uniform(0.001, 1, [n, c])
    for i in range(0, n):
        membership[i] = membership[i] / sum(membership[i])
    return membership


def initCentroid(dataSet, c):
    attrNum = np.shape(dataSet)[1]
    centriod = np.zeros((c, attrNum))
    for i in range(0, attrNum):
        centriod[:, i] = np.random.rand(c)
    return centriod


def calcMembership(membership, centriod, m):
    n, c = membership.shape
    membership = np.zeros((n, c))
    for rowIndex in range(n):
        for colIndex in range(c):
            d_ij = distance(dataSet[rowIndex, :], centriod[colIndex, :])
            membership[rowIndex, colIndex] = 1 / np.sum([
                np.power(d_ij / distance(dataSet[rowIndex, :], item),
                         2 / (m - 1)) for item in centriod
            ])
    return membership


def calcCentriod(membership, dataSet, m):
    n, c = membership.shape
    attrNum = dataSet.shape[1]
    centriod = np.zeros((c, attrNum))
    membershipPower = np.power(membership, m)
    for row in range(c):
        member = np.zeros(attrNum, dtype=np.float_)
        denominator = 0
        for i in range(n):
            member += membershipPower[i][row] * dataSet[i]
            denominator += membershipPower[i][row]
        centriod[row, :] = member / denominator
    return centriod


def calcObjective(membership, centriod, dataSet, m):
    n, c = membership.shape
    res = 0
    membershipPower = np.power(membership, m)
    for i in range(n):
        for j in range(c):
            res += membershipPower[i][j] * distance(dataSet[i], centriod[j])
    return res


def distance(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.linalg.norm(np_x - np_y)


def drawImage(dataSet, std, exp, c):
    """ draw image in 2-d dataset """
    contact = np.column_stack((dataSet, std, exp))
    colors = pltColors.cnames.keys()
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
    plt.title('Scatter')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend()
    plt.grid(True)
    plt.show()


def evaluate(membership, std, dataSet):
    n = len(std)
    classNum = membership.shape[1]
    exp = [np.argmax(item) for item in membership]
    # np.set_printoptions(threshold='nan')
    # print exp
    # drawImage(dataSet, std, exp, classNum)
    a = b = c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            expFlag = exp[i] == exp[j]
            stdFlag = std[i] == std[j]
            if expFlag and stdFlag:
                a += 1
                continue
            if (not expFlag) and (not stdFlag):
                d += 1
                continue
            if expFlag and (not stdFlag):
                b += 1
                continue
            if (not expFlag) and stdFlag:
                c += 1
                continue
    a = float(a)
    JC = a / (a + b + c)
    FMI = (a**2 / ((a + b) * (a + c)))**(1.0 / 2)
    RI = 2 * (a + d) / (n * (n - 1))
    print JC, FMI, RI
    return FMI


def fcmIteration(U, V, dataSet, m, c):
    xi = 1e-6
    delta = float('inf')
    while delta > xi:
        U = calcMembership(U, V, m)
        _V = calcCentriod(U, dataSet, m)
        delta = np.sum(np.power(V - _V, 2))
        V = _V
    return U, V, calcObjective(U, V, dataSet, m)


def fcm(dataSet, m, c):
    n = len(dataSet)
    U = initMembership(n, c)
    V = initCentroid(dataSet, c)
    return fcmIteration(U, V, dataSet, m, c)


def neighbourhoodV(V):
    shape = V.shape
    _V = (np.random.rand(*shape) - 0.5) * neighbourhoodLength * 5 + V.copy()
    return _V


# def neighbourhoodU(U, select1, select2):
#     U[:, [select1, select2]] = U[:, [select2, select1]]
#     return U


def neighbourhood(neighbour):
    return neighbourhoodV(neighbour)


def tabuJudge(obj):
    listLength, c, attrNum = tabuList.shape
    if not listLength:
        return False
    for i in range(min(listLength, tabuLength)):
        if np.sum(np.fabs(tabuList[i] -
                          obj)) < c * attrNum * 0.5 * neighbourhoodLength:
            return True
    return False


def updateTable(tabuObj):
    global tabuList
    if tabuList.shape[0]:
        tabuList = np.delete(tabuList, 0, axis=0)
    tabuList = np.row_stack((tabuList, tabuObj.reshape(1, *tabuObj.shape)))


if __name__ == '__main__':
    dataFilePath = './data/user_knowledge.csv'
    dataSet = loadCsv(dataFilePath)
    classes = dataSet[:, -1]
    dataSet = normalization(dataSet[:, 0:-1])
    c = int(4)
    m = int(2)
    # U, V, J = fcm(dataSet, m, c)
    U = loadCsv('./tem/user_knowledge_U.csv')
    V = loadCsv('./tem/user_knowledge_V.csv')
    J = 36.1533220952
    accuracy = evaluate(U, classes, dataSet) * 100
    # accuracy = 36.1533220952
    print('Accuracy: {0}%').format(accuracy)
    print('J: {0}').format(J)
    """tabu search"""
    _U, _V, _J = U, V, J
    global tabuList
    global tabuLength
    global neighbourhoodLength
    neighbourhoodLength = 0.01
    tabuList = np.array([]).reshape(0, *V.shape)
    tabuLength = 5
    maxSearchNum = 6
    MAX_ITERATION = 20
    curTimes = 0
    xi = 1e-6
    lastlocationJ = _J
    while (curTimes < MAX_ITERATION):
        searchNum = 0
        locationJ = float('inf')
        locationU = locationV = None
        while (searchNum < maxSearchNum):
            neighbourV = neighbourhood(V)
            if not tabuJudge(neighbourV):
                temU, temV, temJ = fcmIteration(U, neighbourV, dataSet, m, c)
                if temJ < locationJ:
                    locationU = temU
                    locationV = temV
                    locationJ = temJ
                searchNum += 1
        # print locationJ
        if locationJ < _J:
            _U, _V, _J = locationU, locationV, locationJ
        U, V = locationU, locationV
        updateTable(locationV)
        if abs(lastlocationJ - locationJ) <= xi:
            break
        else:
            lastlocationJ = locationJ
        curTimes += 1

    print('Accuracy: {0}%').format(evaluate(_U, classes, dataSet) * 100)
    print('J: {0}').format(_J)
