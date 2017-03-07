"""these are some tools about fcm"""
# -*- coding: utf-8 -*-
import numpy as np
import csv


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


def evaluate(membership, std):
    n = len(std)
    exp = [np.argmax(item) for item in membership]
    delta = exp - std
    return 1 - float(np.count_nonzero(delta)) / n


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


if __name__ == '__main__':
    dataFilePath = './data/diabetes.csv'
    dataSet = loadCsv(dataFilePath)
    classes = dataSet[:, -1]
    dataSet = normalization(dataSet[:, 0:-1])
    c = int(2)
    m = int(2)
    U, V, J = fcm(dataSet, m, c)
    print('Accuracy: {0}%').format(evaluate(U, classes) * 100)
