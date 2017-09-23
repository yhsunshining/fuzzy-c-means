import numpy as np
import pandas as pd
from index import *

if __name__ == '__main__':
    timeString = time.strftime('%Y_%m_%d-%H%M%S')
    """ figIndex init """
    global figIndex
    figIndex = 1
    """ figIndex end """
    dataFilePath = './data/pendigit.csv'
    dataSet = loadCsv(dataFilePath)

    global classes
    classes = dataSet[:, -1]
    # dataSet = normalization(dataSet[:, 0:-1])
    dataSet = dataSet[:, 0:-1]
    c = int(len(set(classes)))
    print c
    m = int(2)
    """ calc the time of run more times of iteration """
    # start = time.clock()
    # for i in range(0, 100):
    #     U, V, J = fcm(dataSet, m, c)
    #     accuracy = evaluate(U, classes, dataSet)
    #     printResult(accuracy, J)
    # end = time.clock()
    # print end - start

    U, V, J = fcm(dataSet, m, c)
    accuracy = evaluate(U, classes, dataSet)
    printResult(accuracy, J)

    

    # exp= getExpResult(U)
    # drawImage(dataSet,exp,c,'init',V)
    """ tabu search start """
    # _U, _V, _J = fcm(dataSet, m, c)
    # _accuracy = evaluate(_U, classes, dataSet)
    # printResult(_accuracy, _J)

    # for i in range(0, 100):
    #     start = time.clock()
    #     ts = TabuSearch(MAX_ITERATION=5, extra={
    #         'dataSet': dataSet,
    #         'm': m,
    #         'c': c
    #     })
    #     U, V, J, accuracy = ts.start(_U, _V, _J, _accuracy, dataSet, m, c)
    #     printResult(accuracy, J)
    #     print time.clock() - start

    # start = time.clock()
    ts = TabuSearch(MAX_ITERATION=5, extra={
        'dataSet': dataSet,
        'm': m,
        'c': c
    })
    U, V, J, accuracy = ts.start(U, V, J, accuracy, dataSet, m, c)
    print time.clock() - start
    printResult(accuracy, J)
    """ tabu search end """
    """ SA start """
    start = time.clock()
    V = initCentroid(dataSet, c)
    U = J = accuracy = None
    U, V, J, accuracy = SA(U, V, J, accuracy)
    printResult(accuracy, J)
    print time.clock() - start
    """ SA end """