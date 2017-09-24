import numpy as np
import pandas as pd
from index import *

if __name__ == '__main__':
    timeString = time.strftime('%Y_%m_%d-%H%M%S')
    """ figIndex init """
    global figIndex
    figIndex = 1
    """ figIndex end """
    dataFilePath = '../data/pendigits.tra.csv'
    dataSet = loadCsv(dataFilePath)

    global classes
    classes = dataSet[:, -1]
    dataSet = dataSet[:, 0:-1]
    # dataSet = normalization(dataSet)
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

    """ tabu search start """

    for i in range(0, 100):
        _U, _V, _J = fcm(dataSet, m, c)
        _accuracy = evaluate(_U, classes, dataSet)
        # printResult(_accuracy, _J)
        start = time.clock()
        ts = TabuSearch(MAX_ITERATION=5, extra={
            'dataSet': dataSet,
            'classes': classes,
            'm': m,
            'c': c
        })
        U, V, J, accuracy = ts.start(_U, _V, _J, _accuracy)
        # printResult(accuracy, J)
        print('{0},{1},{2}').format(_accuracy, accuracy, time.clock() - start)

    """ tabu search end """
    """ SA start """
    start = time.clock()
    sa = SA({
        'dataSet': dataSet,
        'classes': classes,
        'm': m,
        'c': c
    })
    U, V, J, accuracy = sa.start()
    printResult(accuracy, J)
    print time.clock() - start
    """ SA end """
