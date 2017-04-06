import numpy as np
import pandas as pd
from index import *
import cv2


class TS(TabuSearch):
    def start(self, U, V, J, dataSet):
        _U, _V, _J = U, V, J
        curTimes = 0
        _tabuLength = 0
        epsilon = 1e-6
        lastlocationJ = _J
        while (curTimes < self.MAX_ITERATION):
            searchNum = 0
            locationJ = float('inf')
            locationU = locationV = None
            while (searchNum < self.maxSearchNum):
                neighbourV = self.neighbourhood(V)
                if not self.tabuJudge(neighbourV):
                    temU, temV, temJ = fcmIteration(U, neighbourV, dataSet, m,
                                                    c)
                    if temJ < locationJ:
                        locationU = temU
                        locationV = temV
                        locationJ = temJ
                    searchNum += 1
            if locationJ < _J:
                _U, _V, _J = locationU, locationV, locationJ
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

        return _U, _V, _J


def convert2D(origin, shape, colNum=1):
    return origin.reshape(shape[0] * shape[1], colNum)


def convert3D(origin, shape, colNum=3):
    return origin.reshape(shape[0], shape[1], colNum)


def rangeMat(data, dimension):
    res = np.zeros((2, dimension))
    res[0] = np.max(data, axis=0)
    res[1] = np.min(data, axis=0)
    return res


def transfer(originU, originV, targetU, targetV, targetRange):
    targetV = targetV * (targetRange[0] - targetRange[1]) + targetRange[1]
    targetExp = getExpResult(targetU)
    series = pd.Series(targetExp)
    targetKeys = series.value_counts().keys()

    originExp = getExpResult(lU)
    series = pd.Series(originExp)
    originKeys = series.value_counts().keys()

    img = cv2.imread('./images/lena.jpg', 0)
    img = cv2.equalizeHist(img)
    out = convert2D(img, img.shape, 1) * np.ones(3)
    out = normalization(out)
    for i in range(c):
        mask = originExp == originKeys[i]
        out[mask] = (out[mask] * 0.5 + 0.5) * fV[targetKeys[i]]

    out = np.array(out, dtype=np.uint8)
    out[:, [0, -1]] = out[:, [-1, 0]]
    out = convert3D(out, img.shape, 3)
    plt.imsave('./' + time.strftime('%Y_%m_%d-%H%M%S') + '.png', out)
    pass


if __name__ == '__main__':
    start = time.clock()
    filterPath = './data/picasso_1.csv'
    inPath = './data/lena.csv'
    filterData = loadCsv(filterPath)
    inData = loadCsv(inPath)
    filterRange = rangeMat(filterData, 3)
    filterData = normalization(filterData)
    inData = normalization(inData)
    c = int(6)
    m = int(2)
    fU, fV, fJ = fcm(filterData, m, c)
    lU, lV, lJ = fcm(inData, m, c)
    transfer(lU, lV, fU, fV, filterRange)
    print time.clock() - start

    start = time.clock()
    filterTs = TS(tabuList=np.array([]).reshape(0, *fV.shape),
                  MAX_ITERATION=20)
    fU, fV, fJ = filterTs.start(fU, fV, fJ, filterData)
    filterTs = TS(tabuList=np.array([]).reshape(0, *lV.shape),
                  MAX_ITERATION=20)
    lU, lV, lJ = filterTs.start(lU, lV, lJ, inData)
    transfer(lU, lV, fU, fV, filterRange)
    print time.clock() - start
