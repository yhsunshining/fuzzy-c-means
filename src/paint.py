from index import *
from transfer import *

if __name__ == '__main__':
    originImagePath = '../images/filter/stars_200.jpg'
    targetImagePath = '../images/filter/golden_gate_bridge.jpg'
    
    start = time.clock()
    filterData, filterShape = loadImageData(targetImagePath, True, False)
    originData, originShape = loadImageData(originImagePath, True, False)
    filterRange = rangeMat(filterData)
    originRange = rangeMat(originData)
    filterData = normalization(filterData)
    originData = normalization(originData)
    originU, originV = loadUV('../tem/stars_200/mb7')
    targetU, targetV = loadUV('../tem/bridge_400/b7')
    transfer(originU, originV, originData, originRange, targetU, targetV,
             filterData, filterRange)
    print time.clock() - start

    start = time.clock()
    originU,originV = loadUV('../tem/stars_200/ma7')
    targetU, targetV = loadUV('../tem/bridge_400/a7')
    transfer(originU, originV, originData, originRange, targetU, targetV,
             filterData, filterRange)
    print time.clock() - start
