import numpy as np
def te(exp, std):
    n = len(std)
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
    print JC, FMI, RI

te( [0, 0, 1, 1, 2, 2],[0, 0, 0, 1, 1, 1])
