import index
import numpy as np 

if __name__ == '__main__':
    fineName = 'pendigits.all'
    dataSet = loadCsv('../data/%s.csv'%(fineName))
    a = []
    sub = 10000
    total = dataSet.shape[0]
    for item in dataSet：
        if np.random.rand() > float(sub)/total:
            a.append(item)
    
    np.savetext('../data/%s-part.csv'%(fineName),np.array(a),',')
