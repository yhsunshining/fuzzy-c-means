from index import loadCsv
import numpy as np
import matplotlib
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import os.path as path

if __name__ == '__main__':
    # dataFilePath = './d31.p.tmax.log'
    # data = loadCsv(dataFilePath)
    # labels=['5/5','10/5','15/5','20/5','5/10']
    # colors = pltColors.cnames.keys()
    # colors = ['gold','blue','yellowgreen','lightpink','white']
    # cols = data.shape[1]
    # fig,ax0 = plt.subplots(ncols=1,figsize=(9,6))
    # ax0.set_title('TSFCM IN D31 (p/Tmax)')
    # pltRange = (0.82,0.96)
    # for i in xrange(cols):
    #     n, bins, patches = ax0.hist(data[:,i],5,histtype='bar',facecolor=colors[i],alpha=0.8-0.05*i,edgecolor='black', label=labels[i], range=pltRange)
    # plt.legend()
    # plt.show()

    # fcm sa tsfcm 
    dataFilePath = './{0}'
    fcm = 'fcm.200.tra.log'
    sa = 'sa.100.tra.log'
    tsfcm='new'
    # x_fcm = loadCsv(dataFilePath.format(fcm))[0:99,-1]
    x_fcm = loadCsv(dataFilePath.format(tsfcm))[0:99,0]
    x_tsfcm = loadCsv(dataFilePath.format(tsfcm))[0:99,1]
    x_sa = loadCsv(dataFilePath.format(sa))[0:99,0]
    fig,ax0 = plt.subplots(ncols=1,figsize=(9,6))
    ax0.set_title('PENDIGITS')
    pltRange = (0.42,0.63)
    n, bins, patches = ax0.hist(x_fcm,100,histtype='bar',facecolor='yellowgreen',alpha=0.75,edgecolor='black', label="fcm", range=pltRange)
    n, bins, patches = ax0.hist(x_tsfcm,100,histtype='bar',facecolor='blue',alpha=0.75,edgecolor='black', label="tsfcm",range=pltRange)
    n, bins, patches = ax0.hist(x_sa,100,histtype='bar',facecolor='hotpink',alpha=0.75,edgecolor='black', label="sa",range=pltRange)
    plt.legend()
    plt.show()

