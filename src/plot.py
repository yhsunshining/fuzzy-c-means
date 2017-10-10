from index import loadCsv
import numpy as np
import matplotlib
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import os.path as path

if __name__ == '__main__':
    dataFilePath = './{0}'
    fcm = 'fcm.200.tra.log'
    sa = 'sa.100.tra.log'
    tsfcm='tsfcm.100.tra.opt.log'
    x_fcm = loadCsv(dataFilePath.format(fcm))[0:99,-1]
    x_tsfcm = loadCsv(dataFilePath.format(tsfcm))[0:99,1]
    x_sa = loadCsv(dataFilePath.format(sa))[0:99,0]
    fig,ax0 = plt.subplots(ncols=1,figsize=(9,6))
    ax0.set_title('PENDIGITS')
    n, bins, patches = ax0.hist(x_fcm,50,histtype='bar',facecolor='yellowgreen',alpha=0.75,edgecolor='black', label="fcm")
    n, bins, patches = ax0.hist(x_tsfcm,50,histtype='bar',facecolor='blue',alpha=0.75,edgecolor='black', label="tsfcm")
    n, bins, patches = ax0.hist(x_sa,50,histtype='bar',facecolor='hotpink',alpha=0.75,edgecolor='black', label="sa")
    plt.legend()
    # fig.subplots_adjust(hspace=0.4)
    plt.show()

