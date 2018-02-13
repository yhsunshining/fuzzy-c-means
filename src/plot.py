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
    # dataFilePath = './{0}'
    # fcm = 'fcm.200.tra.log'
    # sa = 'sa.100.tra.log'
    # tsfcm='circle.log'
    # # x_fcm = loadCsv(dataFilePath.format(fcm))[0:99,-1]
    # x_fcm = loadCsv(dataFilePath.format(tsfcm))[0:99,0]
    # x_tsfcm = loadCsv(dataFilePath.format(tsfcm))[0:99,1]
    # x_sa = loadCsv(dataFilePath.format(sa))[0:99,0]
    # fig,ax0 = plt.subplots(ncols=1,figsize=(9,6))
    # ax0.set_title('PENDIGITS')
    # pltRange = (0.42,0.63)
    # n, bins, patches = ax0.hist(x_fcm,100,histtype='bar',facecolor='yellowgreen',alpha=0.75,edgecolor='black', label="fcm", range=pltRange)
    # n, bins, patches = ax0.hist(x_tsfcm,100,histtype='bar',facecolor='blue',alpha=0.75,edgecolor='black', label="tsfcm",range=pltRange)
    # n, bins, patches = ax0.hist(x_sa,100,histtype='bar',facecolor='hotpink',alpha=0.75,edgecolor='black', label="sa",range=pltRange)
    # plt.legend()
    # plt.show()


    fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(12,5))

    # all_data = loadCsv('../data/boxplotdata.csv')
    all_data = np.array([np.random.normal(0, std, 100) for std in range(6, 10)])
    print len(all_data)
    #fig = plt.figure(figsize=(8,6))
    
    axes[0].violinplot(all_data,showmeans=False,showmedians=True)
    axes[0].set_title('violin plot')
    
    axes[1].boxplot(all_data)
    axes[1].set_title('box plot')
    
    # adding horizontal grid lines
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_xticks([y+1 for y in range(len(all_data))], )
        ax.set_xlabel('xlabel')
        ax.set_ylabel('ylabel')
    
    plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],xticklabels=['abc', 'pso', 'pso-basic', 'tsfcm'],)
    
    plt.show()

