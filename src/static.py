import matplotlib.pyplot as plt
import numpy as np
from optmize import *
 
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(12,5))

all_data = loadCsv('./test.20.log')
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]
 
#fig = plt.figure(figsize=(8,6))
 
axes[0].violinplot(all_data,
               showmeans=False,
               showmedians=True
               )
axes[0].set_title('violin plot')
 
axes[1].boxplot(all_data,
               )
axes[1].set_title('box plot')
 
# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))], )
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
 
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
        xticklabels=['abc', 'pso', 'pso-basic', 'tsfcm'],
        )
 
plt.show()