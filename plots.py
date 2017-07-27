import numpy as np
import os
from glob import glob
from scipy import misc

import seaborn as sns



def cross_corr_plot(data_in, data_out, labels='', annot=False):

    [data,_] = cross_corr(data_in, data_out)
    
    cmap = sns.cubehelix_palette(8,as_cmap=True)
    if labels is '':
        labels = ['S','FS','Prl','A','G','Pdt','O','N']
    
    ax = sns.heatmap(np.transpose(data), cmap=cmap, #center=0,
                     xticklabels=labels, yticklabels=labels,
                     square=True, annot=annot,annot_kws={"alpha":.65}, linewidths=.7)
    ax.set_xlabel("True Labels",fontsize=12.5)
    ax.set_ylabel("Outputs",fontsize=12.5)
    return ax, data



def cross_corr(labels_in, labels_out):

    [entries,nlabels] = labels_in.shape
    corr = np.zeros([nlabels+1,nlabels+1])

    count = 0
    for i in range(entries):
        if (labels_in[i,:].sum() == 1):
            index = np.where(labels_in[i,:] == np.max(labels_in))
        elif (labels_in[i,:].sum() == 0):
            index = -1
        else:
            continue
        if (labels_out[i,:].sum() == 0):
            corr[index,-1] += 1
        else:
            #print(np.max(labels_out[i,:]))
            corr[index,:-1] += labels_out[i,:] #/ np.max(labels_out[i,:]))
        count += 1

    return corr, count
