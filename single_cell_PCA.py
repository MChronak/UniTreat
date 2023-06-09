#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import decimal
import copy
from statistics import mean
from statistics import stdev
from tkinter import *
from tkinter import filedialog
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns




#PCA plotting
def single_cell_PCA(dataset, *elements):
    """
    Applies PCA on a given dataset.
    
    Call by:
    single_cell_PCA(datset, *elements)
    
    -dataset: the dataset on which the PCA is to be applied. Better be a pandas dataFrame.
    -element: the Column name to be included in the PCA, given as a string
    
    Output: 
    - PCA relevant triple-plot (bottom to top): Scree plot, PCA plot, loading score plot
    
    """
    sns.set()
    scaled_data = preprocessing.scale(dataset)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
    labels = ["PC"+str(x) for x in range(1,len(per_var)+1)]

        #Make the plot frame
    fig = plt.figure(figsize =(5,18))
    ax = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    plt.subplots_adjust( hspace=0.3)

        #Scree plot
    ax.set_title("Scree Plot")    
    ax.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Percentage of Explained Variance")

    ax.plot(
        range(1,len(per_var)+1),
        np.cumsum(per_var),
        c='red',
        label='Cumul. Explained Variance')
    ax.legend(loc='center right')

        #PCA plot

    pca_df = pd.DataFrame(pca_data, columns=labels)

    ax2.set_title("PCA")
    ax2.scatter(pca_df.iloc[:,0], pca_df.iloc[:,1])
    ax2.set_xlabel("PC1 - {0}%".format(per_var[0]))
    ax2.set_ylabel("PC2 - {0}%".format(per_var[1]))
    ax2.hlines(y=0, xmin=pca_df.iloc[:,0].min() +0.1*pca_df.iloc[:,0].min(), xmax=pca_df.iloc[:,0].max()+0.1*pca_df.iloc[:,0].max(),color='black', linestyle='-', linewidth=0.75)
    ax2.vlines(x=0, ymin=pca_df.iloc[:,1].min() +0.1*pca_df.iloc[:,1].min(), ymax=pca_df.iloc[:,1].max()+0.1*pca_df.iloc[:,1].max(),color='black', linestyle='-', linewidth=0.75)

    # Loading Scores
    pre_loading_scores = pca.components_

    loading_scores = preprocessing.normalize(pre_loading_scores)

    n_features = pca.n_features_
    feature_names = [str(element) for element in elements]
    pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
    pc_loadings = dict(zip(pc_list, loading_scores))

    loadings_df = pd.DataFrame.from_dict(pc_loadings)
    loadings_df['feature_names'] = feature_names
    loadings_df = loadings_df.set_index('feature_names')

    xs = loading_scores[0]
    ys = loading_scores[1]

    for i, varnames in enumerate(feature_names):
        ax3.scatter(xs[i], ys[i], s=200)
        ax3.text(xs[i], ys[i], varnames)
        ax3.arrow(
            0, 0, # coordinates of arrow base
            xs[i], # length of the arrow along x
            ys[i], # length of the arrow along y
            color='r', 
            head_width=0.01
            )

    ax3.set_title("Loading Scores")
    ax3.set_xlabel("PC1 - {0}%".format(per_var[0]))
    ax3.set_ylabel("PC2 - {0}%".format(per_var[1]))

    circle1 = plt.Circle((0, 0), radius = 1, color = "red", fill = False)
    ax3.hlines(y=0, xmin=-1, xmax=1,color='black', linestyle='-', linewidth=0.75)
    ax3.vlines(x=0, ymin=-1, ymax=1,color='black', linestyle='-', linewidth=0.75)
    ax3.add_patch(circle1)

    plt.savefig("The brilliancy plot", bbox_inches = 'tight', dpi = 300)
    sns.reset_orig()


# In[14]:


single_cell_PCA(output,"P31","Mg25","Cu65","Fe57","Zn64")


# In[ ]:


output2

