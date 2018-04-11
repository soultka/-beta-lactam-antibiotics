import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def drawCorColorMap(X):
    plt.rcParams["figure.figsize"] = [100,100]
    plt.matshow(X.corr(), vmin=-1, vmax=1, cmap='bwr')
    plt.xticks(range(len(X.columns)), range(len(X.columns)))
    plt.yticks(range(len(X.columns)), range(len(X.columns)))
    plt.colorbar()
    plt.show()


def makeCovCorTable(X):
    cols = X.columns
    
    res_cols=['data_x', 'data_y', 'covariance', 'corrlation']
    data = []
    
    for i in range(cols.size):
        for j in range(i+1, cols.size):
            cov = np.cov(X[cols[i]], X[cols[j]])
            var_x = cov[0][0]
            var_y = cov[1][1]
            cov_xy = cov[0][1]
            if cov_xy != 0.0:
                corr = cov_xy/(sqrt(var_x*var_y))
                data.append([cols[i], cols[j], cov_xy, corr])
            print(i, "_", j)
            
    res = pd.DataFrame(data, columns=res_cols)
    sorted_res = res.sort_values(['corrlation', 'covariance'], ascending=[0, 0])
    sorted_res.to_csv('test_result/feature_data/cols_corr.txt', sep='\t')


df = pd.read_csv('test_result/feature_data/350_feature.txt', delimiter='\t') 
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

#drawCorColorMap(X)
#makeCovCorTable(X)