import pandas as pd
import numpy as np
from math import sqrt

df = pd.read_csv('test_result/feature_data/350_feature.txt', delimiter='\t') 
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

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