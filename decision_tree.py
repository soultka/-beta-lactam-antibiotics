# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 00:23:33 2018

@author: parkchangho
"""

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import numpy as np
import pandas as pd

df = pd.read_csv('test_result/feature_data/350_feature.txt', delimiter='\t')
columns = df.columns[0:-1]
X = np.array(df[columns])
X = X.astype(float)

y = np.array(df[['activity']])
y = y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

d_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
d_tree.fit(X_train, y_train)
y_pred_tr = d_tree.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_tr))

dot_data = export_graphviz(d_tree, out_file=None,feature_names=columns,
                          class_names=['no', 'yes'],
                          filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
