# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:49:49 2018

@author: KOREA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from rdkit.Chem.EState import AtomTypes,EState,EState_VSA,Fingerprinter
from rdkit import Chem

df = pd.read_csv('bioactivity-18_5_57_13.txt', delimiter='\t')
columns = ['MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS', 
            'PUBLISHED_VALUE' 
           ]
df = df.drop_duplicates(['CMPD_CHEMBLID']) 
# 중복 제거 666개 
df = df.dropna(subset=['CANONICAL_SMILES','MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS', 
            'PUBLISHED_VALUE' ])
# NAN 제거 655
y = np.array(df[['ACTIVITY_COMMENT']])
X_raw = np.array(df[columns])
for i,act_raw in enumerate(y) :
    if act_raw == 'inactive':
        y[i] = 0
    else :
        y[i] = 1
y=y.astype(int)

Smiles = np.array(df['CANONICAL_SMILES'])
#smile (CANONICAL SMILE)

Mol = []
for sm in Smiles.astype(str):
    Mol.append(Chem.MolFromSmiles(sm))
# Mol rdkit의 mol 객체 

Fingerprint=[]
for j,mol in enumerate(Mol):
        Fingerprint.append(Fingerprinter.FingerprintMol(mol))
     
Fingerprint = np.array(Fingerprint).reshape(-1,158)


XwithFingerprint=np.column_stack((X_raw,Fingerprint))

X_train , X_test , y_train , y_test = train_test_split(XwithFingerprint,y)
clf = MLPClassifier(hidden_layer_sizes=(100,100,100,100),max_iter=10000)  
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred ) 

plt.matshow(confusion_matrix)
plt.title('Confusion_matrix : bioactivity(CHEMBL1293246)')
plt.colorbar()
plt.ylabel('True Activity')
plt.xlabel('Predicted Activity')
plt.show()


print('Accuarcy :', clf.score(X_test,y_test))
print('Confusion Matrix :',confusion_matrix)