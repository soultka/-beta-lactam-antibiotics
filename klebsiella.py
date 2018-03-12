
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split ,cross_val_predict,KFold,GridSearchCV
from sklearn.neural_network import MLPClassifier
from rdkit.Chem.EState import AtomTypes,EState,EState_VSA,Fingerprinter
from rdkit import Chem
from rdkit.Chem import MACCSkeys,rdMolDescriptors,  GraphDescriptors, Descriptors
#import myset
df1 = pd.read_csv('613207.txt', delimiter='\t')
df2 = pd.read_csv('612612.txt', delimiter='\t')
df3 = pd.read_csv('350.txt', delimiter='\t')

columns = ['KLEB_NUM','MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS' ]
df1 = df1.drop_duplicates(['CMPD_CHEMBLID']) 
df1 = df1.dropna(subset=['CANONICAL_SMILES','MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS' ])

df1['KLEB_NUM'] = [1] * len(df1['PSA'])

df2 = df2.drop_duplicates(['CMPD_CHEMBLID']) 
df2 = df2.dropna(subset=['CANONICAL_SMILES','MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS' ])
df2['KLEB_NUM'] = [2] * len(df2['PSA'])

df3 = df3.drop_duplicates(['CMPD_CHEMBLID']) 
df3 = df3.dropna(subset=['CANONICAL_SMILES','MOLWEIGHT', 'ALOGP', 'PSA',
          'NUM_RO5_VIOLATIONS' ])
df3['KLEB_NUM'] = [3] * len(df3['PSA'])

X_df1 =  np.array(df1[columns])
 
X_df2 =  np.array(df2[columns])

X_df3 =  np.array(df3[columns])

Smiles1 = np.array(df1['CANONICAL_SMILES'])
Smiles2 = np.array(df2['CANONICAL_SMILES'])
Smiles3 = np.array(df3['CANONICAL_SMILES'])

Mol1 = []
for sm in Smiles1.astype(str):
    Mol1.append(Chem.MolFromSmiles(sm))

Mol2 = []
for sm in Smiles2.astype(str):
    Mol2.append(Chem.MolFromSmiles(sm))

Mol3 = []
for sm in Smiles3.astype(str):
    Mol3.append(Chem.MolFromSmiles(sm))
