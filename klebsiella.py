
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

df1_std_unit = list(zip(df1['STANDARD_UNITS'] , df1.index , df1['STANDARD_VALUE'],df1['ACTIVITY_COMMENT']))
df2_std_unit = list(zip(df2['STANDARD_UNITS'] , df2.index , df2['STANDARD_VALUE'],df2['ACTIVITY_COMMENT']))
df3_std_unit = list(zip(df3['STANDARD_UNITS'] , df3.index , df3['STANDARD_VALUE'],df3['ACTIVITY_COMMENT']))

y = [] 
for unit_tuple in df1_std_unit:
    units = str(unit_tuple[0])
    index = unit_tuple[1]
    val = unit_tuple[2]
    comment = unit_tuple[3]
    if units == 'nan' :
        if comment == 'ACTIVE':
            y.append(1)
        else :
            y.append(1)
        
    elif units == 'nM': 
        if val <= 10000:
            y.append(1)
        else :
            y.append(0)
    elif units == 'mm':
        if val <= 10:
            y.append(1)
        else :
            y.append(0)
            
    else :
        df1=df1.drop(unit_tuple[1])

for unit_tuple in df3_std_unit:
    units = str(unit_tuple[0])
    index = unit_tuple[1]
    val = unit_tuple[2]
    comment = unit_tuple[3]
    if units == 'nan' :
        if comment == 'ACTIVE':
            y.append(1)
        else :
            y.append(1)
        
    elif units == 'nM': 
        if val <= 10000:
            y.append(1)
        else :
            y.append(0)
    elif units == 'mm':
        if val <= 10:
            y.append(1)
        else :
            y.append(0)
            
    else :
        df3=df3.drop(unit_tuple[1])

X_df1 =  np.array(df1[columns])
 
#X_df2 =  np.array(df2[columns])

X_df3 =  np.array(df3[columns])

X_raw = np.vstack((X_df1,X_df3))

Smiles1 = np.array(df1['CANONICAL_SMILES'])
#Smiles2 = np.array(df2['CANONICAL_SMILES'])
Smiles3 = np.array(df3['CANONICAL_SMILES'])

Mol = []
for sm in Smiles1.astype(str):
    Mol.append(Chem.MolFromSmiles(sm))

'''
for sm in Smiles2.astype(str):
    Mol.append(Chem.MolFromSmiles(sm))
'''
for sm in Smiles3.astype(str):
    Mol.append(Chem.MolFromSmiles(sm))

Fingerprint=[]
MKbits = []
molD = []
Graph = []
Des = []
for mol in Mol:
    Fingerprint.append(Fingerprinter.FingerprintMol(mol))
    bv = MACCSkeys.GenMACCSKeys(mol)
    keysbits=tuple(bv.GetOnBits())
    MK = np.zeros(166).astype(int)
    for bit in keysbits:
        MK[bit-1] = 1
    MKbits.append(MK)    
    molD.append(rdMolDescriptors.CalcAUTOCORR2D(mol))
    Graph.append(GraphDescriptors.BalabanJ(mol))
    minChg , maxChg = Descriptors._ChargeDescriptors(mol)
    Des.append(minChg)
    Des.append(maxChg)
    
MKbits = np.array(MKbits)
Maccs = MKbits.reshape(3169,-1)
molD = np.array(molD)
Graph = np.array(Graph)
Des = np.array(Des).reshape(3169,-1)
Fingerprint = np.array(Fingerprint).reshape(3169,-1)    

X_Finger_Maccs=np.column_stack((X_raw,Fingerprint,Maccs,molD,Graph,Des))
y = np.array(y)
it=0;
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_Finger_Maccs):
    X_train, X_test = X_Finger_Maccs[train_index], X_Finger_Maccs[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(530,521,531), batch_size=300, solver ='adam',
                        learning_rate_init = 0.0001 ,beta_1 = 0.001, beta_2 = 0.001
                        ,max_iter=(int)(1479*3169/132))
   
    clf.fit(X_train , y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred ) 
    print (cm)
    print('Accuarcy ',it,':', clf.score(X_test,y_test))
    it+=1