import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split ,cross_val_predict,KFold
from sklearn.neural_network import MLPClassifier
from rdkit.Chem.EState import AtomTypes,EState,EState_VSA,Fingerprinter
from rdkit import Chem
from rdkit.Chem import MACCSkeys,rdMolDescriptors
import myset

df = pd.read_csv('bioactivity-18_5_57_13.txt', delimiter='\t')
columns = ['MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS', 
           
           ]
df = df.drop_duplicates(['CMPD_CHEMBLID']) 
# 중복 제거 666개 
df = df.dropna(subset=['CANONICAL_SMILES','MOLWEIGHT', 'ALOGP', 'PSA',
           'NUM_RO5_VIOLATIONS', 
           ])
# NAN 제거 655
y = np.array(df[['ACTIVITY_COMMENT']])
X_raw = np.array(df[columns])

for i,act_raw in enumerate(y) :
    if act_raw == 'inactive':
        y[i] = 0
    else :
        y[i] = 1
y=y.astype(int) #MLPClassifier를 위한 string -> int 변환

Smiles = np.array(df['CANONICAL_SMILES'])
#smile (CANONICAL SMILE)

Mol = []
for sm in Smiles.astype(str):
    Mol.append(Chem.MolFromSmiles(sm))
# Mol rdkit의 mol 객체 
Index = set(range(0,166)) - myset.MKSET
Index = list(Index)
Fingerprint=[]
MKbits = []
for j,mol in enumerate(Mol):
    Fingerprint.append(Fingerprinter.FingerprintMol(mol))
    bv = MACCSkeys.GenMACCSKeys(mol)
    keysbits=tuple(bv.GetOnBits())
    MK = np.zeros(166).astype(int)
    for bit in keysbits:
        MK[bit-1] = 1
    MK=np.delete(MK,Index)
    MKbits.append(MK)    
    
MKbits = np.array(MKbits)
Maccs = MKbits.reshape(655,-1)
#Maccs key clustering  1024 ,

Fingerprint = np.array(Fingerprint).reshape(655,-1)
#필요없는 feature 제거  

X_Finger_Maccs=np.column_stack((X_raw,Fingerprint,Maccs))
kf = KFold(n_splits=5)
#crossvalidation 

it=0;
for train_index, test_index in kf.split(X_Finger_Maccs):
    X_train, X_test = X_Finger_Maccs[train_index], X_Finger_Maccs[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(148,225,294), batch_size=132, 
                        learning_rate_init = 0.0001 ,beta_1 = 0.001, beta_2 = 0.001 
                        ,max_iter=1500)
   
    clf.fit(X_train , y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred ) 
    print (cm)
    print('Accuarcy ',it,':', clf.score(X_test,y_test))
    it+=1
