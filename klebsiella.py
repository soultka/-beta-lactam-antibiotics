
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


Fingerprint=np.array([])
MKbits = np.array([])
rd_Des = np.array([])
Graph = np.array([])
Des = np.array([])
Est_VSA = np.array([])

for mol in Mol:
    bv = MACCSkeys.GenMACCSKeys(mol)
    keysbits=tuple(bv.GetOnBits())
    MK = np.zeros(166).astype(int)
    for bit in keysbits:
        MK[bit-1] = 1
    MKbits = np.append(MKbits,MK)    
    
    Fingerprint = np.append(Fingerprint,Fingerprinter.FingerprintMol(mol))
    
    Graph = np.append(Graph,GraphDescriptors.BalabanJ(mol))
    
    minChg , maxChg = Descriptors._ChargeDescriptors(mol)
    Des = np.append(Des,minChg)
    Des = np.append(Des,maxChg)
    Des = np.append(Des,Descriptors.NumValenceElectrons(mol))
    Des = np.append(Des,Descriptors.NumRadicalElectrons(mol))
    Des = np.append(Des,Descriptors._ChargeDescriptors(mol))
    
    Est_VSA = np.append(Est_VSA,EState_VSA.EState_VSA_(mol))
   
    rd_Des = np.append(rd_Des,rdMolDescriptors.CalcAUTOCORR2D(mol))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcCrippenDescriptors(mol)[0])) #tuple 2
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcCrippenDescriptors(mol)[1]) )#tuple 2
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcExactMolWt(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcFractionCSP3(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcTPSA(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.MQNs_(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.PEOE_VSA_(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.SMR_VSA_(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.SlogP_VSA_(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcHallKierAlpha(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcKappa1(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcLabuteASA(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAliphaticRings(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAmideBonds(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAromaticCarbocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAromaticHeterocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAromaticRings(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumAtomStereoCenters(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumHBA(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumHBD(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumHeteroatoms(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumHeterocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumLipinskiHBA(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumLipinskiHBD(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumRings(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumRotatableBonds(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumSaturatedRings(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumSpiroAtoms(mol)))
    rd_Des = np.append(rd_Des,np.array(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)))

    

Fingerprint = Fingerprint.reshape(3169,-1) 
MKbits = MKbits.reshape(3169,-1)
rd_Des = rd_Des.reshape(3169,-1)
Graph = Graph.reshape(3169,-1)
Des = Des.reshape(3169,-1)
Est_VSA = Est_VSA.reshape(3169,-1)

X_Finger_Maccs=np.column_stack((X_raw,Fingerprint,MKbits,rd_Des,Graph,Des,Est_VSA))
y = np.array(y)
it=0;
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_Finger_Maccs):
    X_train, X_test = X_Finger_Maccs[train_index], X_Finger_Maccs[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(624,521,531), batch_size=600, solver ='adam',
                        learning_rate_init = 0.001 ,beta_1 = 0.001, beta_2 = 0.001
                        ,max_iter=(int)(1479*3169/132))
   
    clf.fit(X_train , y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred ) 
    print (cm)
    print('Accuarcy ',it,':', clf.score(X_test,y_test))
    it+=1