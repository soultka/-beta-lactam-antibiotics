# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split ,cross_val_predict,KFold,GridSearchCV
from sklearn.neural_network import MLPClassifier
from rdkit.Chem.EState import AtomTypes,EState,EState_VSA,Fingerprinter
from rdkit import Chem
from rdkit.Chem import MACCSkeys,rdMolDescriptors,  GraphDescriptors, Descriptors

columns = ['MOLWEIGHT', 'ALOGP', 'PSA' , 'NUM_RO5_VIOLATIONS','ASSAY_STRAIN_NUM' ]

def make_y (std_Val):
    y = []
    for val in std_Val:
        if val <= 10:
            y.append(1)
        else:
            y.append(0)
    return y

def make_X_from_df(df, strain_num) :
    num = []
    for strain in df['ASSAY_STRAIN'] :
        num.append(strain_num[strain])
    df['ASSAY_STRAIN_NUM'] = num
    X_raw = np.array(df[columns])
    return X_raw

def makemolfromCanSmiles(df): 
    Mol = []
    for sm in df['CANONICAL_SMILES'] :
        Mol.append(Chem.MolFromSmiles(sm))
    Fingerprint=np.array([])
    MKbits = np.array([])
    rd_Des = np.array([])
    Graph = np.array([])
    Des = np.array([])
    Est_VSA = np.array([])
    i=1
    
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
        if i % 1000 == 0 :
            print(i)
        i+=1
        
    df_len = len(df)
    Fingerprint = Fingerprint.reshape(df_len,-1) 
    MKbits = MKbits.reshape(df_len,-1)
    rd_Des = rd_Des.reshape(df_len,-1)
    Graph = Graph.reshape(df_len,-1)
    Des = Des.reshape(df_len,-1)
    Est_VSA = Est_VSA.reshape(df_len,-1)
    
    return np.column_stack((Fingerprint,MKbits,rd_Des,Graph,Des,Est_VSA))
    
    
    
def accuracy_mlp(Batch_size , Epoch , Hidden_layer ,train_size, X_train, 
                 y_train, X_test, y_test):
    max_it = int(Epoch * Batch_size / train_size) 
    clf = MLPClassifier(hidden_layer_sizes = Hidden_layer ,
                       batch_size = Batch_size , max_iter = max_it)
    clf.fit(X_train,y_train)
    return clf.score(X_test,y_test)
            