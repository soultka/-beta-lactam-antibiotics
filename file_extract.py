import codecs

feature_name = ['STRAINNUM','Fingerprint' , 'Maccs' , 'Auto_Corr' , 'MQN','PEOE_VSA',
                'SMR_VSA', 'slogP_VSA' , 'Crippen1' , 'Crippen2' , 'Molwt' , 'FracCSP3'
                , 'TPSA' , 'HallkierAlpha' , 'kappa1', 'LAbuteASA' , 'AliphaticCarbocycle'
                'AliphaticHeterocycles' , 'AliphaticRings', 'AmideBonds','AromaCarbocycles'
                ,'AromaHeterocycles', 'AromaRings' ,   'AtomStereocenter'
                ,'BrideheadAtoms' , 'HBA' , 'HBD' , 'HeteroAtoms', 'Heterocycles'
                ,'LipinskiHBA', 'LipinskiHBD', 'Ring', 'RotatableBond' 
                ,'SaturatedBond' , 'SaturatedCarbocycles','SaturatedHeterocycles','SaturatedRings','SpiroAtoms', 'UnspecifiedAtomStereoCenters'
                , 'BalabanJ', 'minChg' , 'maxChg' , 'ValenceElec','RadicalElec'
                , 'Estate_VSA'] #finger, mkbits , rd_Des ,Graph, Des ,Est_VSa

feature_size = {'Fingerprint' : 158 ,'Maccs' : 166 , 'Estate_VSA' : 11 , 'Auto_Corr' : 192 , 
  'MQN' : 42  , 'PEOE_VSA' : 14 , 'SMR_VSA' : 10 , 'slogP_VSA' : 12  }

feature_route = 'test_result/feature_data/'

def feature_file_write(file_name , feature_data):
    feature_key = set(feature_size.keys())
    features = ''
    print(feature_key)
    j=0
    for feature in feature_name:
        if feature in feature_key :
            for i in range(feature_size[feature]):
                features += feature + str(i) + '\t'
                j+=1
        else :
            features += feature + '\t'
            j+=1
    features = features[0:-1]
    features += '\n'
    
    print(j)
    print('writestart')
    
    file_name = feature_route + file_name +'.txt'
    f = codecs.open(file_name , 'w' ,'utf-8')
    f.write(features)
    for data in feature_data :
        output_data = ''
        for ft in data:
            output_data =  output_data + str(ft) + '\t'
        output_data = output_data[0:-1]
        output_data = output_data + '\n'
        f.write(str(output_data))
        
    f.close()
