
import pandas as pd
import numpy as np
import make_X_Y as XY
import file_extract

df = pd.read_csv('dataset/chembl350.txt', delimiter='\t')

df = df.dropna(subset=['CANONICAL_SMILES'])

df_std_unit = df['MIC_microM']

y = XY.make_y(df_std_unit) 

strain_Set = set(df['ASSAY_STRAIN'])
strain_Num ={}

for index,strain in enumerate(strain_Set):
    strain_Num[strain] = index+1
 
X_raw = XY.make_X_from_df(df,strain_Num)

mol_factor = XY.makemolfromCanSmiles(df)

X_train = np.column_stack((X_raw , mol_factor))

file_extract.feature_file_write('350_feature' , X_train)
