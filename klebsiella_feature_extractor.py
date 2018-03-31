
import pandas as pd
import numpy as np
import make_X_Y as XY
import file_extract

file_name = input()

df = XY.make_df_from_file(file_name)
y = XY.make_y(df) 

strain_Set = set(df['ASSAY_STRAIN'])
strain_Num ={}

for index,strain in enumerate(strain_Set):
    strain_Num[strain] = index+1
 
X_raw = XY.make_X_from_df(df,strain_Num)

mol_factor = XY.makemolfromCanSmiles(df)

X_feature = np.column_stack((X_raw , mol_factor,y))

file_extract.feature_file_write('350_feature' , X_feature)
