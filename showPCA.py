import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd


def make_df_from_feature(file):
    df = pd.read_csv( file , delimiter='\t')
    return df

def PCA_3D(df):
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]    

    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d') 

    pca = PCA(n_components=3)

    red_x , red_y , red_z= [] , [] ,[]
    blue_x, blue_y , blue_z = [] , [] ,[]

    reduced_x = pca.fit_transform(X)

    for i in range(len(reduced_x)) :
        if y[i] == 0 :
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
            red_z.append(reduced_x[i][2])
        elif y[i] == 1 :
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
            blue_z.append(reduced_x[i][2])

    ax.scatter(red_x,red_y,red_z , marker = 'o' ,color = 'r', s=1 , label = 'inactive')
    ax.scatter(blue_x,blue_y,blue_z , marker = '^' ,color = 'b' ,s=1 ,label = 'active')

    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3') 
    ax.legend()
    ax.grid(True)
    plt.title('CHEMBL350')
    plt.show()
