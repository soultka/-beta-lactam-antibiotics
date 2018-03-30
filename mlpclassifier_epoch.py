import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from random import randint
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# different learning rate schedules and momentum parameters
param = {'solver': 'adam', 'learning_rate_init': 0.01}
label = "adam"
plot_arg = [{'c': 'blue', 'linestyle': '-'}, {'c': 'red', 'linestyle': '-'}]
mlp = []

def makeHLRandom():
    return (randint(1, 300), randint(1, 300), randint(1, 300))

def plot_on_dataset(X, y, hidden, learing_late, batch_size=132):
    #X = MinMaxScaler().fit_transform(X)
    train_scores = []
    test_scores = []
    epoch = 5000

    # Divide the data set into 8 to 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlp = MLPClassifier(hidden_layer_sizes=hidden, batch_size=batch_size, learning_rate_init=learing_late, beta_1=0.001,
                             beta_2=0.001, max_iter=int(4*1*(132/batch_size)), warm_start=True)

    for i in range(epoch):
        mlp.fit(X_train, y_train)
        train_scores.append(mlp.score(X_train, y_train))
        test_scores.append(mlp.score(X_test, y_test))

        print('%.2f' % float(i*100/epoch), '%')
    #print("Training set score: %f" % mlp.score(X_train, y_train))
    #print("Training set loss: %f" % mlp.loss_)

    plt.plot(train_scores, **plot_arg[0])
    plt.plot(test_scores, **plot_arg[1])
    plt.gca().set_ylim([0, 1])
    plt.title("hidden : " + str(hidden) + " learning : " + str(learing_late))
    
    fig = plt.gcf()
    plt.show()
    fig.savefig(str(hidden) + "_" + str(learing_late) + ".png")
    plt.clf()
    

# load / generate some toy datasets
df = pd.read_csv('C:\\Users\\parkchangho\\Documents\\Git\\-beta-lactam-antibiotics\\dataset\\bl112.txt', delimiter=',')
columns = df.columns[0:-1]
X = np.array(df[columns])
X = X.astype(int)

y = np.array(df[['Y']])
y = y.astype(int)

data = (X, y)

HL_set = []
for i in range(10):
    HL_set.append(makeHLRandom())
    
LR_set = []
rate = 0.01
for i in range(10):
    LR_set.append(rate)
    rate /= 2
    
for i in range(10):
    for j in range(10):
        plot_on_dataset(*data, HL_set[i], LR_set[j])