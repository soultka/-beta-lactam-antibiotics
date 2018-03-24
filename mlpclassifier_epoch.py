import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# different learning rate schedules and momentum parameters
param = {'solver': 'adam', 'learning_rate_init': 0.01}
label = "adam"
plot_arg = [{'c': 'blue', 'linestyle': '-'}, {'c': 'red', 'linestyle': '-'}]
mlp = MLPClassifier(hidden_layer_sizes=(18, 175, 256), batch_size=528, learning_rate_init=0.0001, beta_1=0.001,
                        beta_2=0.001, max_iter=1, warm_start=True)

def plot_on_dataset(X, y, epoch):
    X = MinMaxScaler().fit_transform(X)
    train_scores = []
    test_scores = []

    '''
    # KFold
    kf = KFold(n_splits=5)

    train_index = [index for index, _ in kf.split(X)]
    test_index = [index for _, index in kf.split(X)]
    '''

    # Divide the data set into 8 to 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("training: %s" % label)
    for i in range(epoch):
        '''
        X_train, X_test = X[train_index[i%5]], X[test_index[i%5]]
        y_train, y_test = y[train_index[i%5]], y[test_index[i%5]]
        '''

        mlp.fit(X_train, y_train)
        train_scores.append(mlp.score(X_train, y_train))
        test_scores.append(mlp.score(X_test, y_test))
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Training set loss: %f" % mlp.loss_)

    plt.plot(train_scores, **plot_arg[0])
    plt.plot(test_scores, **plot_arg[1])

# load / generate some toy datasets
df = pd.read_csv('C:\\Users\\parkchangho\\Documents\\Git\\-beta-lactam-antibiotics\\bl112.txt', delimiter=',')
columns = df.columns[0:-1]
X = np.array(df[columns])
X = X.astype(int)

y = np.array(df[['Y']])
y = y.astype(int)

data = (X, y)
plot_on_dataset(*data, 5000)
plt.gca().set_ylim([0, 1])
plt.show()