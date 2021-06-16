"""
Multilayer Feed Forward Neural network with a single hidden layer of hidden classes
Dataset 1(a)
"""
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.colors import ListedColormap
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# dataset path on the local system
path = r'C:\PRML-2\dataset 1-a'
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF', '#FE74D5'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#4333FF', '#7C2C64'])


def read_data(path):
    """
        Function used to import data from the dataset
        :param path: Specified path to file containing dataset
        :return: train data, test data

    """
    train = pd.read_csv(path + '/' + 'train.csv', header=None)
    train.columns = ['x1', 'x2', 'y']
    train = train.astype({'x1': 'float64', 'x2': 'float64', 'y': 'int32'})
    test = pd.read_csv(path + '/' + 'dev.csv', header=None)
    test.columns = ['x1', 'x2', 'y']
    test = test.astype({'x1': 'float64', 'x2': 'float64', 'y': 'int32'})
    test.columns = ['x1', 'x2', 'y']
    return train, test


def show_classification_areas(X, Y):
    """
        Function used to plot the decision boundaries
        :param X: Input X data
        :param Y: Input Y data
        :return: No return value
    """

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(30, 25))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points along with Support vectors
    scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend(*scatter.legend_elements())
    plt.show()


def confusion_plot(y_true, y_pred):
    """
            Function to plot the confusion matrix
            :param y_true: Value of the true y value
            :param y_pred: Value of predicted value
            :return: NULL
    """
    array = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(array, ['Class 1', 'Class 2', 'Class3', 'Class4'], ['Class 1', 'Class 2', 'Class3', 'Class 4'])

    group_counts = ['{0:0.0f}'.format(value) for value in array.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in array.flatten() / np.sum(array)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(4, 4)
    ax = sn.heatmap(df_cm, annot=labels, cmap='Blues', fmt='', annot_kws={"fontsize": 10})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_title('Confusion Matrix for Test ', fontsize=20, pad=20)
    plt.show()


train, test = read_data(path)
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values
sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X)
X_testscaled = sc_X.transform(X_test)

"""
Cross-validation In order to find the best hyperparameters
"""
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


"""
define search for training dataset
Peforming Gridsearch across the hyperparameters
"""
grid = dict()
grid['hidden_layer_sizes'] = [1, 2, 5, 10, 25, 50, 100]
model = MLPClassifier(activation="tanh", random_state=1)
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
searchT = GridSearchCV(model, grid, scoring='accuracy', n_jobs=-1)
results = search.fit(X_trainscaled, y)
resultsT = searchT.fit(X_trainscaled, y)
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))

print('Mean Accuracy: %.3f' % resultsT.best_score_)
print('Config: %s' % resultsT.best_params_)
# summarize all
meansT = resultsT.cv_results_['mean_test_score']
paramsT = resultsT.cv_results_['params']
for mean, param in zip(meansT, paramsT):
    print(">%.3f with: %r" % (mean, param))


"""
Best Model
"""
clf = MLPClassifier(hidden_layer_sizes=(32), activation="tanh", random_state=1).fit(X_trainscaled, y)
yhat = clf.predict(X_testscaled)
# show_classification_areas(X_trainscaled,y)
# print(confusion_plot(y_true=y_test,y_pred=yhat))
# print(confusion_plot(y_true=y,y_pred=clf.predict(X_trainscaled)))
print(clf.score(X_trainscaled, y))
print(clf)
