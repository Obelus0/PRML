"""
Perceptron for every pair of classes
Dataset - Dataset 1(a)
"""

import sklearn
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV

# dataset path on the local system
path = r'C:\PRML-2\dataset1-a'
cmap_light = ListedColormap(['#00AAFF', '#FE74D5'])
cmap_bold = ListedColormap(['#4333FF', '#7C2C64'])


def read_data(path, i=0, j=1):
    """
    Function used to import data from the dataset
    :param path: Specified path to file containing dataset
    :return: train data, test data

    """
    train = pd.read_csv(path + '/' + 'train.csv', header=None)
    train.columns = ['x1', 'x2', 'y']
    train = train.astype({'x1': 'float64', 'x2': 'float64', 'y': 'int32'})
    train = train[(train.y == i) | (train.y == j)]
    test = pd.read_csv(path + '/' + 'dev.csv', header=None)
    test.columns = ['x1', 'x2', 'y']
    test = test.astype({'x1': 'float64', 'x2': 'float64', 'y': 'int32'})
    test.columns = ['x1', 'x2', 'y']
    test = test[(test.y == i) | (test.y == j)]
    return train, test

def show_classification_areas(X, Y,model):
    """
    Function used to plot the decision boundaries
    :param X: Input X data
    :param Y: Input Y data
    :return: No return value
    """

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(30, 25))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
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
    df_cm = pd.DataFrame(array, ['Class 1', 'Class 2'], ['Class 1', 'Class 2'])

    group_counts = ['{0:0.0f}'.format(value) for value in array.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in array.flatten() / np.sum(array)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax = sn.heatmap(df_cm, annot=labels, cmap='Blues', fmt='', annot_kws={"fontsize": 10})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_title('Confusion Matrix for Model ', fontsize=20, pad=20)
    plt.show()


def main():
    train, test = read_data(path, 2, 3)

    """
    To split data into X & y
    """
    X = train.iloc[:, :-1].values
    y = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    """
    Cross-validation In order to find the best hyperparameters
    """
    # Creating n-splits and 3 repeats for cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    """
    Finding accuracy using different values of eta and max iterations
    """
    # grid = dict()
    # grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    # define grid for max_iterations
    # grid = dict()
    # grid['max_iter'] = [7,10,100,1000, 10000]

    # define model
    model = Perceptron(eta0=0.001, max_iter=100)

    """
    define search for training dataset
    Peforming Gridsearch across the hyperparameters
    """
    # searchTrain=GridSearchCV(model,grid,scoring='accuracy')
    # resultTrain=searchTrain.fit(X,y)
    # define search for validation dataset
    # search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    # results = search.fit(X, y)

    # fit model
    clf = model.fit(X, y)

    # show results
    show_classification_areas(X, y,model)
    print(confusion_plot(y_true=y_test, y_pred=clf.predict(X_test)))
    print(confusion_plot(y_true=y, y_pred=clf.predict(X)))

    # summarize for Validation Data
    # print('Best Accuracy: %.3f' % results.best_score_)
    # print('Config: %s' % results.best_params_)

    # summarize all for Validation Data
    # means = results.cv_results_['mean_test_score']
    # params = results.cv_results_['params']
    # for mean, param in zip(means, params):
    #    print(">%.3f with: %r" % (mean, param))

    # summarize for Training Data
    # print('Best Accuracy for training data: %.3f' % resultTrain.best_score_)
    # print('Config: %s' % resultTrain.best_params_)

    # summarize all Training Data
    # meansT = resultTrain.cv_results_['mean_test_score']
    # paramsT = resultTrain.cv_results_['params']
    # for mean, param in zip(meansT, paramsT):
    #    print(">%.3f with: %r" % (mean, param))
