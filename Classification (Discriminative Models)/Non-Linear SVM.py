"""
Nonlinear SVM using one-against-the-rest approach :
(a) Polynomial kernel, (b) Gaussian kernel
"""

from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib.markers import MarkerStyle
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# dataset path on the local system
path = r'C:\Users\hadit\PycharmProjects\PRML_ASSIGNMENT_3\Dataset_1B\4'
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF','#FE74D5'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#4333FF' , '#7C2C64'])
cmap_SV= ListedColormap(['#F8FF12','#FF0000','#F8FF12','#F8FF12'])

#importing the dataset
def read_data(path,i=1,j=2):
    """
            Function used to import data from the dataset
            :param path: Specified path to file containing dataset
            :return: train data, test data

    """
    train = pd.read_csv(path+ '/' +'train.csv',header=None)
    train.columns = ['x1','x2','y']
    train = train.astype({'x1':'float64','x2':'float64','y':'int32'})
    test = pd.read_csv(path + '/' + 'dev.csv',header=None)
    test.columns = ['x1', 'x2', 'y']
    test = test.astype({'x1': 'float64', 'x2': 'float64', 'y': 'int32'})
    test.columns = ['x1', 'x2', 'y']
    return train,test

def show_classification_areas(X,Y):
        """
            Function used to plot the decision boundaries
            :param X: Input X data
            :param Y: Input Y data
            :return: No return value
        """

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        Z = rbf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        print(Z.shape)
        print("Hello")
        print(xx.shape)
        plt.figure(1, figsize=(30, 25))
        plt.pcolormesh(xx, yy, Z,cmap=cmap_light)

        # Plot also the training points along with Support vectors
        scatter = plt.scatter(X[:, 0], X[:, 1], c= Y,cmap=cmap_bold)
        scatter = plt.scatter(rbf.support_vectors_[:,0],rbf.support_vectors_[:,1],s=100,
                linewidth=1, facecolors='none',c= rbf.predict(rbf.support_vectors_),cmap=cmap_SV )
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
        df_cm = pd.DataFrame(array, ['Class 1', 'Class 2','Class3'],['Class 1', 'Class 2','Class3'])
                           
        group_counts = ['{0:0.0f}'.format(value) for value in array.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in array.flatten() / np.sum(array)]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(3,3)
        ax = sn.heatmap(df_cm, annot=labels, cmap='Blues', fmt='', annot_kws={"fontsize": 10})
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
        ax.set_ylabel('True label', fontsize=10)
        ax.set_xlabel('Predicted label', fontsize=10)
        ax.set_title('Confusion Matrix for Model ', fontsize=20, pad=20)
        plt.show()

# Obtaining data
train, test = read_data(path)
X =train.iloc[:,:-1].values
y = train.iloc[:,-1].values
X_test= test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

#for polynomial kernel just change rbf to poly
rbf = svm.SVC(kernel='poly',degree=2, gamma=0.1, C=0.1, decision_function_shape='ovr')
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.15, random_state=42)
rbf.fit(X_train, y_train)
yhat= rbf.predict(X_test)

"""
Displaying the confusion plot & the classification areas
"""
show_classification_areas(X_train,y_train)
print(confusion_plot(y_true=y_test,y_pred=yhat))
print(rbf.score(X_test,y_test))
print(rbf.support_vectors_)

ytune = rbf.predict(X_train)
print(confusion_plot(y_true=y_train,y_pred=ytune))
print(classification_report(y_train, ytune))

ytune = rbf.predict(X_validation)
print(confusion_plot(y_true=y_validation,y_pred=ytune))
print(classification_report(y_validation, ytune))

"""
Performing Grid Search to find best parameters
"""
# param_grid = {'C': [0.1,1],
#               'gamma': [0.1, 0.01,1,10],
#               'degree': [1,2,3,5,15]}
#
# grid = GridSearchCV(svm.SVC(kernel = 'poly'), param_grid, refit=True, verbose=3)
#
# # fitting the model for grid search
# grid.fit(X_validation, y_validation)
#
# # print best parameter after tuning
# print(grid.best_params_)
#
# # print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)
#
# meansT = grid.cv_results_['mean_test_score']
# paramsT = grid.cv_results_['params']
# for mean, param in zip(meansT, paramsT):
#    print(">%.3f with: %r" % (mean, param))
#
# grid_predictions = grid.predict(X_test)
#
# # print classification report
# print(classification_report(y_test, grid_predictions))
