"""
    Polynomial Curve Fitting
    Dataset - Univariate_Dataset.csv
    Single variable input
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split,KFold

class polynomial_regression:
    """
      A class used to calculate & plot the polynomial curve fitting of given dataset

      Methods
      -------
        read_data():
            Reads data and converts it into a numpy array

        design_matrix(X):
            Computes the design matrix

        calculate_weights(design_matrix,Y):
            Calculates the weights of the polynomial basis functions and performs the specified regularisation

        train(X,Y):
            Calls the design_matrix & calculate_weights functions

        make_prediction(X,weights):
            Makes the resultant prediction of Y for regression using weights and input X

        plot(y_pred,y_true, x):
            Scatter plot of True y_test value vs Predicted y_test value
            Overlay- Scatter plot of true y_test value vs x_test value, Line plot of  y_pred vs x_test

        cross_validation()
            Performs k-Fold cross validation of dataset
    """

    def __init__(self, lambda_, D, sample_size, regularization=None):
        """
      Attributes
      ----------
            D : int
                An integer representing the degree of complexity of regression

            regularization : str
                a string that determines the type of regularization

            lambda_ : float
                A floating point integer hyperparameter used to reduce overfitting

            sample_size : int
                An integer used to specify the number of data samples to be taken to perform regressionParameters
        """
        self.regularization = regularization
        self.D = D
        self.lambda_ = lambda_
        self.random_sample_size = sample_size

    def read_data(self):
        data = pd.read_csv('function0.csv')
        data = data.sample(self.random_sample_size).to_numpy()
        return data

    def design_matrix(self, X):
        """
        Attributes
        ----------
            X : float array
                    input training data
        """
        design_matrix = [X ** i for i in range(self.D)]
        return np.transpose(design_matrix)

    def calculate_weights(self,design_matrix,Y):
        """
        Attributes
        ----------
            design_matrix : float array
                    design_matrix as computed by float array
            Y : float array
                    Output training data
        """
        # L2 Regularisation #
        if self.regularization == 'L2':
            pseudo_inv = np.dot(
                np.linalg.inv((np.dot(np.transpose(design_matrix), design_matrix)) + self.lambda_ * np.identity(n=self.D)),
                np.transpose(design_matrix))
            return np.dot(pseudo_inv, Y)

        # No Regularisation #
        else:
            pseudo_inv = np.dot(
                np.linalg.inv((np.dot(np.transpose(design_matrix), design_matrix))),
                np.transpose(design_matrix))

            return np.dot(pseudo_inv, Y)

    def train(self,X,Y):
        """
        Attributes
        ----------
            X : float array
                       Input training data
            Y : float array
                       Output training data
        """
        design_matrix = self.design_matrix(X)
        weights = self.calculate_weights(design_matrix,Y)
        return weights,design_matrix

    def make_prediction(self,X,weights):
        """
        Attributes
        ----------
            X : float array
                        Input training data
            weights : float array
                        weights of the basis functions
        """
        design_matrix = self.design_matrix(X)
        return np.dot(design_matrix,weights)

    def plot(self, y_true, y_pred, x):
        """
        Attributes
        ----------
            y_true : float array
                        Output test data
            y_pred : float array
                        predicted output
            x   : float array
                        Input test data
        """
        # Scatter plot of y_true vs y_pred
        plt.scatter(y_true, y_pred)
        plt.plot(np.linspace(np.min(y_true), np.max(y_true), 10), np.linspace(np.min(y_true), np.max(y_true), 10), 'r') #Line to compare with the expected prediction
        plt.xlabel('Y TRUE')
        plt.ylabel('Y PREDICTED')
        plt.title('Lambda = ' + str(self.lambda_))
        plt.show()

        # Scatter plot of x_test vs y_pred & plot of x_test vs y_test
        x = np.reshape(x, newshape=[-1])
        order = np.argsort(x)
        plt.scatter(x=x, y=y_true)
        plt.plot(x[order], y_pred[order], 'red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PREDICTION \n Lambda = ' + str(self.lambda_))
        plt.legend(['prediction', 'data-points'])
        plt.show()

    def cross_validation(self):
        """
            Kfold cross validation - We split dataset into K parts and train on K-1 parts and validate on the left out part
        """
        kfold = KFold(10, shuffle=True, random_state=1)
        data = self.read_data()
        # error from each kth iteration
        errors = []
        for train, test in kfold.split(data):

            #Splitting into test and training data
            X_test, Y_test = data[test][:, 1], data[test][:, 2]
            X_train, Y_train = data[train][:, 1], data[train][:, 2]

            #Training on the split data
            weights, design_matrix = self.train(X_train, Y_train)

            y_pred = self.make_prediction(X_test, weights)
            self.plot(y_true=Y_test, y_pred=y_pred, x=X_test)

            #error matrix
            errors.append(np.mean(y_pred - Y_test) ** 2)

        #cross-validation parameter taken as mean of errors obtained from each iteration
        print("%0.10f mean with a standard deviation of %0.10f across the k-folds" % (np.mean(errors), np.std(errors)))


"""
        Creates a class variable regressor
        Calls the function cross validation in the class polynomial_regression
"""
def main():
    regressor = polynomial_regression(D=3, regularization='L2', lambda_=0.1, sample_size=900)
    regressor.cross_validation()

if __name__ == '__main__':
    main()








