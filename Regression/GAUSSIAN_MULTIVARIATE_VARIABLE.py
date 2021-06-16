"""
    Gaussian Multivariate linear regression
    Dataset - Multivariate_Real_World_Dataset.csv
    Multivariate input
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import VarianceThreshold


class gaussian_regression:
    """
              A class used to calculate & plot the curve fitting of given dataset using Gaussian basis functions

              Methods
              -------
                process_data(data):
                    pre processes the data before training
                design_matrix(X):
                    Computes the design matrix
                calculate_weights(design_matrix,Y):
                    Calculates the weights of the Gaussian basis functions and performs the specified regularisation
                train(X,Y):
                    Calls the design_matrix & calculate_weights functions
                make_prediction(X,weights):
                    Makes the resultant prediction of Y for regression using weights and input X
                K_clustering(X):
                    Splitting input data x1,x2 into K clusters by finding the means of those clusters
                plot(y_pred,y_true, x):
                    Scatter plot of True y_test value vs Predicted y_test value
            """

    def __init__(self, lambda_, D, regularization, sigma):

        self.regularization = regularization
        self.D = D
        self.sigma = sigma
        self.lambda_ = lambda_
        self.k = 5

    def process_data(self, data):

        # Dropping rows which have NA data
        data = data.dropna()
        X = data.drop(['Next_Tmax', 'Next_Tmin'], axis=1)
        Y = data[['Next_Tmax', 'Next_Tmin']]

        variance_filter = VarianceThreshold(threshold=0.1)
        variance_filter.fit(X)

        quassi_static_columns = [column for column in X.columns
                                 if column not in X.columns[variance_filter.get_support()]]

        print(variance_filter.get_support())
        X.drop(labels=quassi_static_columns, axis=1, inplace=True)

        # low variance flter
        k = self.k

        X = X.to_numpy()
        Y = Y.to_numpy()
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))

        #   for i in range(Y.shape[1]):
        #     Y[:, i] = (Y[:, i] - np.min(Y[:, i])) / (np.max(Y[:, i]) - np.min(Y[:, i]))
        #     print(np.std(Y[:,i]))

        # normalizing the columns
        print('SVD calculation.....')
        u, s, vh = np.linalg.svd(X, full_matrices=True)
        sigma = np.zeros([X.shape[0], X.shape[1]])
        for i in range(X.shape[1]):
            sigma[i][i] = s[i]
        X = np.dot(u[:, :k], sigma[:k, :k])

        # SVD
        np.save(file='PC_vals_q3_K=' + str(k), arr=X)
        np.save(file='output_q3', arr=Y)
        return np.hstack([X, Y])

    def calculate_weights(self, design_matrix, Y, means):

        if self.regularization == 'L2':
            pseudo_inv = np.dot(
                np.linalg.inv(
                    (np.dot(np.transpose(design_matrix), design_matrix)) + self.lambda_ * np.identity(n=self.D)),
                np.transpose(design_matrix))
            # L2 regularization
            return np.dot(pseudo_inv, Y)

        if self.regularization == 'tikhonov':
            phi = np.zeros([self.D, self.D])
            for i in range(1, self.D):
                for j in range(1, self.D):
                    phi[i][j] = np.exp(-0.5 * np.linalg.norm((means[i - 1] - means[j - 1])) ** 2 / self.sigma ** 2)

            pseudo_inv = np.dot(
                np.linalg.inv(
                    (np.dot(np.transpose(design_matrix), design_matrix)) + self.lambda_ * phi),
                np.transpose(design_matrix))
            print('-----tikhonov-----')
            # tikhononv regularization
            return np.dot(pseudo_inv, Y)

        else:
            pseudo_inv = np.dot(
                np.linalg.inv((np.dot(np.transpose(design_matrix), design_matrix))),
                np.transpose(design_matrix))

            return np.dot(pseudo_inv, Y)

    def train(self, X, Y):

        means = self.K_clustering(X)
        design_matrix = self.design_matrix(X, means)
        weights = self.calculate_weights(design_matrix, Y, means)
        return weights, design_matrix, means

    def K_clustering(self, X):

        means = random.sample(list(X), self.D - 1)
        yn = np.zeros([len(X), self.D - 1, self.k])
        zn = np.zeros([len(X), self.D - 1])
        for j in range(20):
            for i in range(len(X)):
                class_id = np.argmin([np.linalg.norm(X[i] - mean) for mean in means])
                zn[i][class_id] = 1
                yn[i][class_id] = X[i]

        means = np.sum(yn + float(1e-10), axis=0) / np.expand_dims(np.sum(zn + float(1e-10), axis=0), axis=-1)
        return means

    def design_matrix(self, X, means):

        design_matrix = [np.ones(len(X))]
        for mean in means:
            design_matrix.append(np.exp(-0.5 * np.linalg.norm(X - mean, axis=-1) ** 2 / self.sigma ** 2))
        # print(np.shape(design_matrix))
        return np.transpose(design_matrix)

    def make_prediction(self, X, weights, means):

        design_matrix = self.design_matrix(X, means)
        return np.dot(design_matrix, weights)

    def plot(self, y_true, y_pred):

        # Plot 1 - With Y1 vs Y_predicted
        x = y_true[:, 0]
        y = y_pred[:, 0]
        plt.scatter(x, y)
        plt.plot(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(x), np.max(x), 10), 'r')
        plt.xlabel('Y1')
        plt.ylabel('Y PREDICTED')
        plt.title('Lambda = ' + str(self.lambda_) + ', D = ' + str(self.D))
        plt.show()

        # Plot 2 - With Y2 vs Y_predicted
        x = y_true[:, 1]
        y = y_pred[:, 1]
        plt.scatter(x, y)
        plt.plot(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(x), np.max(x), 10), 'r')
        plt.xlabel('Y2')
        plt.ylabel('Y PREDICTED')
        plt.title('Lambda = ' + str(self.lambda_) + ', D = ' + str(self.D))
        plt.show()

"""
        Creates a class variable regressor
        Performs cross validation
"""
def main():
    regressor = gaussian_regression(D=15, regularization='L2', lambda_=0.1, sigma=10)

    # Reading data
    data = pd.read_csv('Multivariate_Real_World_Dataset.csv')
    data = regressor.process_data(data)

    # Kfold cross validation
    kfold = KFold(8, shuffle=True, random_state=1)
    errors = []

    # Counter
    c = 0

    for train, test in kfold.split(data):
        X_test, Y_test = data[test][:, :-2], data[test][:, -2:]
        X_train, Y_train = data[train][:, :-2], data[train][:, -2:]
        weights, design_matrix, means = regressor.train(X_train, Y_train)
        y_pred = regressor.make_prediction(X_test, weights, means)
        regressor.plot(y_true=Y_test, y_pred=y_pred)
        errors.append(np.mean(y_pred - Y_test) ** 2)
        c += 1
        # To keep track of kth fold
        print(c)

    # Output Cross validation parameter over k-folds
    print("%0.10f accuracy with a standard deviation of %0.10f across the k-folds" % (np.mean(errors), np.std(errors)))
