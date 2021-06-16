"""
    Gaussian Bivariate linear regression
    Dataset - Bivariate_Dataset.csv
    Bivariate input
    Implementing L2 regularisation as well as Tikhonov regularisation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split,KFold
from mpl_toolkits.mplot3d import Axes3D


class gaussian_regression:
    """
          A class used to calculate & plot the curve fitting of given dataset using Gaussian basis functions

          ...

          Methods
          -------
            read_data():
                Reads data and converts it into a numpy array

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

            plot3d(y_true, x_train, x_test,weights,means)
                Overlay - Scatter plot of bivariate input x1, x2 & y_true along with surface plot of x1,x2 & y_pred

            cross_validation()
                Performs k-Fold cross validation of dataset
    """

    def __init__(self,lambda_,D,regularization ,sigma, sample_size ):

        self.regularization = regularization
        self.D = D
        self.sigma = sigma
        self.lambda_ = lambda_
        self.random_sample_size = sample_size

    def read_data(self):
        data = pd.read_csv('function0_2d.csv')
        data = data.sample(self.random_sample_size).to_numpy()
        # X = data[:,1:3]
        # Y = data[:,-1]
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
        return data

    def design_matrix(self,X,means):

        design_matrix = [np.ones(len(X))]
        for mean in means:
            design_matrix.append(np.exp(-0.5 * np.linalg.norm(X - mean, axis=-1) ** 2 / self.sigma ** 2))
        # print(np.shape(design_matrix))
        return np.transpose(design_matrix)

    def calculate_weights(self, design_matrix, Y, means):

        if self.regularization == 'L2':
            pseudo_inv = np.dot(
                np.linalg.inv(
                    (np.dot(np.transpose(design_matrix), design_matrix)) + self.lambda_ * np.identity(n=self.D)),
                np.transpose(design_matrix))
            # L2 regularization
            return np.dot(pseudo_inv, Y)

        elif self.regularization == 'tikhonov':
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

    def train(self,X,Y):

        means = self.K_clustering(X)
        design_matrix = self.design_matrix(X,means)
        weights = self.calculate_weights(design_matrix,Y,means)
        return weights,design_matrix,means

    def make_prediction(self,X,weights,means):

        design_matrix = self.design_matrix(X,means)
        return np.dot(design_matrix,weights)

    def K_clustering(self,X):

        means = random.sample(list(X), self.D - 1)
        yn = np.zeros([len(X), self.D - 1, 2])
        zn = np.zeros([len(X), self.D - 1])
        for j in range(10):
            for i in range(len(X)):
                class_id = np.argmin([np.linalg.norm(X[i] - mean) for mean in means])
                zn[i][class_id] = 1
                yn[i][class_id] = X[i]

        means = np.sum(yn, axis=0) / np.expand_dims(np.sum(zn, axis=0), axis=-1)
        return means

    def plot(self, y_true, y_pred):
        plt.scatter(y_true, y_pred)
        plt.plot(np.linspace(np.min(y_true), np.max(y_true), 10), np.linspace(np.min(y_true), np.max(y_true), 10), 'r')
        plt.xlabel('Y TRUE')
        plt.ylabel('Y PREDICTED')
        plt.title('Lambda = ' + str(self.lambda_))
        plt.show()

    def plot3d(self, y_true, x_train, x_test,weights,means):
        x_surf, y_surf = np.meshgrid(np.linspace(x_test['x1'].min(), x_test['x1'].max(), 20),
                                     np.linspace(x_test['x2'].min(), x_test['x2'].max(), 20))
        onlyX = pd.DataFrame({'x1': x_surf.ravel(), 'x2': y_surf.ravel()})
        y_pred = self.make_prediction(onlyX.to_numpy(), weights, means)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], y_true, c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_surf, y_surf, y_pred.reshape(x_surf.shape), color='b', alpha=0.3)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        plt.show()

    def cross_validation(self):
        kfold = KFold(10, True, 1)
        data = self.read_data()
        errors = []
        for train, test in kfold.split(data):
            X_test, Y_test = data[test][:, 1:3], data[test][:, -1]
            X_train, Y_train = data[train][:, 1:3], data[train][:, -1]
            weights, design_matrix, means = self.train(X_train, Y_train)
            Dataframe_X_test = pd.DataFrame(X_test, columns=['x1', 'x2'])
            self.plot3d(y_true=Y_train,x_train=X_train,x_test=Dataframe_X_test,weights=weights,means=means)
            y_pred = self.make_prediction(X_test, weights, means)
            self.plot(y_true=Y_test, y_pred=y_pred)
            errors.append(np.mean(y_pred - Y_test) ** 2)
        print("%0.10f accuracy with a standard deviation of %0.10f across the k-folds" % (np.mean(errors), np.std(errors)))

"""
        Creates a class variable regressor
        Calls the function cross validation in the class gaussian_regression
"""
def main():
    regressor = gaussian_regression( D =10,regularization= 'L2' , lambda_= 0.01, sigma = 3.3,sample_size=2000)
    regressor.cross_validation()

if __name__ == '__main__':
    main()

















