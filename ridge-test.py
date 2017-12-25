#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class RidgeRegressor(object):
    """
    Linear Least Squares Regression with Tikhonov regularization.
    More simply called Ridge Regression.

    We wish to fit our model so both the least squares residuals and L2 norm
    of the parameters are minimized.
    argmin Theta ||X*Theta - y||^2 + alpha * ||Theta||^2

    A closed form solution is available.
    Theta = (X'X + G'G)^-1 X'y

    Where X contains the independent variables, y the dependent variable and G
    is matrix alpha * I, where alpha is called the regularization parameter.
    When alpha=0 the regression is equivalent to ordinary least squares.

    http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)
    http://en.wikipedia.org/wiki/Tikhonov_regularization
    http://en.wikipedia.org/wiki/Ordinary_least_squares
    """

    def fit(self, X, y, alpha=0):
        """
        Fits our model to our training data.

        Arguments
        ----------
        X: mxn matrix of m examples with n independent variables
        y: dependent variable vector for m examples
        alpha: regularization parameter. A value of 0 will model using the
        ordinary least squares regression.
        """
        #X = np.hstack((np.ones((X.shape[0], 1)), X))
        G = np.eye(X.shape[1])
        G[0, 0] = 0  # Don't regularize bias
        self.params = np.dot(np.linalg.inv(np.dot(X.T, X) + (alpha *np.dot(G.T, G))),
                             np.dot(X.T, y))

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.
        The assumption here is that the new data is iid to the training data.

        Arguments
        ----------
        X: mxn matrix of m examples with n independent variables
        alpha: regularization parameter. Default of 0.

        Returns
        ----------
        Dependent variable vector for m examples
        """
        #X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.params)


if __name__ == '__main__':
    # Create synthetic data

    X = np.load('X_all_orient-k=15.np.npy')
    y = np.load ('YY-sigma1-6.npy')
    #xtest=np.load('Xflos-k=15.npy') 
    #V= np.load('Vtest15.npy')
    
    xtest=np.load('Xb=15.npy')
    #V= np.load('Vim-k=15.npy')
    # Plot synthetic data
    
    # Create feature matrix
    """tX = np.array([X]).T
    tX = np.hstack((tX, np.power(tX, 2), np.power(tX, 3)))
"""
    # Plot regressors
    r = RidgeRegressor()
    #r.fit(X, y)
    #print (r.params)
    print ("\n\n")
   # plt.plot(X, r.predict(X), 'b', label=u'ŷ (alpha=0.0)')
    alpha = 0.1
    r.fit(X, y, alpha)
    #plt.plot(X, r.predict(X), 'y', label=u'ŷ (alpha=%.01f)' % alpha)
    w_0 = r.params
    np.save('w_lamd0_kamen15.npy', w_0)
    print (w_0)
    print ("\n\n")
    w = r.predict(xtest)

"""
    ww =[]
    for i in range(len(V)):
	ww.append(np.dot(w_0.T, V[i]))
"""

print (np.sum(w))
    #print np.sum(ww), "la valeur de WW"
print ("\n\n")
print np.sum(y)


    #wflos = np.dot(xtest,w_0)
    #print (np.sum(wflos))
    #plt.legend()
    #plt.show()
