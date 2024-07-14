"""
Created on Thu Jun 27 2024, 15:07:31 

@author: Amoy Ashesh
"""

import numpy as np

def lagrange_interpolation(x_train,y_train,x_test,n):
    """
    This function computes the Lagrange interpolation polynomial of degree n
    for a given set of training data points (x_train,y_train) and returns the
    interpolated values at the test points x_test.
    
    Parameters:
    -----------
    x_train : numpy array
        The training data points in the x-axis.
    y_train : numpy array
        The training data points in the y-axis.
    x_test : numpy array
        The test data points in the x-axis.
    n : int
        The degree of the Lagrange interpolation polynomial.
    
    Returns:
    --------
    y_test : numpy array
        The interpolated values at the test points x_test.
    """
    y_test = np.zeros(len(x_test))
    for i in range(len(x_test)):
        for j in range(n+1):
            L = 1
            for k in range(n+1):
                if k != j:
                    L *= (x_test[i] - x_train[k])/(x_train[j] - x_train[k])
            y_test[i] += y_train[j]*L
    return y_test

def newton_interpolation(x_train,y_train,x_test,n):
    """
    This function computes the Newton interpolation polynomial of degree n
    for a given set of training data points (x_train,y_train) and returns the
    interpolated values at the test points x_test.
    
    Parameters:
    -----------
    x_train : numpy array
        The training data points in the x-axis.
    y_train : numpy array
        The training data points in the y-axis.
    x_test : numpy array
        The test data points in the x-axis.
    n : int
        The degree of the Newton interpolation polynomial.
    
    Returns:
    --------
    y_test : numpy array
        The interpolated values at the test points x_test.
    """
    y_test = np.zeros(len(x_test))
    for i in range(len(x_test)):
        y_test[i] = y_train[0]
        for j in range(1,n+1):
            f = 0
            for k in range(j):
                p = 1
                for l in range(j):
                    if l != k:
                        p *= (x_test[i] - x_train[l])
                f += y_train[k]/p
            for k in range(j):
                f *= (x_test[i] - x_train[k])
            y_test[i] += f
    return y_test

def neville_interpolation(x_train,y_train,x_test):
    """
    This function computes the Neville interpolation polynomial for a given set
    of training data points (x_train,y_train) and returns the interpolated values
    at the test points x_test.
    
    Parameters:
    -----------
    x_train : numpy array
        The training data points in the x-axis.
    y_train : numpy array
        The training data points in the y-axis.
    x_test : numpy array
        The test data points in the x-axis.
    
    Returns:
    --------
    y_test : numpy array
        The interpolated values at the test points x_test.
    """
    n = len(x_train)
    y_test = np.zeros(len(x_test))
    for i in range(len(x_test)):
        Q = np.zeros((n,n))
        for j in range(n):
            Q[j][0] = y_train[j]
        for j in range(1,n):
            for k in range(1,j+1):
                Q[j][k] = ((x_test[i] - x_train[j-k])*Q[j][k-1] - (x_test[i] - x_train[j])*Q[j-1][k-1])/(x_train[j] - x_train[j-k])
        y_test[i] = Q[n-1][n-1]
    return y_test

def rational_interpolation(x_train,y_train,x_test):
    """
    This function computes the rational interpolation polynomial for a given set
    of training data points (x_train,y_train) and returns the interpolated values
    at the test points x_test.
    
    WARNING: This function is inaccurate and may produce erroneous results.
    ----------
    
    Parameters:
    -----------
    x_train : numpy array
        The training data points in the x-axis.
    y_train : numpy array
        The training data points in the y-axis.
    x_test : numpy array
        The test data points in the x-axis.
    
    Returns:
    --------
    y_test : numpy array
        The interpolated values at the test points x_test.
    """
    n = len(x_train)
    y_test = np.zeros(len(x_test))
    for k in range(len(x_test)):
        # Create differences table
        diff = np.zeros([n,n])
        for i in range(n):
            diff[i][0] = y_train[i]
        for j in range(1,n):
            for i in range(0,n-j):
                diff[i][j] = diff[i+1][j-1]-diff[i][j-1]
        # Create inverse differences table
        Inv_diff = np.zeros([n-1,n-1])
        for i in range(1,n):
            Inv_diff[i-1][0] = (x_train[i]-x_train[0])/(y_train[i]-y_train[0])
        for j in range(1,n):
            for i in range(0,n-j-1):
                Inv_diff[i][j] = (x_train[i+j+1]-x_train[j])/(Inv_diff[i+1][j-1]-Inv_diff[0][j-1])
        # Substitute values to obtain output
        Y = Inv_diff[0][-1]
        for i in range(n-1,-1,-1):
            Y = (x_test[k]-x_train[i])/Y + Inv_diff[0][i-1]
        y_test[k] = y_train[0]+((x_test[k]-x_train[0])/Y) + Y
    return y_test

def gregory_newton_interpolation(x_train,y_train,x_test):
    """
    This function computes the Gregory-Newton interpolation polynomial for a given set
    of training data points (x_train,y_train) and returns the interpolated values
    at the test points x_test.
    
    WARNING: This function is inaccurate and may produce erroneous results.
    ----------
    
    Parameters:
    -----------
    x_train : numpy array
        The training data points in the x-axis.
    y_train : numpy array
        The training data points in the y-axis.
    x_test : numpy array
        The test data points in the x-axis.
    
    Returns:
    --------
    y_test : numpy array
        The interpolated values at the test points x_test.
    """
    n = len(x_train)
    y_test = np.zeros(len(x_test))
    for k in range(len(x_test)):
        # Create differences table
        diff = np.zeros([n,n])
        for i in range(n):
            diff[i][0] = y_train[i]
        for j in range(1,n):
            for i in range(0,n-j):
                diff[i][j] = diff[i+1][j-1]-diff[i][j-1]
        # Compute the value of the polynomial at x_test[k]
        h = x_train[1]-x_train[0]
        s = (x_test[k]-x_train[0])/h
        y_test[k] = y_train[0]
        for i in range(1,n):
            p = 1
            for j in range(1,i+1):
                p *= (s-j+1)/j
            y_test[k] += p*diff[0][i]
    return y_test