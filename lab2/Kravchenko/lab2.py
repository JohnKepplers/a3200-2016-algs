import numpy as np


def linear_regression(x, y):
    m = x.shape[0]
    matrix = np.array(m, 2)
    for j in range(m):
        matrix[j, 0] = 1
    matrix[:, 1] = x
    return my_function(matrix, y)


def polynomial_regression(x, y, k):
    m = x.shape[0]
    matrix = np.array(m, k + 1)
    for j in range(k):
        matrix[:, k] = x ** k
    return my_function(matrix, y)


def functional_regression(x, y, functions):
    k = len(functions)
    m = x.shape[0]
    matrix = np.array(m, k + 1)
    for j in range(m):
        matrix[j, 0] = 1
    for i in range(1, k):
        matrix[:, k] = functions[k](x)
    return my_function(matrix, y)


def my_function(x, y):
    b = (x.T * x).I * x.T * y
    return b
