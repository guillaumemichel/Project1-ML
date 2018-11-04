import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = x.T
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def various_features(input_data, curr_col):
    new_matrix = np.zeros((len(curr_col), 7))
    new_matrix[:, 0] = np.sin(curr_col)
    new_matrix[:, 1] = np.cos(curr_col)
    new_matrix[:, 2] = 1.0/np.sqrt(1.0+abs(curr_col))
    new_matrix[:, 3] = 1.0/(1.0+(curr_col)**2)
    new_matrix[:, 4] = np.exp(curr_col)
    new_matrix[:, 5] = np.tan(curr_col)
    new_matrix[:, 6] = sigmoid(curr_col)
    input_data = np.hstack((input_data, new_matrix))
    return input_data

def multiply_columns(input_data, col1, col2):
    new_col = input_data[:, col1] * input_data[:, col2]
    new_matrix = np.zeros((len(new_col), 5))
    new_matrix[:, 0] = new_col
    new_matrix[:, 1] = np.sin(new_col)
    new_matrix[:, 2] = np.cos(new_col)
    new_matrix[:, 3] = 1.0/np.sqrt(1.0+abs(new_col))
    new_matrix[:, 4] = 1.0/(1.0+(new_col)**2)
    input_data = np.hstack((input_data, new_matrix))
    return input_data

def multiply_columns3(input_data, col1, col2, col3):
    new_col = input_data[:, col1] * input_data[:, col2] * input_data[:, col3]
    new_matrix = np.zeros((len(new_col), 4))
    new_matrix[:, 0] = new_col
    new_matrix[:, 1] = np.sin(new_col)
    new_matrix[:, 2] = np.cos(new_col)
    new_matrix[:, 3] = 1.0/np.sqrt(1.0+abs(new_col))
    input_data = np.hstack((input_data, new_matrix))
    return input_data

def add_features(input_data):
    """Construct new features, i.e. add basis functions"""

    #Indices of the column for which we build the polynomials
    degree_12 = [0, 1, 13]
    degree_8 = [3, 4, 5, 9, 10, 11, 12, 23]
    degree_5 = [2, 6, 11, 21, 22, 24, 25, 26, 27, 28, 29]
    degree_3 = [7, 8, 16, 19]
    degree_2 = [14, 15, 17, 18, 20]

    ###For each column, build the corresponding polynomial and craft other features###

    for i in range(len(degree_2)):
        curr_col = input_data[:, degree_2[i]]
        phi_temp = build_poly(curr_col, 2)
        input_data = np.hstack((input_data, phi_temp))
        input_data = various_features(input_data, curr_col)

    for i in range(len(degree_3)):
        curr_col = input_data[:, degree_3[i]]
        phi_temp = build_poly(curr_col, 3)
        input_data = np.hstack((input_data, phi_temp))
        input_data = various_features(input_data, curr_col)

    for i in range(len(degree_5)):
        curr_col = input_data[:, degree_5[i]]
        phi_temp = build_poly(curr_col, 5)
        input_data = np.hstack((input_data, phi_temp))
        input_data = various_features(input_data, curr_col)

    for i in range(len(degree_8)):
        curr_col = input_data[:, degree_8[i]]
        phi_temp = build_poly(curr_col, 8)
        input_data = np.hstack((input_data, phi_temp))
        input_data = various_features(input_data, curr_col)

    for i in range(len(degree_12)):
        curr_col = input_data[:, degree_12[i]]
        phi_temp = build_poly(curr_col, 12)
        input_data = np.hstack((input_data, phi_temp))
        input_data = various_features(input_data, curr_col)

    #ADD features to multiply column between them
    input_data = multiply_columns3(input_data, 0, 1, 13)
    input_data = multiply_columns3(input_data, 0, 3, 13)
    input_data = multiply_columns3(input_data, 0, 4, 13)
    input_data = multiply_columns3(input_data, 0, 5, 13)
    input_data = multiply_columns3(input_data, 0, 9, 13)
    input_data = multiply_columns3(input_data, 0, 10, 13)
    input_data = multiply_columns3(input_data, 0, 11, 13)
    input_data = multiply_columns3(input_data, 0, 12, 13)
    input_data = multiply_columns3(input_data, 0, 23, 13)

    input_data = multiply_columns3(input_data, 0, 1, 3)
    input_data = multiply_columns3(input_data, 0, 1, 4)
    input_data = multiply_columns3(input_data, 0, 1, 5)
    input_data = multiply_columns3(input_data, 0, 1, 9)
    input_data = multiply_columns3(input_data, 0, 1, 10)
    input_data = multiply_columns3(input_data, 0, 1, 11)
    input_data = multiply_columns3(input_data, 0, 1, 12)
    input_data = multiply_columns3(input_data, 0, 1, 23)

    input_data = multiply_columns3(input_data, 0, 11, 3)
    input_data = multiply_columns3(input_data, 0, 11, 4)
    input_data = multiply_columns3(input_data, 0, 11, 5)
    input_data = multiply_columns3(input_data, 0, 11, 9)
    input_data = multiply_columns3(input_data, 0, 11, 10)
    input_data = multiply_columns3(input_data, 0, 11, 12)
    input_data = multiply_columns3(input_data, 0, 11, 23)

    input_data = multiply_columns3(input_data, 1, 13, 3)
    input_data = multiply_columns3(input_data, 1, 13, 4)
    input_data = multiply_columns3(input_data, 1, 13, 5)
    input_data = multiply_columns3(input_data, 1, 13, 9)
    input_data = multiply_columns3(input_data, 1, 13, 10)
    input_data = multiply_columns3(input_data, 1, 13, 11)
    input_data = multiply_columns3(input_data, 1, 13, 12)
    input_data = multiply_columns3(input_data, 1, 13, 23)

    input_data = multiply_columns3(input_data, 1, 5, 3)
    input_data = multiply_columns3(input_data, 1, 5, 4)
    input_data = multiply_columns3(input_data, 1, 5, 9)
    input_data = multiply_columns3(input_data, 1, 5, 10)
    input_data = multiply_columns3(input_data, 1, 5, 11)
    input_data = multiply_columns3(input_data, 1, 5, 12)
    input_data = multiply_columns3(input_data, 1, 5, 23)

    input_data = multiply_columns(input_data, 0, 13)
    input_data = multiply_columns(input_data, 1, 13)
    input_data = multiply_columns(input_data, 0, 1)

    input_data = multiply_columns(input_data, 0, 3)
    input_data = multiply_columns(input_data, 0, 4)
    input_data = multiply_columns(input_data, 0, 5)
    input_data = multiply_columns(input_data, 0, 9)
    input_data = multiply_columns(input_data, 0, 10)
    input_data = multiply_columns(input_data, 0, 11)
    input_data = multiply_columns(input_data, 0, 12)
    input_data = multiply_columns(input_data, 0, 23)

    input_data = multiply_columns(input_data, 1, 3)
    input_data = multiply_columns(input_data, 1, 4)
    input_data = multiply_columns(input_data, 1, 5)
    input_data = multiply_columns(input_data, 1, 9)
    input_data = multiply_columns(input_data, 1, 10)
    input_data = multiply_columns(input_data, 1, 11)
    input_data = multiply_columns(input_data, 1, 12)
    input_data = multiply_columns(input_data, 1, 23)

    input_data = multiply_columns(input_data, 13, 3)
    input_data = multiply_columns(input_data, 13, 4)
    input_data = multiply_columns(input_data, 13, 5)
    input_data = multiply_columns(input_data, 13, 9)
    input_data = multiply_columns(input_data, 13, 10)
    input_data = multiply_columns(input_data, 13, 11)
    input_data = multiply_columns(input_data, 13, 12)
    input_data = multiply_columns(input_data, 13, 23)

    input_data = multiply_columns(input_data, 3, 4)
    input_data = multiply_columns(input_data, 3, 5)
    input_data = multiply_columns(input_data, 3, 9)
    input_data = multiply_columns(input_data, 3, 10)
    input_data = multiply_columns(input_data, 3, 11)
    input_data = multiply_columns(input_data, 3, 12)
    input_data = multiply_columns(input_data, 3, 23)

    input_data = multiply_columns(input_data, 4, 5)
    input_data = multiply_columns(input_data, 4, 9)
    input_data = multiply_columns(input_data, 4, 10)
    input_data = multiply_columns(input_data, 4, 11)
    input_data = multiply_columns(input_data, 4, 12)
    input_data = multiply_columns(input_data, 4, 23)

    input_data = multiply_columns(input_data, 5, 9)
    input_data = multiply_columns(input_data, 5, 10)
    input_data = multiply_columns(input_data, 5, 11)
    input_data = multiply_columns(input_data, 5, 12)
    input_data = multiply_columns(input_data, 5, 23)

    input_data = multiply_columns(input_data, 9, 10)
    input_data = multiply_columns(input_data, 9, 11)
    input_data = multiply_columns(input_data, 9, 12)
    input_data = multiply_columns(input_data, 9, 23)

    input_data = multiply_columns(input_data, 10, 11)
    input_data = multiply_columns(input_data, 10, 12)
    input_data = multiply_columns(input_data, 10, 23)

    input_data = multiply_columns(input_data, 11, 12)
    input_data = multiply_columns(input_data, 11, 23)

    input_data = multiply_columns(input_data, 12, 23)

    #remove unrelevant features
    input_data = np.delete(input_data, degree_2 + degree_3 + degree_5 + degree_8 + degree_12, axis=1)

    return input_data
