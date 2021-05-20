import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import pandas as pd
# Problem Statement
# ==================
# To begin with, use only the first 20 data points in your file.
# Solve the polynomial curve fitting regression problem using error function minimisation.
# Define your own error function other than the sum-of-squares error.
# Try different error formulations and report the results.
# Use a goodness-of-fit measure for polynomials of different order.
# Can you distinguish overfitting, underfitting, and the best fit?
# Obtain an estimate for the noise variance.
# Introduce regularisation and observe the changes. For quadratic regularisation, can you obtain an estimate of the optimal value for the regularisation parameter lamba? What is your corresponding best guess for the underlying polynomial? And the noise variance?
# Now repeat all of the above using the full data set of 100 data points.
# How are your results affected by adding more data? Comment on the differences.
# What is your final estimate of the underlying polynomial? Why?

# 10 - degree 6
# 15 - degree 7
# 20 - degree 8
# 100 - degree 9
def parse_input():
    global x, y
    filename = '..\data\\regression_data.txt'
    x, y = [], []
    training_count = 0
    reg_type = 0 # Valid values 0 & 1
    try:
        training_count = int(input('Enter input training count: 20 or 100  :'))
        print(training_count)
        if (training_count != 20 and training_count != 100):
            print("Training count should either be 20 or 100")
            sys.exit(1)
    except:
        sys.exit(1)

    try:
        reg_type = int(input('Please enter the type of regression which needs to be performed.'
                                   '\n 1. ''0'' for non-regularized  \n 2. ''1'' for regularized :'))
        print(reg_type)
        if (reg_type != 0 and reg_type != 1):
            print("Regression Type can either be 0 or 1")
            sys.exit(1)
    except:
        sys.exit(1)


    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            x.append(row[1])
            y.append(row[3])
    # Number of records in the file
    n = len(x)
    x = np.asarray(x, dtype=float).reshape((n, 1))
    y = np.asarray(y, dtype=float).reshape((n, 1))
    x = x[0:training_count, :]
    y = y[0:training_count, :]

    # Data Visualization
    plt.scatter(x, y, color="MediumVioletRed", alpha=0.4)
    plt.title("Data Visualization " + str(training_count) + ' data points')
    plt.xlabel("input variable x")
    plt.ylabel("target variable y")
    plt.show()
    return reg_type

# self implementation of k-fold
def kfold(k_idx, data_len):
    # Using 80-20 rule
    test_index_len = int(0.2*data_len)
    train_index_len = int(0.8*data_len)
    test_index = np.array([], dtype=np.int)
    total_idx = np.arange(0,data_len)
    test_idx= np.arange(0, data_len, test_index_len)
    curr_test_idx = test_idx[k_idx]
    test_index = np.arange(curr_test_idx, curr_test_idx + test_index_len)
    train_index = np.array(list(set(total_idx.tolist()) - set(test_index.tolist())))
    train_index = train_index.astype(int)
    return train_index, test_index


# Find the optimal degree for the polynomial
def find_poly_degree(reg_type):
    global deg, poly_reg, x_train, y_train, polyfit_x, reg_method
    if reg_type == 0:
        # For normal regression regularizer is 0 and set to one iteration
        lambda_range = [0]
    else:
        lambda_range = [1e-2, 1e-1, 1, 10, 1e2]
    lambda_len = len(lambda_range)
    degree_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    deg, best_gof, reg_lambda = 0, 0, 0
    k = 5
    k_range = [0,1,2,3,4]
    data_len = len(x)
    r2_gof_train, mse_mat_train, mae_mat_train, r2_gof_test, mse_mat_test, mae_mat_test = np.zeros((11, lambda_len)), np.zeros((11, lambda_len)), np.zeros((11, lambda_len)),np.zeros((11, lambda_len)), np.zeros((11, lambda_len)), np.zeros((11, lambda_len))
    for poly_degree in degree_range:
        poly_reg = PolynomialFeatures(degree=poly_degree)
        lambda_idx = 0

        for reg_lambda in lambda_range:
            mse_train_error_arr, mse_test_error_arr, mae_train_error_arr, mae_test_error_arr, train_fit_arr, test_fit_arr = np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(k)
            for k_idx in k_range:
                train_index, test_index = kfold(k_idx, data_len)
                x_train, x_valid = x[train_index], x[test_index]
                y_train, y_valid = y[train_index], y[test_index]

                reg_method = Ridge(alpha=reg_lambda)
                polyfit_x = poly_reg.fit_transform(x_train)
                poly_reg.fit(polyfit_x, y_train)
                reg_method.fit(polyfit_x, y_train)

                # train_error
                mse_train_error_arr[k_idx],mae_train_error_arr[k_idx], train_fit_arr[k_idx] = training_metrics(polyfit_x, reg_method, y_train)
                mse_test_error_arr[k_idx],  mae_test_error_arr[k_idx], test_fit_arr[k_idx] = training_metrics(poly_reg.fit_transform(x_valid), reg_method, y_valid)
            # Average over k folds

            mse_mat_train[poly_degree, lambda_idx], mse_mat_test[poly_degree, lambda_idx], mae_mat_train[poly_degree, lambda_idx], mae_mat_test[poly_degree, lambda_idx], r2_gof_train[poly_degree, lambda_idx], r2_gof_test[poly_degree, lambda_idx]\
                = np.average(mse_train_error_arr), np.average(mse_test_error_arr), np.average(mae_train_error_arr), np.average(mae_test_error_arr), np.average(train_fit_arr), np.average(test_fit_arr)

            if (r2_gof_test[poly_degree, lambda_idx] > best_gof):
                best_gof = r2_gof_test[poly_degree, lambda_idx]
                deg = poly_degree
                opt_reg_lambda = reg_lambda
            lambda_idx = lambda_idx + 1
    print('Optimal Degree ', deg, 'Optimal Lambda ', opt_reg_lambda)

    #Identifying fit of the model
    print("MSE ", mse_mat_train, mse_mat_test )
    if(reg_type == 0):
        plt.plot(np.log10(mse_mat_train[:,:]), color='LightSlateGray', label='Train Error')
        plt.plot(np.log10(mse_mat_test[:,:]), color='MediumVioletRed', label='Test Error')
        plt.xlabel('Degree')
    else:
        plt.plot(np.log10(mse_mat_train[deg, :]), color='LightSlateGray', label='Train Error')
        plt.plot(np.log10(mse_mat_test[deg, :]), color='MediumVioletRed', label='Test Error')
        plt.xlabel('Regularizer Lambda Index in Range [1e-2, 1e-1, 1, 10, 100]')
    plt.ylabel('Error')
    plt.legend(loc="upper right")
    plt.show()
    print("MAE ", mae_mat_train, mae_mat_test)
    print("GOF ", r2_gof_test)
    return deg, opt_reg_lambda


def training_metrics(x, reg_method, y):
    mse_error = mean_squared_error(y, reg_method.predict(x))
    mae_error = mean_absolute_error(y, reg_method.predict(x))
    gof_fit = r2_score(y, reg_method.predict(x))
    return mse_error, mae_error, gof_fit


#  Fit Model to Data
def fit_model(deg, reg_lambda):
    global poly_reg, reg_method, x, y
    # Fit Regression Model
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(x)
    poly_reg.fit(X_poly, y)
    reg_method = Ridge(alpha=reg_lambda)
    reg_method.fit(X_poly, y)
    print("Coefficient(s)")
    print(reg_method.coef_)
    print("Intercept")
    print(reg_method.intercept_)
    print("Variance")
    print(np.var(y - reg_method.predict(poly_reg.fit_transform(x))))
    # visualising the data prediction results -- do we have to?
    plt.scatter(x, y, color='MediumVioletRed', alpha=0.4, label='Data')
    X_grid = np.arange(min(x), max(x), 0.001)
    X_grid = X_grid.reshape(len(X_grid), 1)
    x = X_grid
    y = reg_method.predict(poly_reg.fit_transform(X_grid))
    [x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
    plt.plot(x, y, color='LightSlateGray', label='Model')
    plt.xlabel("X ")
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    reg_type = parse_input()
    deg, reg_lambda = find_poly_degree(reg_type)
    fit_model(deg, reg_lambda)
