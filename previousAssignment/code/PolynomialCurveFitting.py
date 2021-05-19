import numpy as np
import matplotlib.pyplot as plt 
import csv
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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

#Find the optimal degree for the polynomial
def find_poly_degree(reg_type):
    global deg, poly_reg, x_train, y_train, X_poly, reg_method
    if reg_type == 0:
        # For normal regression regularizer is 0 and set to one iteration
        lambda_range = [0]
    else:
        lambda_range = [1e-2, 1e-1, 1, 10, 1e2]
    lambda_len = len(lambda_range)
    degree_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    deg, max_gof, reg_lambda = 0, 0, 0
    k = 5
    r2_gof_train, mse_mat_train, mae_mat_train, r2_gof_test, mse_mat_test, mae_mat_test = np.zeros((11, lambda_len)), np.zeros((11, lambda_len)), np.zeros((11, lambda_len)),np.zeros((11, lambda_len)), np.zeros((11, lambda_len)), np.zeros((11, lambda_len))
    for poly_degree in degree_range:
        poly_reg = PolynomialFeatures(degree=poly_degree)
        lambda_idx = 0
        for reg_lambda in lambda_range:
            kf = KFold(n_splits=k, random_state=None, shuffle=False)
            mse_train_error_arr, mse_test_error_arr, mae_train_error_arr, mae_test_error_arr, train_fit_arr, test_fit_arr = np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(k)
            k_idx = 0
            for train_index, test_index in kf.split(x):
                x_train, x_valid = x[train_index], x[test_index]
                y_train, y_valid = y[train_index], y[test_index]
                X_poly = poly_reg.fit_transform(x_train)
                poly_reg.fit(X_poly, y_train)
                reg_method = Ridge(alpha=reg_lambda)
                reg_method.fit(X_poly, y_train)
                # Mean Square Error
                mse_train_error_arr[k_idx] = (mean_squared_error(y_train, reg_method.predict(X_poly)))
                mse_test_error_arr[k_idx] = (mean_squared_error(y_valid, reg_method.predict(poly_reg.fit_transform(x_valid))))
                # Mean Absolute Error
                mae_train_error_arr[k_idx] = (mean_absolute_error(y_train, reg_method.predict(X_poly)))
                mae_test_error_arr[k_idx] = (mean_absolute_error(y_valid, reg_method.predict(poly_reg.fit_transform(x_valid))))
                # Goodness of Fit
                train_fit_arr[k_idx] = (r2_score(y_train, reg_method.predict(X_poly)))
                test_fit_arr[k_idx] = (r2_score(y_valid, reg_method.predict(poly_reg.fit_transform(x_valid))))

                k_idx = k_idx + 1
            # Average over k folds
            mse_mat_train[poly_degree, lambda_idx] = np.average(mse_train_error_arr)
            mse_mat_test[poly_degree, lambda_idx] = np.average(mse_test_error_arr)
            mae_mat_train[poly_degree, lambda_idx] = np.average(mae_train_error_arr)
            mae_mat_test[poly_degree, lambda_idx] = np.average(mae_test_error_arr)
            r2_gof_train[poly_degree, lambda_idx] = np.average(train_fit_arr)
            r2_gof_test[poly_degree, lambda_idx] = np.average(test_fit_arr)
            if (r2_gof_test[poly_degree, lambda_idx] > max_gof):
                max_gof = r2_gof_test[poly_degree, lambda_idx]
                deg = poly_degree
                opt_reg_lambda = reg_lambda
            lambda_idx = lambda_idx + 1
    print('Optimal Degree ', deg, 'Optimal Lamda ', opt_reg_lambda)

    #Identifying fit of the model
    print("MSE ", mse_mat_train, mse_mat_test )
    if(reg_type == 0):
        plt.plot(np.log10(mse_mat_train[:,:]), color='LightSlateGray', label='Train Error')
        plt.plot(np.log10(mse_mat_test[:,:]), color='MediumVioletRed', label='Test Error')
    else:
        plt.plot(np.log10(mse_mat_train[deg, :]), color='LightSlateGray', label='Train Error')
        plt.plot(np.log10(mse_mat_test[deg, :]), color='MediumVioletRed', label='Test Error')
    if(reg_type ==0):
        plt.xlabel('Degree')
    else:
        plt.xlabel('Lambda(Regularization)')
    plt.ylabel('Error')
    plt.legend(loc="upper right")
    plt.show()
    # print("MAE ", m_abs_err_mat_train, m_abs_err_mat_test)
    print("GOF ", r2_gof_test)
    return deg, opt_reg_lambda

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
    # visualising the data prediction results
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
