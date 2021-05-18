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
                                   '\n 1. ''0'' for non-regularized &  \n 2. ''1'' for regularized :'))
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

    degree_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # degree_range = [7,8]
    deg, max_gof, reg_lambda = 0, 0, 0
    k = 5
    r2_goodness_of_fit, m_sqr_err_mat, m_abs_err_mat = np.zeros((11, 2)), np.zeros((11, 2)), np.zeros((11, 2))
    for i in degree_range:
        poly_reg = PolynomialFeatures(degree=i)
        j = 0
        for reg_lambda in lambda_range:
            avg_m_sqr_err_mat_train, avg_m_sqr_err_mat_test, avg_m_abs_err_mat_train, avg_m_abs_err_mat_test, avg_r2_gof_train, avg_r2_gof_test = 0, 0, 0, 0, 0, 0
            kf = KFold(n_splits=k, random_state=None, shuffle=False)
            for train_index, test_index in kf.split(x):
                x_train, x_valid = x[train_index], x[test_index]
                y_train, y_valid = y[train_index], y[test_index]
                X_poly = poly_reg.fit_transform(x_train)
                poly_reg.fit(X_poly, y_train)
                reg_method = Ridge(alpha=reg_lambda)
                reg_method.fit(X_poly, y_train)

                train_error_poly_1 = (mean_squared_error(y_train, reg_method.predict(X_poly)))
                test_error_poly_1 = (mean_squared_error(y_valid, reg_method.predict(poly_reg.fit_transform(x_valid))))
                m_sqr_err_mat[i, 0] = train_error_poly_1
                m_sqr_err_mat[i, 1] = test_error_poly_1

                avg_m_sqr_err_mat_train = avg_m_sqr_err_mat_train + m_sqr_err_mat[i, 0]
                avg_m_sqr_err_mat_test = avg_m_sqr_err_mat_test + m_sqr_err_mat[i, 1]

                train_error_poly_2 = (mean_absolute_error(y_train, reg_method.predict(X_poly)))
                test_error_poly_2 = (mean_absolute_error(y_valid, reg_method.predict(poly_reg.fit_transform(x_valid))))
                m_abs_err_mat[i, 0] = train_error_poly_2
                m_abs_err_mat[i, 1] = test_error_poly_2

                avg_m_abs_err_mat_train = avg_m_abs_err_mat_train + m_abs_err_mat[i, 0]
                avg_m_abs_err_mat_test = avg_m_abs_err_mat_test + m_abs_err_mat[i, 1]

                train_fit = (r2_score(y_train, reg_method.predict(X_poly)))
                test_fit = (r2_score(y_valid, reg_method.predict(poly_reg.fit_transform(x_valid))))
                r2_goodness_of_fit[i, 0] = train_fit
                r2_goodness_of_fit[i, 1] = test_fit

                avg_r2_gof_train = avg_r2_gof_train + r2_goodness_of_fit[i, 0]
                avg_r2_gof_test = avg_r2_gof_test + r2_goodness_of_fit[i, 1]
            # print(i)
            # print(reg_lambda)
            m_sqr_err_mat[i, 0] = avg_m_sqr_err_mat_train / k
            # print(m_sqr_err_mat[i, 0])
            m_sqr_err_mat[i, 1] = avg_m_sqr_err_mat_test / k
            # print(m_sqr_err_mat[i, 1])

            m_abs_err_mat[i, 0] = avg_m_abs_err_mat_train / k
            m_abs_err_mat[i, 1] = avg_m_abs_err_mat_test / k

            r2_goodness_of_fit[i, 0] = avg_r2_gof_train / k
            # print(r2_goodness_of_fit[i, 0])
            r2_goodness_of_fit[i, 1] = avg_r2_gof_test / k
            # print(r2_goodness_of_fit[i, 1])

            if (r2_goodness_of_fit[i, 1] > max_gof):
                print(r2_goodness_of_fit[i, 1])
                max_gof = r2_goodness_of_fit[i, 1]
                deg = i
                opt_reg_lambda = reg_lambda
            # j = j + 1
    print(deg, opt_reg_lambda)

    #Identifying fit of the model
    print("MSE ", m_sqr_err_mat)
    plt.plot(np.log10(m_sqr_err_mat[:, 0]), color='LightSlateGray', label='Train Error')
    plt.plot(np.log10(m_sqr_err_mat[:, 1]), color='MediumVioletRed', label='Test Error')
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.legend(loc="upper right")
    plt.show()
    print("MAE ", m_abs_err_mat)
    print("GOF ", r2_goodness_of_fit)
    return deg, reg_lambda


def fit_model(deg, reg_lambda):
    global x_train, y_train, poly_reg, X_poly, reg_method, x, y
    # Fit Regression Model
    x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    #16 Test data points
    # 4 Train data points
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(x_train)
    poly_reg.fit(X_poly, y_train)
    reg_method = Ridge(alpha=reg_lambda)
    reg_method.fit(X_poly, y_train)
    print(reg_method.coef_)
    print(reg_method.intercept_)
    print("Variance")
    print(np.var(y - reg_method.predict(poly_reg.fit_transform(x))))
    # visualising the training set results
    plt.scatter(x_train, y_train, color='MediumVioletRed', alpha=0.4)
    x = x_train
    y = reg_method.predict(poly_reg.fit_transform(x_train))
    [x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
    plt.plot(x, y, color='LightSlateGray')
    plt.title('Train Data Model Fit')
    plt.show()
    # visualising the test set results
    # plt.scatter(X_test, y_test, color='MediumVioletRed')
    x = X_test
    y = reg_method.predict(poly_reg.fit_transform(X_test))
    [x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
    # plt.plot(x, y, color='LightSlateGray')
    # plt.show()

    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1)
    plt.scatter(X_test, y_test, color='MediumVioletRed',alpha=0.4)
    plt.plot(X_grid, reg_method.predict(poly_reg.fit_transform(X_grid)), color='LightSlateGray')
    plt.title("Test Data Prediction")
    plt.xlabel("X ")
    plt.ylabel("Y")
    plt.show()


if __name__ == '__main__':
    reg_type = parse_input()
    deg, reg_lambda = find_poly_degree(reg_type)
    fit_model(deg, reg_lambda)
