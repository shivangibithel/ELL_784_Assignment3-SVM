import numpy as np
import sys
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 1. Degree values 1 to 11 ? Find the optimal range - Done. between 4 to 9
# With 100 samples - d= 8
# With 20 Samples d = 6
# 2. Merge the classes - Done
# 3. Check the regularizer alpha - Vishnu
# 4. Dataset Update - Done
# 5. Fix the plot - Done
# 6. Check if KFold validation is required.

# 1. Done - To begin with, use only the first 20 data points in your file.
# 2. Solve the polynomial curve fitting regression problem using error function minimisation.(sklearn not
# 3. Define your own error function other than the sum-of-squares error. (TBD) - Vishnu
# 4. Try different error formulations and report the results. Done(MAE, MAVE, MSE).. Graph
# 5. Done - Use a goodness-of-fit measure for polynomials of different order.
# Can you distinguish overfitting, underfitting, and the best fit? To Check -Vishnu


# 6. Obtain an estimate for the noise variance.TBD
# 7. Introduce regularisation and observe the changes. For quadratic regularisation, can you obtain an estimate of the optimal value for the regularisation parameter lamba? What is your corresponding best guess for the underlying polynomial? And the noise variance?
# 8. Done - Now repeat all of the above using the full data set of 100 data points.
# 9. How are your results affected by adding more data? Comment on the differences.
# 10. What is your final estimate of the underlying polynomial? Why?

'''------------------------Reading the data---------------------------------'''
def parse():
    global x, y
    filename = '..\data\\regression_data.txt'
    x = []
    y = []
    X = []
    training_count = 0
    try:
        training_count = int(input('Enter input training count: 20 or 100  :'))
        print(training_count)
        if(training_count !=20 and training_count !=100):
            print("Training count should either be 20 or 100")
            sys.exit(1)
    except:
        sys.exit(1)
    #Parse the file and read column values
    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter=' ')
        for row in reader:
            x.append(float(row[1]))
            y.append(float(row[3]))
    #Number of records in the file
    n = len(x)
    x = np.asarray(x, dtype=float).reshape((n,1))
    y=  np.asarray(y, dtype=float).reshape((n,1))
    x = x[0:training_count,:]
    y = y[0:training_count,:]

    #Data Visualization
    plt.scatter(x,y, color="MediumVioletRed", alpha=0.4)
    plt.title("Data Visualization "+ str(training_count) + ' data points')
    plt.xlabel("input variable x")
    plt.ylabel("target variable y")
    plt.show()

# Vary the polynomial degree to find the minimum error and maximum goodness of fit
def find_poly_degree():
    global deg, poly_reg, X_train, y_train, X_poly, lin_reg_2
    deg, max_gof = 0, 0
    k = 5
    r2_goodness_of_fit, m_sqr_err_mat, m_abs_err_mat = np.zeros((11, 2)),np.zeros((11, 2)),np.zeros((11, 2))

    for i in range(0, 10):
        poly_reg = PolynomialFeatures(degree=i)
        avg_m_sqr_err_mat_train, avg_m_sqr_err_mat_test, avg_m_abs_err_mat_train, avg_m_abs_err_mat_test, avg_r2_gof_train, avg_r2_gof_test = 0, 0, 0, 0, 0, 0
        kf = KFold(n_splits=k, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(x):
            #train-test data split
            X_train, X_valid = x[train_index], x[test_index]
            y_train, y_valid = y[train_index], y[test_index]

            X_poly = poly_reg.fit_transform(X_train)
            poly_reg.fit(X_poly, y_train)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(X_poly, y_train)

            train_error_poly_1 = (mean_squared_error(y_train, lin_reg_2.predict(X_poly)))
            test_error_poly_1 = (mean_squared_error(y_valid, lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
            m_sqr_err_mat[i, 0] = train_error_poly_1
            m_sqr_err_mat[i, 1] = test_error_poly_1

            avg_m_sqr_err_mat_train = avg_m_sqr_err_mat_train + m_sqr_err_mat[i, 0]
            avg_m_sqr_err_mat_test = avg_m_sqr_err_mat_test + m_sqr_err_mat[i, 1]

            train_error_poly_2 = (mean_absolute_error(y_train, lin_reg_2.predict(X_poly)))
            test_error_poly_2 = (mean_absolute_error(y_valid, lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
            m_abs_err_mat[i, 0] = train_error_poly_2
            m_abs_err_mat[i, 1] = test_error_poly_2

            avg_m_abs_err_mat_train = avg_m_abs_err_mat_train + m_abs_err_mat[i, 0]
            avg_m_abs_err_mat_test = avg_m_abs_err_mat_test + m_abs_err_mat[i, 1]

            # train_error_poly_3 = (median_absolute_error(y_train, reg_method.predict(polyfit_x)))
            # test_error_poly_3 = (median_absolute_error(y_valid, reg_method.predict(poly_reg.fit_transform(X_valid))))
            # med_abs_err_mat[i, 0] = train_error_poly_3
            # med_abs_err_mat[i, 1] = test_error_poly_3
            #
            # avg_med_abs_err_mat_train = avg_med_abs_err_mat_train + med_abs_err_mat[i, 0]
            # avg_med_abs_err_mat_test = avg_med_abs_err_mat_test + med_abs_err_mat[i, 1]

            train_fit = (r2_score(y_train, lin_reg_2.predict(X_poly)))
            test_fit = (r2_score(y_valid, lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
            r2_goodness_of_fit[i, 0] = train_fit
            r2_goodness_of_fit[i, 1] = test_fit

            avg_r2_gof_train = avg_r2_gof_train + r2_goodness_of_fit[i, 0]
            avg_r2_gof_test = avg_r2_gof_test + r2_goodness_of_fit[i, 1]

        m_sqr_err_mat[i, 0] = avg_m_sqr_err_mat_train / k
        m_sqr_err_mat[i, 1] = avg_m_sqr_err_mat_test / k

        m_abs_err_mat[i, 0] = avg_m_abs_err_mat_train / k
        m_abs_err_mat[i, 1] = avg_m_abs_err_mat_test / k

        # med_abs_err_mat[i, 0] = avg_med_abs_err_mat_train / k
        # med_abs_err_mat[i, 1] = avg_med_abs_err_mat_test / k

        r2_goodness_of_fit[i, 0] = avg_r2_gof_train / k
        r2_goodness_of_fit[i, 1] = avg_r2_gof_test / k

        if (r2_goodness_of_fit[i, 1] > max_gof):
            #TODO find it while minimizing error
            max_gof = r2_goodness_of_fit[i, 1]
            deg = i
    # error_matrix = np.hstack((m_sqr_err_mat, m_abs_err_mat, med_abs_err_mat, r2_goodness_of_fit))
    print('Degree M of Polynomial', deg)
    '''---Plotting the error matrix and identifying the area of overfitting------'''
    print(m_sqr_err_mat)

    cmap = plt.get_cmap('Pastel1')
    plt.set_cmap(cmap)
    for i in range(4, 9):
        plt.plot(np.log10(m_sqr_err_mat[i, 0]), color='LightSlateGray')
        plt.plot(np.log10(m_sqr_err_mat[i, 1]), color='MediumVioletRed')
    plt.title("Squared Error Matrix")
    plt.xlabel("Train Error")
    plt.ylabel("Test Error")
    plt.show()
    # print(m_abs_err_mat)
    # plt.plot(np.log10(m_abs_err_mat[:, 0]), color='LightSlateGray')
    # plt.plot(np.log10(m_abs_err_mat[:, 1]), color='MediumVioletRed')
    # plt.title("Absolute Error Matrix")
    # plt.xlabel("Train Error")
    # plt.ylabel("Test Error")
    # # plt.show()
    # print(med_abs_err_mat)
    # plt.plot(med_abs_err_mat[:, 0], color='LightSlateGray')
    # plt.plot(med_abs_err_mat[:, 1], color='MediumVioletRed')
    # plt.title("Median Absolute Error Matrix")
    # plt.xlabel("Train Error")
    # plt.ylabel("Test Error")
    # # plt.show()
    # print(r2_goodness_of_fit)
    # plt.plot(r2_goodness_of_fit[:, 0], color='LightSlateGray')
    # plt.plot(r2_goodness_of_fit[:, 1], color='MediumVioletRed', alpha=0.4)
    # plt.title("Goodness of fit")
    # plt.xlabel("Train Error")
    # plt.ylabel("Test Error")
    # plt.show()
    return deg

'''---------------Fitting the Polynomial Regression Model and  -----------------'''
def fit_model(deg):
    global X_train, y_train, poly_reg, X_poly, lin_reg_2, x, y
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y_train)
    print(lin_reg_2.coef_)
    print(lin_reg_2.intercept_)
    '''-----------------------------------Variance---------------------------------'''
    print("Variance")
    print(np.var(y - lin_reg_2.predict(poly_reg.fit_transform(x))))
    '''--------------Visualizing Train and Test Set separately ----------------'''
    # visualising the trainig set results
    plt.clf()
    plt.scatter(X_train, y_train, color='MediumVioletRed', alpha=0.4)
    x = X_train
    y = lin_reg_2.predict(poly_reg.fit_transform(X_train))
    [x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
    # plt.plot(x, y, color='LightSlateGray')
    # plt.xlabel('X Train')
    # plt.ylabel('Predicted Y')
    # plt.title('Training Set Results')
    # plt.show()
    # # visualising the test set results
    # plt.scatter(X_test, y_test, color='MediumVioletRed', alpha=0.4)
    # x = X_test
    # y = reg_method.predict(poly_reg.fit_transform(X_test))
    # [x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
    # plt.plot(x, y, color='LightSlateGray')
    # plt.title('Test Set Results')
    # plt.xlabel('X Test')
    # plt.ylabel('Predicted Y')
    # plt.show()
    # '''---Now lets visualize the Polynomial Regression with respect to the test data--------'''
    # X_grid = np.arange(min(X), max(X), 0.1)
    # X_grid = X_grid.reshape(len(X_grid), 1)
    # plt.scatter(X_test, y_test, color='MediumVioletRed', alpha=0.4)
    # plt.plot(X_grid, reg_method.predict(poly_reg.fit_transform(X_grid)), color='LightSlateGray')
    # plt.title("Prediction")
    # plt.xlabel("X level")
    # plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    parse()
    deg = find_poly_degree()
    fit_model(deg)