import numpy as np
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 1. Degree values 1 to 11 ? Find the optimal range
# 2. Merge the classes
# 3. Check the regularizer alpha
# 4. Dataset Update - Done
# 5. Fix the plot - Done
# 6. Check if KFold validation is required.
'''------------------------Reading the data---------------------------------'''

filename = '..\data\\regression_data.txt'

x,y = [], []

with open(filename,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        x.append(float(row[1]))
        y.append(float(row[3]))

n = len(x)

X = np.asarray(x, dtype = np.float64)
X = X.reshape((n,1))
y = np.asarray(y, dtype = np.float64)
y = y.reshape((n,1))

X = X[0:20,:]
y = y[0:20,:]

'''------------------------Lets visualize the data ---------------------------------'''

plt.scatter(X,y,cmap="MediumVioletRed", alpha=0.4, edgecolors="grey", linewidth=2)
plt.title("Scatter data")
plt.xlabel("X values")
plt.ylabel("y Values")
plt.show()

'''
The goodness of fit of a statistical model describes how well it fits a set of observations. 
Measures of goodness of fit typically summarize the discrepancy between observed values
and the values expected under the model in question.
measure R2

total ss = regression ss +residual ss
r-square = regress sum of square/ total sum of square

r2 is always 0-1
0 : poor fitting 
1 : good fitting 
 
if R2 value is : 0.4745
The regression model can explain about 47.45 % variation in the y values.

'''


'''-----------Lets try with different degree polynomial and plot-------------'''



m_sqr_err_mat = np.zeros((11, 2))
m_abs_err_mat = np.zeros((11, 2))
med_abs_err_mat = np.zeros((11, 2))
r2_goodness_of_fit = np.zeros((11, 2))
deg = 0 ;
max_gof = 0


for i in range(0,11):

    poly_reg = PolynomialFeatures(degree = i)

    avg_m_sqr_err_mat_train = 0
    avg_m_sqr_err_mat_test = 0

    avg_m_abs_err_mat_train = 0
    avg_m_abs_err_mat_test = 0

    avg_med_abs_err_mat_train = 0
    avg_med_abs_err_mat_test = 0

    avg_r2_goodness_of_fit_train = 0
    avg_r2_goodness_of_fit_test = 0

    kf = KFold(n_splits = 2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):

        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]


        X_poly = poly_reg.fit_transform(X_train)
        poly_reg.fit(X_poly,y_train)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly,y_train)

        train_error_poly_1 = (mean_squared_error(y_train,lin_reg_2.predict(X_poly)))
        test_error_poly_1 = (mean_squared_error(y_valid,lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
        m_sqr_err_mat[i,0] = train_error_poly_1
        m_sqr_err_mat[i,1] = test_error_poly_1

        avg_m_sqr_err_mat_train = avg_m_sqr_err_mat_train + m_sqr_err_mat[i,0]
        avg_m_sqr_err_mat_test = avg_m_sqr_err_mat_test + m_sqr_err_mat[i,1]

        train_error_poly_2 = (mean_absolute_error(y_train,lin_reg_2.predict(X_poly)))
        test_error_poly_2 = (mean_absolute_error(y_valid,lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
        m_abs_err_mat[i,0] = train_error_poly_2
        m_abs_err_mat[i,1] = test_error_poly_2

        avg_m_abs_err_mat_train = avg_m_abs_err_mat_train + m_abs_err_mat[i,0]
        avg_m_abs_err_mat_test = avg_m_abs_err_mat_test + m_abs_err_mat[i,1]

        train_error_poly_3 = (median_absolute_error(y_train,lin_reg_2.predict(X_poly)))
        test_error_poly_3 = (median_absolute_error(y_valid,lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
        med_abs_err_mat[i,0] = train_error_poly_3
        med_abs_err_mat[i,1] = test_error_poly_3

        avg_med_abs_err_mat_train = avg_med_abs_err_mat_train + med_abs_err_mat[i,0]
        avg_med_abs_err_mat_test = avg_med_abs_err_mat_test + med_abs_err_mat[i,1]

        train_fit = (r2_score(y_train,lin_reg_2.predict(X_poly)))
        test_fit = (r2_score(y_valid,lin_reg_2.predict(poly_reg.fit_transform(X_valid))))
        r2_goodness_of_fit[i,0] = train_fit
        r2_goodness_of_fit[i,1] = test_fit

        avg_r2_goodness_of_fit_train = avg_r2_goodness_of_fit_train + r2_goodness_of_fit[i,0]
        avg_r2_goodness_of_fit_test = avg_r2_goodness_of_fit_test + r2_goodness_of_fit[i,1]

    m_sqr_err_mat[i,0] = avg_m_sqr_err_mat_train/5
    m_sqr_err_mat[i,1] = avg_m_sqr_err_mat_test/5

    m_abs_err_mat[i,0] = avg_m_abs_err_mat_train/5
    m_abs_err_mat[i,1] = avg_m_abs_err_mat_test/5

    med_abs_err_mat[i,0] = avg_med_abs_err_mat_train/5
    med_abs_err_mat[i,1] = avg_med_abs_err_mat_test/5

    r2_goodness_of_fit[i,0] = avg_r2_goodness_of_fit_train/5
    r2_goodness_of_fit[i,1] = avg_r2_goodness_of_fit_test/5

    if(r2_goodness_of_fit[i,1] > max_gof):
        max_gof = r2_goodness_of_fit[i,1]
        deg = i

error_matrix = np.hstack((m_sqr_err_mat, m_abs_err_mat, med_abs_err_mat, r2_goodness_of_fit))


'''---Plotting the error matrix and identifying the area of overfitting------'''

print(m_sqr_err_mat)
plt.plot(np.log10(m_sqr_err_mat[:, 0]), color ='LightSlateGray')
plt.plot(np.log10(m_sqr_err_mat[:, 1]), color ='MediumVioletRed')
plt.title("Squared Error Matrix")
plt.xlabel("Train Error")
plt.ylabel("Test Error")
plt.show()

print(m_abs_err_mat)
plt.plot(np.log10(m_abs_err_mat[:, 0]), color ='LightSlateGray')
plt.plot(np.log10(m_abs_err_mat[:, 1]), color ='MediumVioletRed')
plt.title("Absolute Error Matrix")
plt.xlabel("Train Error")
plt.ylabel("Test Error")
plt.show()

print(med_abs_err_mat)
plt.plot(med_abs_err_mat[:,0],color = 'LightSlateGray')
plt.plot(med_abs_err_mat[:,1],color = 'MediumVioletRed')
plt.title("Median Absolute Error Matrix")
plt.xlabel("Train Error")
plt.ylabel("Test Error")
plt.show()

print(r2_goodness_of_fit)
plt.plot(r2_goodness_of_fit[:,0],color = 'LightSlateGray')
plt.plot(r2_goodness_of_fit[:,1],color = 'MediumVioletRed')
plt.title("Goodness of fit")
plt.xlabel("Train Error")
plt.ylabel("Test Error")
plt.show()

'''---------------Fitting the Polynomial Regression Model and  -----------------'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0 )


poly_reg = PolynomialFeatures(degree = deg)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly,y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
print(lin_reg_2.coef_)
print(lin_reg_2.intercept_)

'''-----------------------------------Variance---------------------------------'''

print("Variance")
print(np.var(y - lin_reg_2.predict(poly_reg.fit_transform(X))))

'''--------------Visualizing Train and Test Set separately ----------------'''

# visualising the trainig set results
plt.scatter(X_train,y_train,color = 'MediumVioletRed')
x = X_train
y = lin_reg_2.predict(poly_reg.fit_transform(X_train))
[x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
plt.plot(x,y,color = 'LightSlateGray')
plt.xlabel('X Train')
plt.ylabel('Predicted Y')
plt.title('Training Set Results')
plt.show()

#visualising the test set results 
plt.scatter(X_test,y_test,color = 'LightSlateGray')
x = X_test
y = lin_reg_2.predict(poly_reg.fit_transform(X_test))
[x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
plt.plot(x,y,color = 'MediumVioletRed')
plt.title('Test Set Results')
plt.xlabel('X Test')
plt.ylabel('Predicted Y')
plt.show()


'''---Now lets visualize the Polynomial Regression with respect to the test data--------'''

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_test,y_test,color = 'MediumVioletRed')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color ='LightSlateGray')
plt.title("Prediction")
plt.xlabel("X level")
plt.ylabel("Y")
plt.show()

# if __name__ == '__main__':
#     init()