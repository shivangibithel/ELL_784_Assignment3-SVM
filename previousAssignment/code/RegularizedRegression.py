import numpy as np
import sys
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

'''------------------------Reading the data---------------------------------'''

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

with open(filename,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        x.append(row[1])
        y.append(row[3])
        
n = len(x)
X = np.asarray(x, dtype = np.float64)
X = X.reshape((n,1))
y = np.asarray(y, dtype = np.float64)
y = y.reshape((n,1))

X = X[0:training_count,:]
y = y[0:training_count,:]

'''------------------------Lets visualize the data ---------------------------------'''

plt.scatter(X,y,color = 'MediumVioletRed', alpha =0.4)
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

'''------------ Fitting Ridge Regression to the training data Set ---------------------'''

'''--------------------------Looking for best alpha and degree----------------------------------'''

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

alpha_range = [1e-2,1e-1,1,10,1e2]
m = len(alpha_range)

m_sqr_err_mat_train = np.zeros((11, m))
m_sqr_err_mat_test = np.zeros((11, m))


r2_goodness_of_fit_train_mat = np.zeros((11, m))
r2_goodness_of_fit_test_mat = np.zeros((11, m))

deg = 0 
max_gof = 0
opt_deg = 0
opt_alpha = 0

alpha_range = [0.01,0.1,1,10,100]

for i in range(0,11):
    
    poly_reg = PolynomialFeatures(degree = i)
    j = 0
    
    for alpha in alpha_range:
        
        avg_m_sqr_err_mat_train = 0
        avg_m_sqr_err_mat_test = 0
        
        avg_r2_score_train = 0 
        avg_r2_score_test = 0 
    
        kf = KFold(n_splits = 5, random_state=None, shuffle=False)
        
        for train_index, test_index in kf.split(X):
            
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
        
            X_poly = poly_reg.fit_transform(X_train)
            poly_reg.fit(X_poly,y_train)
            ridgeregression = Ridge(alpha = alpha)
            ridgeregression.fit(X_poly,y_train)

            train_error_poly_1 = (mean_squared_error(y_train,ridgeregression.predict(X_poly)))
            test_error_poly_1 = (mean_squared_error(y_valid,ridgeregression.predict(poly_reg.fit_transform(X_valid))))
        
            avg_m_sqr_err_mat_train = avg_m_sqr_err_mat_train + train_error_poly_1
            avg_m_sqr_err_mat_test = avg_m_sqr_err_mat_test + test_error_poly_1

            r2_gof_train = (r2_score(y_train,ridgeregression.predict(X_poly)))
            r2_gof_test = (r2_score(y_valid,ridgeregression.predict(poly_reg.fit_transform(X_valid))))
        
            avg_r2_score_train = avg_r2_score_train + r2_gof_train
            avg_r2_score_test = avg_r2_score_test + r2_gof_test

        m_sqr_err_mat_train[i,j] = avg_m_sqr_err_mat_train/5
        m_sqr_err_mat_test[i,j] = avg_m_sqr_err_mat_test/5
        
        r2_goodness_of_fit_train_mat[i,j] = avg_r2_score_train/5
        r2_goodness_of_fit_test_mat[i,j] = avg_r2_score_test/5
        
        if(r2_goodness_of_fit_test_mat[i,j] > max_gof):
            print(r2_goodness_of_fit_test_mat[i,j])
            max_gof = r2_goodness_of_fit_test_mat[i,j]
            opt_deg = i
            opt_alpha = alpha
            
        j = j+1

'''---Plotting the error matrix and identifying the area of overfitting------'''

plt.plot(np.log10(m_sqr_err_mat_train[opt_deg, :]), color ='LightSlateGray')
plt.plot(np.log10(m_sqr_err_mat_test[opt_deg, :]), color ='MediumVioletRed')


'''----------------------Goodness of Fit measure--------------------------------'''


'''---Plotting the error matrix and identifying the area of overfitting------'''

print('Goodness_of_fit')
plt.plot(np.log10(r2_goodness_of_fit_train_mat[opt_deg, :]), color ='LightSlateGray')
plt.plot(np.log10(r2_goodness_of_fit_test_mat[opt_deg, :]), color ='MediumVioletRed')
plt.show()



'''-----------------------------------Variance---------------------------------'''

print("Variance")
ridgeregression = Ridge(alpha = opt_alpha)
poly_reg = PolynomialFeatures(degree = opt_deg)
X_poly = poly_reg.fit_transform(X_train)
ridgeregression.fit(X_poly,y_train) 
print(ridgeregression.coef_)
print(ridgeregression.intercept_)
print(np.var(y - ridgeregression.predict(poly_reg.fit_transform(X))))

