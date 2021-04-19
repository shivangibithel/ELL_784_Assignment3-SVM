import numpy as sri
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

'''------------------------Reading the data---------------------------------'''

filename = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignment_3_ELL784/data_ml_ass_3.txt'

x,y = [], []

with open(filename,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        x.append(row[1])
        y.append(row[3])
        
n = len(x)
X = sri.asarray(x, dtype = sri.float64)
X = X.reshape((n,1))
y = sri.asarray(y, dtype = sri.float64)
y = y.reshape((n,1))
Y = y

'''------------------------Lets visualize the data ---------------------------------'''

plt.scatter(X,y,color = 'red')
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


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
m_sqr_err_mat = sri.zeros((11,2))
m_abs_err_mat = sri.zeros((11,2))
med_abs_err_mat = sri.zeros((11,2))
r2_goodness_of_fit = sri.zeros((11,2))
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
    
    kf = KFold(n_splits = 5, random_state=None, shuffle=False)
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

error_matrix = sri.hstack((m_sqr_err_mat,m_abs_err_mat,med_abs_err_mat,r2_goodness_of_fit))

    
'''---Plotting the error matrix and identifying the area of overfitting------'''

print(m_sqr_err_mat)
plt.plot(sri.log10(m_sqr_err_mat[:,0]),color = 'blue')
plt.plot(sri.log10(m_sqr_err_mat[:,1]),color = 'red')
plt.show()

print(m_abs_err_mat)
plt.plot(sri.log10(m_abs_err_mat[:,0]),color = 'blue')
plt.plot(sri.log10(m_abs_err_mat[:,1]),color = 'red')

print(med_abs_err_mat)
plt.plot(sri.log10(med_abs_err_mat[:,0]),color = 'blue')
plt.plot(sri.log10(med_abs_err_mat[:,1]),color = 'red')

print(r2_goodness_of_fit)
plt.plot(r2_goodness_of_fit[:,0],color = 'blue')
plt.plot(r2_goodness_of_fit[:,1],color = 'red')

'''---------------Fitting the Polynomial Regression Model and  -----------------'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0 )


poly_reg = PolynomialFeatures(degree = deg)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly,y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
print(lin_reg_2.coef_)
print(lin_reg_2.intercept_)

'''-----------------------------------Variance---------------------------------'''

print("Variance")
print(sri.var(Y - lin_reg_2.predict(poly_reg.fit_transform(X))))
 
'''--------------Visualizing Train and Test Set separately ----------------'''

#visualising the trainig set results 
plt.scatter(X_train,y_train,color = 'R')
x = X_train
y = lin_reg_2.predict(poly_reg.fit_transform(X_train))
[x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
plt.plot(x,y,color = 'Blue')
plt.show()

#visualising the test set results 
plt.scatter(X_test,y_test,color = 'R')
x = X_test
y = lin_reg_2.predict(poly_reg.fit_transform(X_test))
[x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
plt.plot(x,y,color = 'Blue')
plt.show()


'''---Now lets visualize the Polynomial Regression with respect to the test data--------'''

X_grid = sri.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color ='blue')
plt.title("Prediction")
plt.xlabel("X level")
plt.ylabel("Y")
plt.show()