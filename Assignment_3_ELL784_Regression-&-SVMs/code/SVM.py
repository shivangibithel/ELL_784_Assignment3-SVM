import numpy as sri
import matplotlib.pyplot as plt 
import csv

'''--------------------------Reading the train data---------------------------------'''

filename = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignment_3_ELL784/RNA_train_data.txt'
count = 0 ;
with open(filename,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        count = count+1;

print ('No of Train Records = ',count)

x = sri.zeros((count,8))
y = sri.zeros((count,1))

row_num = 0 ;
with open(filename,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        k = len(row)
        for strings in range(k):
            
            if strings == 0:
                y[row_num][0] =  int(row[strings])
            elif (row[strings])[0] == '1':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][0] = temp_float
            elif (row[strings])[0] == '2':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][1] = temp_float
            elif (row[strings])[0] == '3':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][2] = temp_float
            elif (row[strings])[0] == '4':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][3] = temp_float
            elif (row[strings])[0] == '5':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][4] = temp_float
            elif (row[strings])[0] == '6':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][5] = temp_float
            elif (row[strings])[0] == '7':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][6] = temp_float
            elif (row[strings])[0] == '8':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                x[row_num][7] = temp_float
            
        row_num = row_num + 1; 
        
'''--------------------------Reading the test data---------------------------------'''

filename_test = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignment_3_ELL784/RNA_test_data.txt'
count_test = 0 ;
with open(filename_test,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        count_test = count_test + 1;

print ('No of Test Records = ',count_test)

X_test = sri.zeros((count_test,8))
y_test = sri.zeros((count_test,1))

row_num_test = 0 ;
with open(filename_test,'r') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        k = len(row)
        for strings in range(k):
            
            if strings == 0:
                y_test[row_num_test][0] =  int(row[strings])
            elif (row[strings])[0] == '1':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][0] = temp_float
            elif (row[strings])[0] == '2':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][1] = temp_float
            elif (row[strings])[0] == '3':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][2] = temp_float
            elif (row[strings])[0] == '4':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][3] = temp_float
            elif (row[strings])[0] == '5':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][4] = temp_float
            elif (row[strings])[0] == '6':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][5] = temp_float
            elif (row[strings])[0] == '7':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][6] = temp_float
            elif (row[strings])[0] == '8':
                temp_string = (row[strings])[2:]
                temp_float= float(temp_string)
                X_test[row_num_test][7] = temp_float
            
        row_num_test = row_num_test + 1; 



'''-------Splitting the dataset into the Training set and Validation set---------'''

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.50, random_state = 0)

'''-------------------------Feature Scaling---------------------------------'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)


'''---------- data in numpy array for online visualisation(in 8 dimentions)----------------'''

save_file_name = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignment_3_ELL784/numpy_data/'

sri.save(save_file_name+'X.npy',x)
sri.save(save_file_name+'y.npy',y)
sri.save(save_file_name+'X_train.npy',X_train)
sri.save(save_file_name+'X_valid.npy',X_valid)
sri.save(save_file_name+'y_train.npy',y_train)
sri.save(save_file_name+'y_valid.npy',y_valid)
sri.save(save_file_name+'X_test.npy',X_test)
sri.savetxt(save_file_name +'X.txt', x, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')


'''--------------------------Visualizing the data(PCA) ----------------------'''
 
from sklearn.decomposition import PCA
n_components = 2 

pca_50 = PCA(n_components)
pca_result_50 = pca_50.fit_transform(x)
plt.scatter(pca_result_50[:,0],pca_result_50[:,1])

'''---------------------Visualizing the data(t-SNE)---------------------------'''       
            
from sklearn.manifold import TSNE
n_components = 2 

x_tsne = TSNE(n_components).fit_transform(x)
print(x_tsne .shape)
plt.scatter(x_tsne[:,0],x_tsne[:,1])
plt.show()

#projection class wise 

x_1 = sri.zeros((0,n_components))
x_2 = sri.zeros((0,n_components))


for i in range(y.shape[0]):
    print (i)
    if y[i,0] == 1:
        x_1 = sri.append(x_1,x_tsne[i,:])
    else:
        x_2 = sri.append(x_2,x_tsne[i,:])

x_1 = x_1.reshape(-1,n_components)
x_2 = x_2.reshape(-1,n_components)

plt.scatter(x_1[:,0],x_1[:,1],color='red')
plt.scatter(x_2[:,0],x_2[:,1],color='blue')
plt.show()

'''------------Fitting SVM to the Training set(TRAINING)--------------------'''
from sklearn.svm import SVC

classifier = SVC(C = 0.001,kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
print(classifier.coef_)
print(classifier.intercept_)


'''------------Predicting the Test set results(TESTING)---------------------'''

y_pred_valid = classifier.predict(X_valid)

'''---------Performance(Confusion Matrix and Efficiency)--------------------'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred_valid)
print(cm)

Accuracy = (cm[0,0]+cm[1,1])/(y_valid.size)
print (Accuracy)


'''------------Running in a loop(TRAIN -> Fit -> Accuracy) -------------------------'''
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

i,max_acc,opt_C,opt_gamma = (0,)*4
#C_2d_range = sri.logspace(-2, 10, 13)
#C_2d_range = sri.arange(0.01, 100, 0.03)
C_2d_range = [1e-3,1e-2,1e-1,1,10,1e2]
acc_array = sri.zeros((len(C_2d_range),1))
c_array = sri.zeros((len(C_2d_range),1))
    
for C in C_2d_range:
    classifier = SVC(C = C,kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred_valid = classifier.predict(X_valid)
    cm = confusion_matrix(y_valid, y_pred_valid)
    Accuracy = (cm[0,0]+cm[1,1])/(y_valid.size)
    #print (Accuracy)
    acc_array[i,0]= Accuracy*100
    c_array[i,0] = C
    if Accuracy > max_acc:
        max_acc = Accuracy
        opt_C = C
    i = i+1
    
print("Maximum Accuracy : ",max_acc,"Optimal Value of C is : ",opt_C)    

plt.plot(acc_array)
plt.show()

classifier = SVC(C = opt_C,kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
print('Corresponding Optimal Coefficients and Intercept')
print(classifier.coef_)
print(classifier.intercept_)

 
'''--------------Same Visualization with RBF Kernal-------------------------'''


C_2d_range = [1,10,50,100,150]
gamma_2d_range = [1e-2,1e-1, 1, 1e1,1e+2]

m1 = len(C_2d_range)
from sklearn.model_selection import KFold

acc_mat = sri.zeros((m1,m1))
i,j,max_acc,opt_gamma = (0,)*4

for C in C_2d_range:
    j = 0
    for gamma in gamma_2d_range:
        Avg_Acc = 0
        kf = KFold(n_splits = 5, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(x):
            
            X_train, X_valid = x[train_index], x[test_index]
            y_train, y_valid = y[train_index], y[test_index]
        
            classifier = SVC(C = C,gamma = gamma,kernel = 'rbf', random_state = 0)
            classifier.fit(X_train, y_train)
            
            y_pred_valid = classifier.predict(X_valid)
            
            cm = confusion_matrix(y_valid, y_pred_valid)
            
            Accuracy = (cm[0,0]+cm[1,1])/(y_valid.size)
            
            Avg_Acc = Avg_Acc + Accuracy
            
        Avg_Acc = Avg_Acc/5       
        print("Set Avg Acc = ",Avg_Acc*100)
        acc_mat[i][j] = (Avg_Acc)*100
        if Avg_Acc*100 > max_acc:
            max_acc = acc_mat[i][j]
            opt_C = C
            opt_gamma = gamma
        j = j+1
    i = i+1

print("Maximum Accuracy : ",max_acc)    
print("Optimal Value of C is : ",opt_C)
print("Optimal Value of Gamma : ",opt_gamma)
print("Optimal Value of Sigma : ",sri.sqrt(sri.divide(1,2*opt_gamma)))


plt.plot(acc_mat[:,0])
plt.show()


classifier = SVC(C = opt_C,gamma = opt_gamma,kernel = 'rbf', random_state = 0)
classifier.fit(x, y)
print('Corresponding Optimal Intercept : ',classifier.intercept_)
y_pred_test = classifier.predict(X_test)


'''----------------Lets now save the predicted values--------------------------'''

row_num_test = 0 ;

f = open(filename_test)
f1 = open('RNA_test_output.txt', 'w+')

for line in f.readlines():
    text = str(int(y_pred_test[row_num_test])) + line[1:]
    f1.write(text)
    row_num_test = row_num_test +1
    
f.close()
f1.close()