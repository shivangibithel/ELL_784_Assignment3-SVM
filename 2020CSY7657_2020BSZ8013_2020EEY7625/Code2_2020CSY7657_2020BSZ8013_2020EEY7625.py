import csv

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_input(filename, x, y):
    row_num = 0
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            l = len(row)
            for features in range(l):
                if features == 0:
                    y[row_num][0] = int(row[features])
                else:
                    for i in range(1, 9):
                        if (row[features])[0] == str(i):
                            temp_string = (row[features])[2:]
                            temp_float = float(temp_string)
                            x[row_num][i - 1] = temp_float
            row_num = row_num + 1

    return x, y


def linear_kernel():
    i, max_acc, opt_C, opt_gamma = 0, 0, 0, 0
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 150, 200]
    acc_array, c_array = np.zeros((len(C_values), 1)), np.zeros((len(C_values), 1))

    for C in C_values:
        classifier = SVC(C=C, kernel='linear', random_state=0)
        classifier.fit(X_train, y_train.ravel())
        y_pred_valid = classifier.predict(X_valid)
        cm = confusion_matrix(y_valid, y_pred_valid)
        Accuracy = (cm[0, 0] + cm[1, 1]) / (y_valid.size)
        # print (Accuracy)
        acc_array[i, 0] = Accuracy * 100
        c_array[i, 0] = C
        if Accuracy > max_acc:
            max_acc = Accuracy
            opt_C = C
        i = i + 1

    print("Maximum Accuracy : ", max_acc * 100)
    print("Optimal Value of C is : ", opt_C)
    # print(acc_array)
    # print(c_array)
    # Uncomment this to plot the graphs
    # plt.plot(c_array, acc_array)
    # plt.xlabel("C Value")
    # plt.ylabel("Accuracy Value")
    # plt.yscale('linear')
    # plt.grid(True)
    # plt.xscale('log')
    # plt.show()
    return opt_C


def svm(C, X_train, y_train, X_valid, y_valid):
    classifier = SVC(C=C, kernel='linear', random_state=0)
    classifier.fit(X_train, y_train.ravel())
    print("Coefficient of the optimal Classifier", classifier.coef_)
    print("Intercept of the optimal Classifier", classifier.intercept_)

    y_valid_pred = classifier.predict(X_valid)

    cm = confusion_matrix(y_valid, y_valid_pred)
    print("confusion_matrix:\n", cm)

    Accuracy = (cm[0, 0] + cm[1, 1]) / (y_valid.size)
    print("Accuracy = ", Accuracy * 100)


def RBF_Kernel():
    DataSet1 = pd.DataFrame(X_valid)
    DataSet2 = pd.DataFrame(y_valid)
    DataSet = pd.concat([DataSet1, DataSet2], axis=1, ignore_index=True, sort=False)

    x_columns = np.r_[0:8]
    X = DataSet.iloc[:, x_columns].values  # X feature vector
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 150, 200]
    gamma_range = [0.001, 0.01, 0.1, 1, 10, 100, 150, 200]

    m1 = len(C_range)

    acc_mat = np.zeros((m1, m1))
    q, p, max_acc, opt_gamma, opt_C = 0, 0, 0, 0, 0

    for C in C_range:
        p = 0
        for gamma in gamma_range:
            Avg_Acc, k = 0, 5
            ss = int(X.shape[0] / k)
            for i in range(k):
                x_kfold_test = DataSet.iloc[i * ss: (i + 1) * ss, x_columns].values  # Test subset
                x_kfold_train = DataSet.iloc[
                    np.r_[0: i * ss, (i + 1) * ss: k * ss], x_columns].values  # Training subset
                y_kfold_test = DataSet.iloc[i * ss: (i + 1) * ss, len(DataSet.columns) - 1].values  # Test Y
                y_kfold_train = DataSet.iloc[
                    np.r_[0: i * ss, (i + 1) * ss: k * ss], len(DataSet.columns) - 1].values  # Training Y

                classifier = SVC(C=C, gamma=gamma, kernel='rbf', random_state=0)
                classifier.fit(x_kfold_train, y_kfold_train.ravel())

                y_kfold_pred = classifier.predict(x_kfold_test)

                cm = confusion_matrix(y_kfold_test, y_kfold_pred)
                Accuracy = (cm[0, 0] + cm[1, 1]) / (y_kfold_test.size)
                Avg_Acc += Accuracy
            Avg_Acc = (Avg_Acc) / 5
            print("Set Avg Acc = ", Avg_Acc * 100)
            acc_mat[q][p] = (Avg_Acc) * 100
            if Avg_Acc * 100 > max_acc:
                max_acc = acc_mat[q][p]
                opt_C = C
                opt_gamma = gamma
            p += 1
        q += 1
    print(acc_mat)
    print("Maximum Accuracy : ", max_acc)
    print("Optimal Value of C : ", opt_C)
    print("Optimal Value of Gamma : ", opt_gamma)
    print("Optimal Value of Sigma : ", np.sqrt(np.divide(1, 2 * opt_gamma)))

    # Uncomment this to plot the graphs
    # plt.plot(C_range, acc_mat[:, 0])
    # plt.xlabel("C Value")
    # plt.ylabel("Accuracy Value")
    # plt.yscale('linear')
    # plt.grid(True)
    # plt.xscale('log')
    # plt.show()
    # plt.plot(acc_mat[:, 0])
    # plt.show()
    return opt_C, opt_gamma


def save_output(y_pred_test, filename_test, save_file_name_output):
    row_num_test = 0
    file_test = open(filename_test)
    file_output = open(save_file_name_output, 'w+')

    for line in file_test.readlines():
        text = str(int(y_pred_test[row_num_test])) + line[1:]
        file_output.write(text)
        row_num_test = row_num_test + 1

    file_test.close()
    file_output.close()


if __name__ == '__main__':
    # Read Train Data
    df = pd.read_csv('data\\RNA_train_data.txt', sep=" ", header=None)
    count = len(df)
    print('No of Train Records = ', count)
    x = np.zeros((count, 8))  # 8 features per record
    y = np.zeros((count, 1))  # 1 label per record
    x, y = parse_input('data\\RNA_train_data.txt', x, y)
    # np.savetxt('train_data.txt',x)
    # np.savetxt('train_label.txt',y)

    # Read Test Data
    df = pd.read_csv('data\\RNA_test_data.txt', sep=" ", header=None)
    count_test = len(df)
    X_test = np.zeros((count_test, 8))  # 8 features per record
    y_test = np.zeros((count_test, 1))
    print('No of Test Records = ', count_test)
    X_test, y_test = parse_input('data\\RNA_test_data.txt', X_test, y_test)
    # np.savetxt('test_data.txt',X_test)
    # np.savetxt('test_label.txt',y_test)

    # 1. Spilt the training data set to form validation and training data sets. (50% random)
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.50, shuffle=True, random_state=None)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)

    '''Classification using linear SVM'''

    #   1. Train a set of linear SVMs with different values of the regularisation parameter C using the training data set.
    opt_C = linear_kernel()
    svm(opt_C, X_train, y_train, X_valid, y_valid)  # training svm with opt_C

    '''Classification using Gaussian (RBF) kernel SVM'''
    # 2. choose 50% of the training set as the cross validation set. Next, divide the cross validation set into 5 subsets of equal size.

    opt_C, opt_gamma = RBF_Kernel()
    classifier = SVC(C=opt_C, gamma=opt_gamma, kernel='rbf', random_state=0)
    classifier.fit(x, y.ravel())
    print('Corresponding Optimal Intercept : ', classifier.intercept_)
    y_pred_test = classifier.predict(X_test)

    save_output(y_pred_test, 'data\RNA_test_data.txt', 'data\RNA_test_output.txt')
