import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def count_data(filename, type):
    count = 0
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            count = count + 1
    print('No of ', type, ' Records = ', count)
    return count

    # reading data available as: 1 1:0.73834 2:0.0878964 3:0.812912 4:0.829121 5:0.197331 6:0.263444 7:0.449439 8:0.628271


def read_data(filename, x, y):
    row_num = 0
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            k = len(row)
            for strings in range(k):

                if strings == 0:
                    y[row_num][0] = int(row[strings])
                elif (row[strings])[0] == '1':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][0] = temp_float
                elif (row[strings])[0] == '2':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][1] = temp_float
                elif (row[strings])[0] == '3':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][2] = temp_float
                elif (row[strings])[0] == '4':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][3] = temp_float
                elif (row[strings])[0] == '5':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][4] = temp_float
                elif (row[strings])[0] == '6':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][5] = temp_float
                elif (row[strings])[0] == '7':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][6] = temp_float
                elif (row[strings])[0] == '8':
                    temp_string = (row[strings])[2:]
                    temp_float = float(temp_string)
                    x[row_num][7] = temp_float

            row_num = row_num + 1

        return x, y


def save_data(save_file_name, x, y, X_train, X_valid, y_train, y_valid, X_test):
    np.save(save_file_name + 'X.npy', x)
    np.save(save_file_name + 'y.npy', y)
    np.save(save_file_name + 'X_train.npy', X_train)
    np.save(save_file_name + 'X_valid.npy', X_valid)
    np.save(save_file_name + 'y_train.npy', y_train)
    np.save(save_file_name + 'y_valid.npy', y_valid)
    np.save(save_file_name + 'X_test.npy', X_test)
    np.savetxt(save_file_name + 'X.txt', x, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='',
               comments='# ')


def pca(n_components, x):
    pca_ = PCA(n_components)
    pca_result = pca_.fit_transform(x)
    # plt.scatter(pca_result[:, 0], pca_result[:, 1])


def tsne(n_components, x):
    x_tsne = TSNE(n_components).fit_transform(x)
    print(x_tsne.shape)
    # plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    # plt.show()
    return x_tsne


def svm(C, X_train, y_train, X_valid, y_valid):
    classifier = SVC(C=C, kernel='linear', random_state=0)
    classifier.fit(X_train, y_train.ravel())
    # print(classifier.coef_)
    # print(classifier.intercept_)

    # Predicting the Test set results(TESTING)
    y_pred_valid = classifier.predict(X_valid)

    # Performance(Confusion Matrix and Efficiency)
    cm = confusion_matrix(y_valid, y_pred_valid)
    print("confusion_matrix:\n", cm)

    Accuracy = (cm[0, 0] + cm[1, 1]) / (y_valid.size)
    print(Accuracy)


def optimize_linearK():
    i, max_acc, opt_C, opt_gamma = (0,) * 4
    # C_2d_range = np.logspace(-2, 10, 13)
    # C_2d_range = np.arange(0.01, 100, 0.03)
    C_2d_range = [1e-3, 1e-2, 1e-1, 1, 10, 1e2]
    acc_array = np.zeros((len(C_2d_range), 1))
    c_array = np.zeros((len(C_2d_range), 1))

    for C in C_2d_range:
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

    print("Maximum Accuracy : ", max_acc)
    print("Optimal Value of C is : ", opt_C)

    # plt.plot(acc_array)
    # plt.show()
    return opt_C


def optimize_rbf():
    C_2d_range = [1, 10, 50, 100, 150]
    gamma_2d_range = [1e-2, 1e-1, 1, 1e1, 1e+2]

    m1 = len(C_2d_range)

    acc_mat = np.zeros((m1, m1))
    i, j, max_acc, opt_gamma = (0,) * 4

    for C in C_2d_range:
        j = 0
        for gamma in gamma_2d_range:
            Avg_Acc = 0
            kf = KFold(n_splits=5, random_state=None, shuffle=False)
            for train_index, test_index in kf.split(x):
                X_train, X_valid = x[train_index], x[test_index]
                y_train, y_valid = y[train_index], y[test_index]

                classifier = SVC(C=C, gamma=gamma, kernel='rbf', random_state=0)
                classifier.fit(X_train, y_train.ravel())

                y_pred_valid = classifier.predict(X_valid)

                cm = confusion_matrix(y_valid, y_pred_valid)

                Accuracy = (cm[0, 0] + cm[1, 1]) / (y_valid.size)

                Avg_Acc = Avg_Acc + Accuracy
            # why 100??
            Avg_Acc = Avg_Acc / 5
            print("Set Avg Acc = ", Avg_Acc * 100)
            acc_mat[i][j] = (Avg_Acc) * 100
            if Avg_Acc * 100 > max_acc:
                max_acc = acc_mat[i][j]
                opt_C = C
                opt_gamma = gamma
            j = j + 1
        i = i + 1

    print("Maximum Accuracy : ", max_acc)
    print("Optimal Value of C is : ", opt_C)
    print("Optimal Value of Gamma : ", opt_gamma)
    print("Optimal Value of Sigma : ", np.sqrt(np.divide(1, 2 * opt_gamma)))

    # plt.plot(acc_mat[:, 0])
    # plt.show()
    return opt_C, opt_gamma


def save_output(filename_test, save_file_name_output):
    row_num_test = 0
    f = open(filename_test)
    f1 = open(save_file_name_output, 'w+')

    for line in f.readlines():
        text = str(int(y_pred_test[row_num_test])) + line[1:]
        f1.write(text)
        row_num_test = row_num_test + 1

    f.close()
    f1.close()


filename = '..\data\RNA_train_data.txt'
filename_test = '..\data\RNA_test_data.txt'
save_file_name = '..\data\Data_SVM_'
save_file_name_output = '..\data\RNA_test_output.txt'
if __name__ == '__main__':
    # counting and reading train data
    count = count_data(filename, type="Train")
    x = np.zeros((count, 8))
    # 8 features per record
    y = np.zeros((count, 1))
    # 1 label per record
    x, y = read_data(filename, x, y)

    # counting and reading test data
    count_test = count_data(filename_test, type="Test")
    X_test = np.zeros((count_test, 8))
    # 8 features per record
    y_test = np.zeros((count_test, 1))
    # 1 label per record

    X_test, y_test = read_data(filename_test, X_test, y_test)

    # Splitting the dataset into the Training set and Validation set
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.50, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)

    # saving data in numpy array for online visualisation(in 8 dimensions)
    save_data(save_file_name, x, y, X_train, X_valid, y_train, y_valid, X_test)

    n_components = 2

    # PCA Visualization
    # pca(n_components, x)

    # t-SNE Visualization
    x_tsne = tsne(n_components, x)

    # projection class wise
    x_1 = np.zeros((0, n_components))
    x_2 = np.zeros((0, n_components))

    for i in range(y.shape[0]):
        # print(i)
        if y[i, 0] == 1:
            x_1 = np.append(x_1, x_tsne[i, :])
        else:
            x_2 = np.append(x_2, x_tsne[i, :])

    # reshape(-1,n_components) means that where we have -1, the dimension remains same. x_1.shape --> (same,n_components)
    x_1 = x_1.reshape(-1, n_components)
    x_2 = x_2.reshape(-1, n_components)

    # plt.scatter(x_1[:, 0], x_1[:, 1], color='red')
    # plt.scatter(x_2[:, 0], x_2[:, 1], color='blue')
    # plt.show()

    # Fitting SVM to the Training set(TRAINING)
    # svm(0.001, X_train, y_train, X_valid, y_valid)

    '''------------Running in a loop(TRAIN -> Fit -> Accuracy) -------------------------'''
    # parameter tuning for Linear kernel (PART A)
    opt_C = optimize_linearK()
    svm(opt_C, X_train, y_train, X_valid, y_valid)

    # parameter tuning for RBF kernel (PART B)
    opt_C, opt_gamma = optimize_rbf()
    classifier = SVC(C=opt_C, gamma=opt_gamma, kernel='rbf', random_state=0)
    classifier.fit(x, y.ravel())
    print('Corresponding Optimal Intercept : ', classifier.intercept_)
    y_pred_test = classifier.predict(X_test)

    save_output(filename_test, save_file_name_output)
