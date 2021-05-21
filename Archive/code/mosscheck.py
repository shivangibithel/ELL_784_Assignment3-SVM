
import numpy as np


# y = [31437261321,
# 1.43856E+11,
# 1.42927E+11,
# 1.35975E+11,
# 1.34316E+11
# ]
# print(np.average(y))
dataset = [1,2,3,4,5,6,7,8,9,10]
train = []
test = []
fold = [[2, 1], [6, 0], [7, 8], [9, 5], [4, 3]]
cross_val={'train': train, 'test': test}
for i, testi in enumerate(fold):
    train.append(fold[:i] + fold[i+1:])
    test.append(testi)
print(cross_val)
print(enumerate(fold))



