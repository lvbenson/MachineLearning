import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from SVM_test import w, x_train_data, y_train, x_test_data, y_test



accuracy = np.dot(w, x_train_data)

#negative = accuracy < 0
#positive = accuracy > 0

correct_neg = []
correct_pos = []
for x,y in zip(accuracy,y_train):
    if x < 0 and y == -1:
        correct_neg.append(1)
    elif x > 0 and y == 1:
        correct_pos.append(1)
    else:
        pass

acc_neg = len(correct_neg)/60000
print(acc_neg, 'acc neg train')

acc_pos = len(correct_pos)/60000
print(acc_pos, 'acc pos train')

accuracy_test = np.dot(w,x_test_data)
correct_neg_test = []
correct_pos_test = []
for x,y in zip(accuracy_test,y_test):
    if x < 0 and y == -1:
        correct_neg_test.append(1)
    elif x > 0 and y == 1:
        correct_pos_test.append(1)
    else:
        pass
acc_neg_test = len(correct_neg_test)/10000
print(acc_neg_test,'acc neg test')
acc_pos_test = len(correct_pos_test)/10000
print(acc_pos_test, 'acc pos test')



#for each element, where it is TRUE if value is greater than 1
#do a boolean AND against the class vector, check for each "true" is the corresponding target "1"
