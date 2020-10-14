import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
"""
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("hw2_X_train", "hw2_Y_train",
        "data_train.csv", 60000)
convert("hw2_X_test", "hw2_Y_test",
        "data_test.csv", 10000)
"""
train = pd.read_csv('data_train.csv',header=None)
test = pd.read_csv('data_test.csv',header=None)
#print(train.head(3))

x_train = train.drop(columns=[0])
x_train = np.array(x_train)
x_train_data = np.insert(x_train,784,-1,axis=1)

y_train = train[0]

x_test = test.drop(columns=[0])
x_test = np.array(x_test)
x_test_data = np.insert(x_test,784,-1,axis=1)

y_test = test[0]

y_train = y_train.map({0:1, 1:1, 2:1, 3:1, 4:1, 5:-1, 6:1, 7:-1, 8:-1, 9:-1})
y_train = np.array(y_train)

y_test = y_test.map({0:1, 1:1, 2:1, 3:1, 4:1, 5:-1, 6:1, 7:-1, 8:-1, 9:-1})
y_test = np.array(y_test)

def SVM(X, Y):

    w = np.ones(len(X[0]))
    learning = 1.0
    iterations = 1100


    for count in range(1,iterations):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(x, w)) < 1:
                w = w + learning * ((x * Y[i]) + (-2*(1/iterations)* w))
            else:
                w = w + learning * (-2*(1/count)*w)

    return w

w = SVM(x_train_data,y_train)
#print(w)

#test the accuracy
#use the weight vector (w*x)+b where b is 0
#check if the dot product is greater than 1, classification is 1
#if dot product is less than 1, classification is -1

correct_pos = []
correct_neg = []
incorrect = []
for c, x in enumerate(x_train_data):
    dot = np.dot(w,x)
    if dot > 0 and y_train[c] == 1:
        correct_pos.append(1)
    elif dot < 0 and y_train[c] == -1:
        correct_neg.append(1)
    else:
        incorrect.append(1)

corr_pos_train = len(correct_pos)
corr_neg_train = len(correct_neg)
incorr_train = len(incorrect)
accuracy_train = (corr_pos_train+corr_neg_train)/(60000)
print(accuracy_train,':Accuracy of training set')

correct_pos_test = []
correct_neg_test = []
incorrect_test = []
for c, x in enumerate(x_test_data):
    dot = np.dot(w,x)
    if dot > 0 and y_test[c] == 1:
        correct_pos_test.append(1)
    elif dot < 0 and y_test[c] == -1:
        correct_neg_test.append(1)
    else:
        incorrect_test.append(1)


corr_pos_test = len(correct_pos_test)
corr_neg_test = len(correct_neg_test)
incorr_test = len(incorrect_test)
accuracy_test = (corr_pos_test+corr_neg_test)/(10000)
print(accuracy_test,':Accuracy of test set')

