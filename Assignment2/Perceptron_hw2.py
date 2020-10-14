import random
import matplotlib.pyplot as plt
import numpy as np

"""
f=open('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/hw2_data1.txt',"r")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split(',')[0])
f.close()

target = []
for sign in result:
    if sign == '+':
        target.append(1)
    else:
        target.append(-1)

f=open('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/hw2_data1.txt',"r")
lines1=f.readlines()
result2=[]
for x in lines1:
    result2.append(x.split(',')[1])
f.close()

x_coords = []
for coord in result2:
    x_coords.append(coord)

f=open('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/hw2_data1.txt',"r")
lines2=f.readlines()
result3=[]
for x in lines2:
    x_ = x.split(',')[2].rstrip('\n')
    #newx = x_.replace("'","")
    result3.append(x_)
    #result3.replace("'","")
    #print(result3)
#print(result3)

y_coords = []
for coord in result3:
    y_coords.append(coord)
#print(y_coords)

data = []
for x,y in zip(x_coords,y_coords):
    data.append([x,y])
#print(data)

X = np.array(data,dtype=float)
dataset = np.insert(X,2,1,axis=1)
#print(dataset)
"""
def Perceptron(dataset,target):   
    incorrect_classify = dataset
    correct_classify = []
    class0 = []
    class1 = []
    w = np.random.randint(-10,10,dataset.shape[1])
    while list(incorrect_classify):
        index = np.random.choice(incorrect_classify.shape[0])  
        example = incorrect_classify[index]
        if np.dot(example, w) <= 0 and target[index] == -1:
            correct_classify.append(example) #add to correct classify
            incorrect_classify = np.delete(incorrect_classify,index,0) #delete from incorrect classify
            target = np.delete(target,index)
            class0.append(example)
            update_weight = False
        elif np.dot(example, w) > 0 and target[index] == 1:
            correct_classify.append(example) #add to correct classify
            incorrect_classify = np.delete(incorrect_classify,index,0) #delete from incorrect classify
            target = np.delete(target,index)
            class1.append(example)
            update_weight = False
        else:
            update_weight = True
            w = w + example*target[index] #update weight
    if update_weight: #if the weights were updated on the last iteration, then convergence didn't occur
        print(
            """
            Convergence NOT reached. Dataset not linearly separable.
            """
            )
    """
    #check geometrically if points are correctly classified    
    plt.xlim((-21, 21))
    plt.ylim((-21, 21))
    plt.subplot(1, 2, 1)
    for sample in dataset:
        check1 = w[0]*sample[0] + w[1]*sample[1] + w[2]
        if check1 > 0:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
            
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    #plot with negative signs for class -1, plus signs for class 1
    NewX = np.arange(0,100)
    plt.plot(NewX, (-w[0]/w[1])*NewX - (w[2]/w[1]), c='red',label='boundary') #plot boundary line
    plt.title('Perceptron')
    plt.legend()
    plt.show()
    
    
    #another check
    plt.subplot(1,2,2)
    plt.xlim((-21, 21))
    plt.ylim((-21, 21))
    plt.scatter((*zip(*class0)),c='blue',label='class 0')
    plt.scatter((*zip(*class1)),c='green',label='class 1')
    plt.show()
    """
    #return weight vector
    print(w)
    return(w)
        
#Perceptron(dataset,target)

#Okay, the perceptron doesn't really work with this data. 