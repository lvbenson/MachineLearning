
"""
@author: Lauren Benson
"""

import random
import matplotlib.pyplot as plt
import numpy as np


#part 1
def random_boundary(k0,k1):
    #gets random numbers for a, b, c
    og_vector = []
    a = random.randint(-20,20)
    while a==0: #make sure a can't be 0
        a = random.randint(-20,20)
    b = random.randint(-20,20)
    b=0
    while b==0: #make sure b can't be 0
        b = random.randint(-20,20)
    c = random.randint(-20,20)
    og_vector.append(a)
    og_vector.append(b)
    og_vector.append(c)
    print(og_vector,'og vector!!!')
    k0_points = []
    k1_points = []
    k1_class = []
    k0_class = []
    
    while len(k0_points) < k0: #add points to class until number reached
        #gets random point
        x1 = random.randint(-20,20)
        y1 = random.randint(-20,20)
        if a*x1 + b*y1 + c > 0: #checks if in class 0 (greater than 0)
            k0_points.append([x1,y1]) #if yes, add to k0 class
            k0_class.append(1)
        else:
            pass
    while len(k1_points) <= k1:
        x2 = random.randint(-20,20)
        y2 = random.randint(-20,20)
        if a*x2 + b*y2 + c < 0: #checks if in class 1 (less than or equal to 0)
            k1_points.append([x2,y2])
            k1_class.append(-1)
        else:
            pass
    plt.xlim((-21, 21))
    plt.ylim((-21, 21))
    plt.subplot(1, 3,1)
    plt.scatter((*zip(*k0_points)),c='green',label='Class 0')
    plt.scatter((*zip(*k1_points)),c='blue',label='Class 1')
    X = np.arange(-20,20)
    random_boundary.bound_x = X #save for part 2
    plt.plot(X, (-a/b)*X - (c/b), c='red',label='boundary') #plot boundary line
    plt.title('Linearly Separable Dataset')
    plt.legend()
    plt.show()
    #giant list of classes 0 and 1
    for i in k1_points:
        k0_points.append(i)
        
    for i in k1_class:
        k0_class.append(i)
    #sets up dataset for part 2. 
    #as a vector of ones to the input vector, effectively a bias that we use instead of the bias constant
    X = np.array(k0_points)
    data = np.insert(X,2,1,axis=1)
    random_boundary.dataset = data
    Y = k0_class
    #creates target vector, to be used in part 2. Vector of ones with randomized sign.
    #Y = [(-1)**random.randint(0,1) for i in range(len(data))]
    random_boundary.target = Y
#k0 and k1 are numbers of points in the two classes
random_boundary(20,25)

#Question 3, part 2
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

    #check geometrically if points are correctly classified    
    plt.xlim((-21, 21))
    plt.ylim((-21, 21))
    plt.subplot(1, 3, 2)
    for sample in dataset:
        check1 = w[0]*sample[0] + w[1]*sample[1] + w[2]
        if check1 > 0:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
            
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    #plot with negative signs for class -1, plus signs for class 1
    NewX = random_boundary.bound_x
    plt.plot(NewX, (-w[0]/w[1])*NewX - (w[2]/w[1]), c='red',label='boundary') #plot boundary line
    plt.title('Perceptron')
    plt.legend()
    plt.show()
    
    
    #another check
    plt.subplot(1,3,3)
    plt.xlim((-21, 21))
    plt.ylim((-21, 21))
    plt.scatter((*zip(*class0)),c='blue',label='class 0')
    plt.scatter((*zip(*class1)),c='green',label='class 1')
    plt.show()
    #return weight vector
    print(w)
    return(w)
        
Perceptron(random_boundary.dataset,random_boundary.target)

#try with a non-linearly separable dataset
X = np.array([
    [-2,4,1],
    [4,1,1],
    [1,-6,1],
    [2,4,1],
    [6,-2,1],
    [4,-4,1],
    [2,8,1],
    [-4,6,1]])

Y = [-1,-1,-1,-1,-1,-1,-1,-1]
      
#Perceptron(X,Y)