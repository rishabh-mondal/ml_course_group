<<<<<<< HEAD

=======
# from os import P_ALL
from itertools import count
>>>>>>> e5f3f3364c912daaf91a9e601d1ebcf3d66a7031
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
<<<<<<< HEAD
# num_average_time = 100

# Learn DTs 
# ...
# 
=======
num_average_time = 2 # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)

>>>>>>> e5f3f3364c912daaf91a9e601d1ebcf3d66a7031
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs

def run_time_complexity(case, maxdepth):
    n_array=[]
    m_array=[]
    fit_time=[]
    predict_time=[]
    avg_fit_time=[]
    avg_predict_time=[]

    for n in range(30,32): #rows
        for m in range(5,7): #columns
            n_array.append(n)
            m_array.append(m) 
            for i in range(num_average_time):               

                if(case=="DD"):
                    #Discrete input discrete output
                    X =  pd.DataFrame({i: pd.Series(np.random.randint(m, size=n), dtype="category") for i in range(m)})
                    y = pd.Series(np.random.randint(m, size=n), dtype="category")
                    label="DISCRETE INPUT DISCRETE OUTPUT"

                if(case=="DR"):
                    #discrete input real output
                    X = pd.DataFrame({i: pd.Series(np.random.randint(m, size=n), dtype="category") for i in range(m)})
                    y = pd.Series(np.random.randn(n))
                    label="DISCRETE INPUT REAL OUTPUT"

                if(case=="RD"):
                    #real input discrete output
                    X = pd.DataFrame(np.random.randn(n, m))
                    y = pd.Series(np.random.randint(m, size=n), dtype="category")
                    label="REAL INPUT DISCRETE OUTPUT"

                if(case=="RR"):
                    #real input real output
                    X = pd.DataFrame(np.random.randn(n, m))
                    y = pd.Series(np.random.randn(n))
                    label="REAL INPUT REAL OUTPUT"

                tree=DecisionTree(criterion='information_gain',max_depth=maxdepth)
                
                starttime=time.time()

                tree.fit(X,y)   # type: ignore
                endtime=time.time()
                fit_time.append((endtime - starttime))

                temp = num_average_time
                starttime=time.time()
                
                y_hat=tree.predict(X) # type: ignore

                endtime=time.time()
                
                predict_time.append((endtime-starttime))

            print('Training time for ',n,'X',m,'is ',np.sum(fit_time) / num_average_time)
            print('Testing time for ',n,'X',m,'is ',np.sum(predict_time) / num_average_time)

            avg_fit_time.append(np.sum(fit_time) / num_average_time)
            avg_predict_time.append(np.sum(predict_time) /num_average_time)
            # print('Avg training time for',label,' ',avg_fit_time)
            # print('Avg test time for',label,' ',avg_predict_time)

    scatter_plot(n_array, m_array, avg_fit_time, label+" Depth "+str(maxdepth)+' Fit') # type: ignore
    scatter_plot(n_array, m_array, avg_predict_time, label+" Depth "+str(maxdepth)+' Predict') # type: ignore

# Function to plot the results
<<<<<<< HEAD
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

def scatter_plot(x,y,z, title):
    figure = plt.figure()
    ax = plt.axes(projection = '3d')
    (x,y,z, color = 'blue')
    ax.set_xlabel('N')
    ax.set_ylabel('P')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show()

def find_runtime(case, maxdepth):
    n_list = list()
    p_list = list()
    fit_time_list = list()
    predict_time_list = list()
    
    for n in range(10,20):
        for p in range(2,7):
            n_list.append(n)
            p_list.append(p)

            # Real Input Real Output Case
            if(case == 1):
                X = pd.DataFrame(np.random.randn(n, p))
                y = pd.Series(np.random.randn(n))
                title = 'Real Input Real Output case'

            # Real Input Discrete Output
            if(case == 2):
                X = pd.DataFrame(np.random.randn(n, p))
                y = pd.Series(np.random.randint(p, size = n), dtype="category")
                title = 'Real Input Discrete Output'

            # Discrete Input Discrete Output
            if(case == 3):
                X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n), dtype="category") for i in range(p)})
                y = pd.Series(np.random.randint(p, size = n), dtype="category")
                title = 'Discrete Input Discrete Output'

            # Discrete Input Real Output
            if(case == 4):
                X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n), dtype="category") for i in range(p)})
                y = pd.Series(np.random.randn(n))
                title = 'Discrete Input Real Output'



            # X,y, title = Dataset(n,p, case)
            tree = DecisionTree(criterion='gini_index', max_depth=maxdepth)
            
            starttime = time.time()
            tree.fit(X,y)
            endtime = time.time()

            fit_time_list.append(endtime - starttime)

            starttime = time.time()
            y_hat = tree.predict(X)
            endtime = time.time()

            predict_time_list.append(endtime - starttime)
    scatter_plot(n_list, p_list, fit_time_list, title + ' fit')
    scatter_plot(n_list, p_list, predict_time_list, title + ' predict')


find_runtime(1,5)
find_runtime(2,5)
find_runtime(3,5)
find_runtime(4,5)
=======

def scatter_plot(n_array, m_array, time, label):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(n_array, m_array, time)
    ax.plot3D(n_array, m_array, time)# type: ignore
    ax.set_xlabel("N")
    ax.set_ylabel("M")
    ax.set_zlabel("Time") # type: ignore
    plt.title(label)   
    plt.savefig("./Experiments/"+label+".png")

# Other functions
# ...
    
# Run the functions, Learn the DTs and Show the results/plots
    
run_time_complexity("DD", 5)
run_time_complexity("DR", 5)
run_time_complexity("RR", 5)
run_time_complexity("RD", 5)


>>>>>>> e5f3f3364c912daaf91a9e601d1ebcf3d66a7031

