
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
# num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
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

