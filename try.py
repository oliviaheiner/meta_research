# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
import time

# Define dimensions of our inputs
m = 150000
n = 100

# Make a mxn matrix A with random values from some normal distribution
A = np.random.normal(loc = -800, scale =1000, size=(m,n))

# Make a nx1 matrix x of all ones
x_target = np.random.normal(loc = 700, scale = 2000, size=(n, 1))

# set b = A*x_target
b = np.dot(A,x_target)

# set lambda value
lam = 1

# set epsilon, the error value at which we stop
epsilon = 10**(-6)

# make a datafram to store all the data
trial_data = pd.DataFrame(columns=['Beta', 'Run Time', 'Iteration Time'])


# make random values for X_0
starting_point = np.random.normal(loc = 700, scale = 2000, size=(n, 1))

# set beta value
beta = 1
while(beta<=m):
    # reassign x to be a list, where x[i] will represent the point Xi in the algorithm
    x = [None]*10000

    # make random values for X_0
    x[0] = starting_point

    # set initial error value between X_0 and the actual value for X
    # error = [None]*600000
    error = LA.norm(x[0] - x_target)

    start_time = time.time()

    i = 0
    while(error>epsilon):
        T_k_indices = random.sample(range(0, A_rows), beta)
        T_errors = [((abs((np.dot(A[j,:],x[i])-b[j])))[0]) for j in T_k_indices]
        t_k = T_k_indices[T_errors.index(max(T_errors))]
        a_t_k = A[t_k:(t_k+1),:]
        x[i+1] = x[i] - lam*(((np.dot(a_t_k,x[i]))-b[t_k])/((np.linalg.norm(a_t_k))**2))*np.transpose(a_t_k)
        i = i+1
        error = LA.norm(x[i] - x_target)

    time_elapsed[beta] = time.time() - start_time

    total_iterations[beta] = i
    print("beta = ", beta, " , iterations = ", i, "time elapsed = ", time_elapsed[beta])
    beta = beta + 50

np.savetxt("total_iterations.csv", total_iterations, delimiter=",")
np.savetxt("time_elapsed.csv",time_elapsed, delimiter=",")
np.savetxt("initial_x.csv",starting_point, delimiter=",")

cleanedIterations = [x for x in total_iterations if x != None]
cleanedTimes = [x for x in time_elapsed if x != None]
total_iterations = np.array(cleanedIterations)
time_elapsed = np.array(cleanedTimes)
np.savetxt("total_iterations.csv", total_iterations, delimiter=",")
np.savetxt("time_elapsed.csv",time_elapsed, delimiter=",")




#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.plot(error_values)
#ax1.title.set_text('Error Value vs Iterations')
#ax1.set_xlabel('Iteration Number')
#ax1.set_ylabel('Error = ||x-x*||')
#
#ax2.plot(range(1,5))
#
#plt.show()
