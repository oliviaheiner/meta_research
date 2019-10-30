# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
import time
import os
import meta_project_function_definitions as meta

# Define output path
current_directory = os.getcwd()
trial_directory = current_directory + '/trial_data/trial_1'

# Define dimensions of our inputs
m = 150000
n = 100

# Define what max value of beta you want to try. Must be <= m
beta_max = 50

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
trial_data = pd.DataFrame(columns=['Beta', 'Run Time', 'Total Iterations'])

# make random values for X_0
starting_point = np.random.normal(loc = 700, scale = 2000, size=(n, 1))

# set beta value
beta = 1
while(beta<=beta_max):
    # set x to be starting_point vector
    x = starting_point

    # calculate initial error value
    error = LA.norm(x - x_target)

    # record start time
    start_time = time.time()

    # make a variable to count iterations
    iterations = 0

    while(error>epsilon):
        iterations += 1
        x = meta.SKM(x,A,b,beta,lam,m)
        error = LA.norm(x - x_target)

    time_elapsed = time.time() - start_time
    trial_data = trial_data.append({'Beta':beta,'Run Time':time_elapsed,'Total Iterations':iterations},ignore_index=True)
    print("beta = ", beta, " , iterations = ", iterations, "time elapsed = ", time_elapsed)
    beta += 1

optimal_run_time_index = trial_data[['Run Time']].idxmin()
optimal_beta = trial_data['Beta'][optimal_run_time_index].item()

f = plt.figure()
plt.plot(trial_data['Beta'],trial_data['Run Time'])
plt.axvline(x=optimal_beta, linestyle = '--', c = 'r', linewidth = '1')
plt.xlim(1,beta_max)
plt.xlabel('Beta')
plt.ylabel('Run Time')
plt.xticks(list(plt.xticks()[0]) + [optimal_beta])
file_path = trial_directory + '/time_vs_beta.pdf'
f.savefig(file_path, bbox_inches='tight')


file_path = trial_directory + '/beta_iter_time.csv'
trial_data.to_csv(file_path,index=False)

file_path = trial_directory + '/matrix_A.csv'
np.savetxt(file_path, A, delimiter=",")

file_path = trial_directory + '/x_target.csv'
np.savetxt(file_path, x_target, delimiter=",")

file_path = trial_directory + '/x_0.csv'
np.savetxt(file_path, starting_point, delimiter=",")
