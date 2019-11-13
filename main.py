# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
import time
import os
import meta_project_function_definitions as meta


###=============================================================================
### Set Up All The Input Variables
###=============================================================================

# Define output path
current_directory = os.getcwd()
trials_directory = current_directory + '/trial_data'
trial_id = 1

# Define dimensions of our inputs
m = 50000
n = 100

# Define what max value of beta you want to try. Must be <= m
beta_max = 50
beta_increment = 1 #this is how much we'll increment beta at each iteration

# Make a row normalized mxn matrix A with random values from some
# normal distribution (loc = mean, scale = std_dev)
std_dev = 10
Mean = 0
A = np.random.normal(loc = Mean, scale = std_dev, size=(m,n))

# Add ones matrix to A
A = A + np.ones((m,n))

# Normalize A
A = meta.normalizeRows(A)

# Make a nx1 matrix x of all ones
x_target = np.random.normal(loc = 0, scale = 1, size=(n, 1))

# set b = A*x_target
b = np.dot(A,x_target)

# set lambda value
lam = 1

# set epsilon, the error value at which we stop
epsilon = 10**(-6)

# make random values for X_0
starting_point = np.random.normal(loc = 0, scale = 1, size=(n, 1))



###=============================================================================
### Run Trials and Export The Plots and Data
###=============================================================================

trial_data = meta.run_SKM_vary_beta(beta_max, beta_increment,x_target,A,b,lam,m,epsilon,starting_point)

figure, optimal_beta = meta.plot_beta_v_time_results(trial_data)

meta.save_time_vs_beta_inputs_and_results(trials_directory,trial_id,figure,trial_data,A,x_target,starting_point)
