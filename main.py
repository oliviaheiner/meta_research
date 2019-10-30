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
m = 150000
n = 100

# Define what max value of beta you want to try. Must be <= m
beta_max = 50
beta_increment = 1 #this is how much we'll increment beta at each iteration

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

# make random values for X_0
starting_point = np.random.normal(loc = 700, scale = 2000, size=(n, 1))



###=============================================================================
### Run Trials and Export The Plots and Data
###=============================================================================

trial_data = meta.run_SKM_vary_beta(beta_max, beta_increment,x_target,A,b,lam,m,epsilon,starting_point)

figure, optimal_beta = meta.plot_beta_v_time_results(trial_data)

meta.save_time_vs_beta_inputs_and_results(trials_directory,trial_id,figure,trial_data,A,x_target,starting_point)
