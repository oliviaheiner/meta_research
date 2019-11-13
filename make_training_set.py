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

# Define dimensions of our inputs
m = 50000
n = 100

# Define what max value of beta you want to try. Must be <= m
beta_max = 50
beta_increment = 1 #this is how much we'll increment beta at each iteration

# Define Lambda and epsilon
lam = 1
epsilon = 10**(-6)

# Make a row normalized mxn matrix A with random values from some
# normal distribution (loc = mean, scale = std_dev)
std_dev = 100
number_of_matrices_to_test = 10

avg_opt_betas,std_opt_betas = meta.std_dev_to_optimal_beta(std_dev = std_dev,shape = [m,n],beta_max = beta_max,beta_increment = beta_increment,Lambda = lam,epsilon = epsilon,num_trials = number_of_matrices_to_test)
