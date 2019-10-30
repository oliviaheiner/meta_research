import random
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time


def SKM(x,A,b,beta,lam,m):
    sample_of_indices = random.sample(range(0, m), beta)
    distances_to_hyperplane = [((abs((np.dot(A[j,:],x)-b[j])))[0]) for j in sample_of_indices]
    t_k = sample_of_indices[distances_to_hyperplane.index(max(distances_to_hyperplane))]
    a_t_k = A[t_k:(t_k+1),:]
    x = x - lam*(((np.dot(a_t_k,x))-b[t_k])/((np.linalg.norm(a_t_k))**2))*np.transpose(a_t_k)
    return x



def run_SKM_vary_beta(beta_max, beta_increment,x_target,A,b,lam,m,epsilon,starting_point):
    # make a dataframe to store all the data
    trial_data = pd.DataFrame(columns=['Beta', 'Run Time', 'Total Iterations'])
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
            x = SKM(x,A,b,beta,lam,m)
            error = LA.norm(x - x_target)

        time_elapsed = time.time() - start_time
        trial_data = trial_data.append({'Beta':beta,'Run Time':time_elapsed,'Total Iterations':iterations},ignore_index=True)
        print("beta = ", beta, " , iterations = ", iterations, "time elapsed = ", time_elapsed)
        beta += beta_increment
    return trial_data




def save_time_vs_beta_inputs_and_results(trials_directory,trial_id,figure,trial_data,A,x_target,starting_point):
    single_trial_dir = trials_directory + '/trial_' + str(trial_id)

    file_path = single_trial_dir + '/time_vs_beta.pdf'
    figure.savefig(file_path, bbox_inches='tight')

    file_path = single_trial_dir + '/beta_iter_time.csv'
    trial_data.to_csv(file_path,index=False)

    file_path = single_trial_dir + '/matrix_A.csv'
    np.savetxt(file_path, A, delimiter=",")

    file_path = single_trial_dir + '/x_target.csv'
    np.savetxt(file_path, x_target, delimiter=",")

    file_path = single_trial_dir + '/x_0.csv'
    np.savetxt(file_path, starting_point, delimiter=",")




def plot_beta_v_time_results(trial_data):
    optimal_run_time_index = trial_data[['Run Time']].idxmin()
    optimal_beta = trial_data['Beta'][optimal_run_time_index].item()
    f = plt.figure()
    plt.plot(trial_data['Beta'],trial_data['Run Time'])
    plt.axvline(x=optimal_beta, linestyle = '--', c = 'r', linewidth = '1')
    plt.xlim(1,trial_data['Beta'].max())
    plt.xlabel('Beta')
    plt.ylabel('Run Time')
    plt.xticks(list(plt.xticks()[0]) + [optimal_beta])
    return [f,optimal_beta]
