import cross_validation
import pandas as pd
import numpy as np
import random

# function that takes whole set of data (two DataFrames - X and Y) and runs cross_validation with diffrent parameters as needed in the project description. returns DataFrame with params and scores
# Params: DataFrame, DataFrame
# Returns: DataFrame
def run(X_features, Y_diagnosis):
    columns=["n_splits", "metric", "k_best_features", "no_of_n_neighbors", "Score matrix", "Average"]
    run_results = pd.DataFrame(columns=columns)

    metrics = ["euclidean", "manhattan"]
    max_best_features = 35
    n_neighbors = [1, 5, 10]
    no_of_crossvalid_runs = 2
    no_of_folds = 5

    random_states = [420, 2137] # needs to be the size of no_of_crossvalid_runs
    check_random_states(random_states, no_of_crossvalid_runs)

    for metric in metrics:
        for k_best_features in range(1, max_best_features + 1): # 1 to max_best_features
            for no_of_n_neighbors in n_neighbors:
                score_matrix = np.zeros((no_of_crossvalid_runs, no_of_folds), float)

                for run in range(no_of_crossvalid_runs):
                    score_matrix[run] = cross_validation.run_crossvalid(X_features, Y_diagnosis, no_of_folds, no_of_n_neighbors, k_best_features, metric, random_states[run])

                average = np.mean(score_matrix)

                run_results = run_results.append({"n_splits" : no_of_folds, "metric" : metric, "k_best_features" : k_best_features, "no_of_n_neighbors" : no_of_n_neighbors, "Score matrix" : score_matrix, "Average" : average}, ignore_index=True)

    run_results.sort_values(by=['Average'], ascending=False, inplace=True)

    return run_results

def check_random_states(random_states, no_of_crossvalid_runs):
    # if for below checks if random_states is size of no_of_crossvalid_runs and if not fills with needed data
    if (len(random_states) < no_of_crossvalid_runs):
        random.seed()
        for i in range(len(random_states), no_of_crossvalid_runs):
            random_states.append(random.randint(1, 100000))
