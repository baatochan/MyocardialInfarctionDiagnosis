import cross_validation
import pandas as pd

# function that takes whole set of data (two DataFrames - X and Y) and runs cross_validation with diffrent parameters as needed in the project description. returns DataFrame with params and scores
# Params: DataFrame, DataFrame
# Returns: DataFrame
def run(X_features, Y_diagnosis):
    columns=["n_splits", "metric", "k_best_features", "n_neighbors", "Crossvalid run", "Scores"]
    run_results = pd.DataFrame(columns=columns)

    for metric in ["euclidean", "manhattan"]:
        for k_best_features in range(1, 21): # 1 to 20
            for n_neighbors in [1, 5, 10]:
                for run in range(5):
                    random_states = [69, 420, 911, 1004, 2137]
                    score = cross_validation.run_crossvalid(X_features, Y_diagnosis, 2, n_neighbors, k_best_features, metric, random_states[run])

                    run_results = run_results.append({"n_splits" : 2, "metric" : metric, "k_best_features" : k_best_features, "n_neighbors" : n_neighbors, "Crossvalid run" : run, "Scores" : score}, ignore_index=True)

    return run_results
