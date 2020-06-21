import numpy as np
import pandas as pd

def run(benchmark_data, no_of_crossvalid_runs, no_of_folds):
    no_of_benchmark_scores = len(benchmark_data.index)
    result = pd.DataFrame(np.zeros((no_of_benchmark_scores, no_of_benchmark_scores)))

    for i in range(0, no_of_benchmark_scores):
        matrix_i = benchmark_data.at[i, 'Score matrix']
        for j in range(i + 1, no_of_benchmark_scores):
            matrix_j = benchmark_data.at[j, 'Score matrix']

            t_value = calculate_t(matrix_i, matrix_j, no_of_crossvalid_runs, no_of_folds)

            result.at[i, j] = t_value
            result.at[j, i] = t_value*-1

    print(result)

def calculate_t(matrix_i, matrix_j, no_of_crossvalid_runs, no_of_folds):
    matrix_s = matrix_i - matrix_j
    avg_matrix_s = np.mean(matrix_s)
    return 420 # to be implemented
