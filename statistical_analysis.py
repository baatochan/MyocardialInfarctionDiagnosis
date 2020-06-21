import numpy as np
import pandas as pd
import math

def run(benchmark_data, no_of_crossvalid_runs, no_of_folds):
    no_of_benchmark_scores = len(benchmark_data.index)
    result = pd.DataFrame(np.zeros((no_of_benchmark_scores, no_of_benchmark_scores)))

    for i in range(0, no_of_benchmark_scores):
        matrix_i = benchmark_data.at[i, 'Score matrix']
        for j in range(i + 1, no_of_benchmark_scores):
            matrix_j = benchmark_data.at[j, 'Score matrix']

            t_value = calculate_t(matrix_i, matrix_j, no_of_crossvalid_runs, no_of_folds)

            result.at[i, j] = t_value
            result.at[j, i] = t_value * -1

    result_bool = check_critical_value(result)
    sum_rows(result_bool)

    result_bool.sort_values(by=['Sum'], ascending=False, inplace=True)

    return result_bool

def calculate_t(matrix_i, matrix_j, no_of_crossvalid_runs, no_of_folds):
    no_of_crossvalid_runs = matrix_i.shape[0]
    no_of_folds = matrix_i.shape[1]

    if ((matrix_i == matrix_j).all()):
        return 0;

    matrix_s = matrix_i - matrix_j
    avg_matrix_s = np.mean(matrix_s)
    t_value = 0

    for i in range(0, no_of_crossvalid_runs):
        for j in range(0, no_of_folds):
            sigma = matrix_s[i, j]
            sigma = sigma - avg_matrix_s
            sigma = sigma * sigma
            sigma = sigma / ((no_of_folds * no_of_crossvalid_runs) - 1)
            t_value = t_value + sigma

    t_value = t_value * ((1 / (no_of_folds * no_of_crossvalid_runs)) + (1 / no_of_folds))
    t_value = math.sqrt(t_value)
    t_value = avg_matrix_s / t_value

    return t_value

def check_critical_value(matrix):
    critical_value = 2.2622

    matrix_size = len(matrix.index)
    result = pd.DataFrame(np.zeros((matrix_size, matrix_size)))

    for i in range(0, matrix_size):
        for j in range(0, matrix_size):
            if (matrix.at[i, j] > critical_value):
                result.at[i, j] = 1
            elif (matrix.at[i, j] < (-1 * critical_value)):
                result.at[i, j] = -1
            else:
                result.at[i, j] = 0

    return result

def sum_rows(matrix):
    result = [];
    matrix_size = len(matrix.index)

    for i in range(0, matrix_size):
        sum = 0;
        for j in range(0, matrix_size):
            sum = sum + matrix.at[i, j]

        result.append(sum)

    matrix["Sum"] = result
