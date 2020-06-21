import load_data
import select_features
import cross_validation
import model_benchmark
import statistical_analysis

no_of_crossvalid_runs = 2
no_of_folds = 5

# load list of features from file. why? because i thought that putting this list in the txt file would be more useful than hardcoing into the array in python code.
features = load_data.load_features()

# scan all files (names hardcoded) from data dir and loads all samples into one big array (actually it's DataFrame, but whatever).
(X_features, Y_diagnosis) = load_data.load_data_from_files(features)

# creates and prints feature ranking using all samples.
#feature_ranking = select_features.create_feature_ranking(X_features, Y_diagnosis)
#print(feature_ranking)

# run and print score of one cross_validation with sample params.
#score = cross_validation.run_crossvalid(X_features, Y_diagnosis, 2, 3, 5, 'manhattan', 420)
#print(score)

# run a function that tests diffrent set of parameters for knn cross_validation
run_results = model_benchmark.run(X_features, Y_diagnosis, no_of_crossvalid_runs, no_of_folds)
print(run_results)

(analysis_result, benchmark_order) = statistical_analysis.run(run_results, no_of_crossvalid_runs, no_of_folds)
print(analysis_result)

run_results.sort_index(inplace=True)
run_results["Wage"] = benchmark_order
run_results.sort_values(by=['Wage'], ascending=False, inplace=True)
print(run_results)
