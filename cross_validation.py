import select_features
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# function that does k-fold cross validation of knn algorithm. k is represented by n_splits. it selects k best features where k is represented by k_best_features. parameters for knn - n_neighbors and metric should be straight forward. metric is taken as a string which is metric identifier used by sklearn lib. random_state is used to ensure that every run data is splitted in the same way each run the random_state is the same. random_state can be set to None for random run.
# Params: DataFrame, DataFrame, int, int, int, string, int/None
# Returns: List
def run_crossvalid(X_features, Y_diagnosis, n_splits, n_neighbors, k_best_features, metric, random_state):
    scores = []

    split_algorithm = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_diagnosis):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_diagnosis.iloc[train_samples_indexes]
        Y_test = Y_diagnosis.iloc[test_samples_indexes]

        X_train_best, X_test_best = select_features.select_k_best_features_train_and_test(X_train, Y_train, X_test, k_best_features)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        knn.fit(X_train_best, Y_train.values.ravel())

        scores.append(knn.score(X_test_best, Y_test))

    return scores
