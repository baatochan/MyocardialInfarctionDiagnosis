import select_features
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def run_crossvalid(X_features, Y_diagnosis, n_neighbors, n_best_features, metric):
    split_algorithm = StratifiedKFold(n_splits=5, random_state=420, shuffle=True)
    for train_index, test_index in split_algorithm.split(X_features, Y_diagnosis):
        X_train = X_features.iloc[train_index]
        X_test = X_features.iloc[test_index]
        Y_train = Y_diagnosis.iloc[train_index]
        Y_test = Y_diagnosis.iloc[test_index]

        X_train_best, X_test_best = select_features.select_k_best_features_train_and_test(X_train, Y_train, X_test, n_best_features)

        #print(Y_train)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(X_train_best, Y_train.values.ravel())

        score = knn.score(X_test_best, Y_test)
        print(score)
