from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

def select_k_best_features(X_features, Y_diagnosis, k):
    # Create and fit selector
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k)
    select_k_best_classifier.fit(X_features, Y_diagnosis)

    # Get columns to keep and create new dataframe with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_best_features = X_features.iloc[:, new_features]

    return X_best_features

def select_k_best_features_train_and_test(X_train, Y_train, X_test, k):
    # Create and fit selector
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k)
    select_k_best_classifier.fit(X_train, Y_train)

    # Get columns to keep and create new dataframe with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_train_best_features = X_train.iloc[:, new_features]
    X_test_best_features = X_test.iloc[:, new_features]

    return (X_train_best_features, X_test_best_features)

def create_feature_ranking(X_features, Y_diagnosis):
    (chi, pval) = chi2(X_features, Y_diagnosis)

    result = pd.DataFrame(X_features.columns, columns=['Feature name'])
    result["chi"] = chi
    result["pval"] = pval

    result.sort_values(by=['chi'], ascending=False, inplace=True)

    return result
