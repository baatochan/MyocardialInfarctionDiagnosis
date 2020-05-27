from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# function that selects k best features using chi2 algorithm from X_features DataFrame provided as the parameter and returns another DataFrame which contains all the samples with only those selected features. DataFrame preserves feature names labels.
# Params: DataFrame, DataFrame, int
# Returns: DataFrame
def select_k_best_features(X_features, Y_diagnosis, k_best_features):
    # create and fit selector
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X_features, Y_diagnosis)

    # get columns to keep and create new DataFrame with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_best_features = X_features.iloc[:, new_features]

    return X_best_features

# function that is modification of the above function. selects k best features using chi2 algorithm from X_train (needs Y_train to do it as well). X_test is provided to this function only because this DataFrame needs to be stripped to the same features as the one selected from training set. test data ARE NOT USED for feature selection algorithm. the result of this algoritm is applied on this data set for compability with training data. int represents number of best features to select. function returns two DataFrames which are stripped train and test DataFrames respectively.
# Params: DataFrame, DataFrame, DataFrame, int
# Returns: (DataFrame, DataFrame)
def select_k_best_features_train_and_test(X_train, Y_train, X_test, k_best_features):
    # create and fit selector
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X_train, Y_train)

    # get columns to keep and create new DataFrame with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_train_best_features = X_train.iloc[:, new_features]
    # create second DataFrame from test set which contains only selected features
    X_test_best_features = X_test.iloc[:, new_features]

    return (X_train_best_features, X_test_best_features)

# function that creates feture ranking using chi2 algorithm. takes set of all samples that are used for ranking creation. returns DataFrame which contains the ranking in the form of 3 colums - feature name, chi2 value and p value. it is then sorted from the best to the worst features in the given set.
# Params: DataFrame, DataFrame
# Returns: DataFrame
def create_feature_ranking(X_features, Y_diagnosis):
    (chi, pval) = chi2(X_features, Y_diagnosis)

    result = pd.DataFrame(X_features.columns, columns=['Feature name'])
    result["chi"] = chi
    result["pval"] = pval

    result.sort_values(by=['chi'], ascending=False, inplace=True)

    return result
