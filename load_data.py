import pandas as pd

# function that reads features names from one file and returns it as a list
# Params: None
# Returns: List
def load_features():
    features = pd.read_csv('data/features.txt', header=None)
    features = features[0].tolist()

    return features

# function that scans all the data provided in the data dir and putting in together into one big array of samples. it uses DataFrame so it keeps labels for features names and number labels for every sample. returns two DataFrames - one with samples and all the features, the other one with diagnosis. row with index X in one DataFrame corresponds to row in the other DataFrame with the same index.
# Params: List
# Returns: (DataFrame, DataFrame)
def load_data_from_files(features):
    X_features = pd.DataFrame(columns=features)
    Y_diagnosis = pd.DataFrame(columns=["Diagnosis"])

    i = 1
    for file in ["inne", "ang_prect", "ang_prct_2", "mi", "mi_np"]:
        X_current_file = pd.read_csv('data/' + file + '.txt', sep='\t', header=None)
        X_current_file = X_current_file.transpose() # one sample in the file is represented by the column, not the row, so 2d array needs to be transposed
        X_current_file.columns = features
        X_features = pd.concat([X_features, X_current_file], ignore_index=True)

        diagnosis = [i] * len(X_current_file.index) # diagnosis is represented by integer ranged from 1 to 5.
        Y_current_file = pd.DataFrame(diagnosis)
        Y_current_file.columns = ["Diagnosis"]
        Y_diagnosis = pd.concat([Y_diagnosis, Y_current_file], ignore_index=True)

        i += 1

    # DataFrames needs to be converted to numeric as they are loaded as text by default. provided samples consistes of only numbers so this conversion should be safe.
    X_features = X_features.apply(pd.to_numeric)
    Y_diagnosis = Y_diagnosis.apply(pd.to_numeric)

    return (X_features, Y_diagnosis)
