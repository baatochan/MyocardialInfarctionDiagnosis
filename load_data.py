import pandas as pd

def load_features():
    features = pd.read_csv('data/features.txt', header=None)
    features = features[0].tolist()

    return features

def load_data_from_files(features):
    X_features = pd.DataFrame(columns = features)
    Y_diagnosis = pd.DataFrame(columns = ["Diagnosis"])

    i = 1
    for file in ["inne", "ang_prect", "ang_prct_2", "mi", "mi_np"]:
        tempX = pd.read_csv('data/' + file + '.txt', sep='\t', header=None)
        tempX = tempX.transpose()
        tempX.columns = features
        X_features = pd.concat([X_features, tempX], ignore_index=True)

        diagnosis = [i] * len(tempX.index)
        tempY = pd.DataFrame(diagnosis)
        tempY.columns = ["Diagnosis"]
        Y_diagnosis = pd.concat([Y_diagnosis, tempY], ignore_index=True)

        i += 1

    X_features = X_features.apply(pd.to_numeric)
    Y_diagnosis = Y_diagnosis.apply(pd.to_numeric)

    return (X_features, Y_diagnosis)
