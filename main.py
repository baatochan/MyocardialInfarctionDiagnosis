import load_data
import select_features

features = load_data.load_features()
(X_features, Y_diagnosis) = load_data.load_data_from_files(features)

X_best_features = select_features.select_k_best_features(X_features, Y_diagnosis, 20)

feature_ranking = select_features.create_feature_ranking(X_features, Y_diagnosis)

print feature_ranking
