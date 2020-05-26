import load_data
import select_features
import cross_validation

features = load_data.load_features()
(X_features, Y_diagnosis) = load_data.load_data_from_files(features)

#X_best_features = select_features.select_k_best_features(X_features, Y_diagnosis, 20)

#feature_ranking = select_features.create_feature_ranking(X_features, Y_diagnosis)

#print(X_best_features)

cross_validation.run_crossvalid(X_features, Y_diagnosis, 3, 5)
