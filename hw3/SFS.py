from tools import load_data
from KNN1 import KNN_classifier


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

knn = KNN_classifier(9)
knn.train(train_data, train_labels)

best_features = []
best_accuracy = 0
for number_of_features in range(len(test_data[0])):
    local_best_accuracy = best_accuracy
    local_best_feature_set = best_features
    for feature_index in range(len(test_data[0])):
        if feature_index not in best_features:
            feature_set = best_features.copy()
            feature_set.append(feature_index)
            predicted = knn.predict(test_data, feature_set)
            accuracy = knn.accuracy(predicted, test_labels)
            if accuracy > local_best_accuracy:
                local_best_feature_set = feature_set.copy()
                local_best_accuracy = accuracy
    if local_best_accuracy >= best_accuracy:
        best_features = local_best_feature_set.copy()
        best_accuracy = local_best_accuracy
    else:
        break

print(best_features)

"""
best feature set: 
[1, 7, 6, 5, 2]
it's accuracy: 0.82
"""
