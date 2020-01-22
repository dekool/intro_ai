from tools import load_data
from KNN1 import KNN_classifier
import itertools


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

knn = KNN_classifier(9)
knn.train(train_data, train_labels)

best_features = []
best_accuracy = 0
for feature_set in itertools.chain.from_iterable(itertools.combinations(range(len(test_data[0])), n) for n in range(len(test_data[0])+1)):
    predicted = knn.predict(test_data, feature_set)
    accuracy = knn.accuracy(predicted, test_labels)
    if accuracy > best_accuracy:
        best_features = feature_set
        best_accuracy = accuracy

print(list(best_features))

"""
best feature set: 
(1, 2, 5, 6, 7)
it's accuracy: 0.82
"""