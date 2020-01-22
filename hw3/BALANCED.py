from sklearn import tree
import sklearn
from tools import load_data

train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

positive_counter = 0
for label in train_labels:
    if label == '1':
        positive_counter += 1

# balance the data
balanced_train_data = []
balanced_train_labels = []
for i in range(len(train_data)):
    if train_labels[i] == '1':
        balanced_train_data.append(train_data[i])
        balanced_train_labels.append(train_labels[i])
    elif positive_counter > 0:  # enter negative sample only if positive counter in positive
        balanced_train_data.append(train_data[i])
        balanced_train_labels.append(train_labels[i])
        positive_counter -= 1

# we enter constant random seed to avoid different results each run
id3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=2)
id3 = id3.fit(train_data, train_labels)
predicted = id3.predict(test_data)

# we enter constant random seed to avoid different results each run
id3_balanced = tree.DecisionTreeClassifier(criterion="entropy", random_state=2)
id3_balanced = id3_balanced.fit(balanced_train_data, balanced_train_labels)
predicted_balanced = id3_balanced.predict(test_data)

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
print("confusion matrix:")
print([[tp, fp], [fn, tn]])
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted_balanced).ravel()
print("confusion matrix of balanced data:")
print([[tp, fp], [fn, tn]])

def error(predicted, test_labels):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
    return 4*fn + fp

error_normal = error(predicted, test_labels)
error_balanced = error(predicted_balanced, test_labels)
print("error normal: " + str(error_normal))
print("error balanced: " + str(error_balanced))
