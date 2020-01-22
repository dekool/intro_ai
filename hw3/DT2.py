from sklearn import tree
import sklearn
from tools import load_data


def error(predicted, test_labels):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
    return 4*fn + fp


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

# we enter constant random seed to avoid different results each run
id3_9 = tree.DecisionTreeClassifier(criterion="entropy", class_weight={'0': (1/5), '1': (4/5)}, random_state=0, min_samples_split=9)
id3_9 = id3_9.fit(train_data, train_labels)
predicted_9 = id3_9.predict(test_data)

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted_9).ravel()
print([[tp, fp], [fn, tn]])
