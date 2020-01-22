from sklearn import tree
import sklearn
from tools import load_data


def error(predicted, test_labels):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
    return 4*fn + fp


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

id3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
id3 = id3.fit(train_data, train_labels)
predicted = id3.predict(test_data)

id3_27 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=27)
id3_27 = id3_27.fit(train_data, train_labels)
predicted_27 = id3_27.predict(test_data)

error_not_cut = error(predicted, test_labels)
error_cut = error(predicted_27, test_labels)
print("error cut: " + str(error_cut))
print("error not cut: " + str(error_not_cut))