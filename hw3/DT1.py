from sklearn import tree
import sklearn
from tools import load_data


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

# we enter constant random seed to avoid different results each run
id3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
id3 = id3.fit(train_data, train_labels)
predicted = id3.predict(test_data)

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
print([[tp, fp], [fn, tn]])
