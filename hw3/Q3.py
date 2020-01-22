from sklearn import tree
import sklearn
from tools import load_data
import matplotlib.pyplot as plt

train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

def calculate_accuracy(predicted, test_labels):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
    return (tn + tp) / (tn + fp + fn + tp)


id3_3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=3)
id3_3 = id3_3.fit(train_data, train_labels)
predicted_3 = id3_3.predict(test_data)

id3_9 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=9)
id3_9 = id3_9.fit(train_data, train_labels)
predicted_9 = id3_9.predict(test_data)

id3_27 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=27)
id3_27 = id3_27.fit(train_data, train_labels)
predicted_27 = id3_27.predict(test_data)

acc_3 = calculate_accuracy(predicted_3, test_labels)
acc_9 = calculate_accuracy(predicted_9, test_labels)
acc_27 = calculate_accuracy(predicted_27, test_labels)

plt.plot([3, 9, 27], [acc_3, acc_9, acc_27])
plt.ylabel('accuracy')
plt.xlabel('minimum cut')
plt.ylim(bottom=0.6, top=0.8)
plt.show()