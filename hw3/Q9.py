from sklearn import tree
import sklearn
from tools import load_data
import matplotlib.pyplot as plt
import numpy as np


def error(predicted, test_labels):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
    return 4*fn + fp


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

# we enter constant random seed to avoid different results each run
id3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
id3 = id3.fit(train_data, train_labels)
predicted = id3.predict(test_data)

prob = [0.05, 0.1, 0.2]
predictions_arr = []
for p in prob:
    new_predicted = []
    for data in predicted:
        if data == '0':
            new_predicted.append(np.random.choice((['0', '1']), p=[1-p, p]))
        else:
            new_predicted.append(data)
    predictions_arr.append(new_predicted)

plt.plot(prob, [error(predictions_arr[0], test_labels), error(predictions_arr[1], test_labels),
                error(predictions_arr[2], test_labels)])
plt.ylabel('error_W')
plt.xlabel('p')
plt.ylim(bottom=120, top=140)
plt.show()
