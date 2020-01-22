from sklearn import tree
import sklearn
from tools import load_data
import matplotlib.pyplot as plt
import numpy as np


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")

#################################################
############# Question 2 ########################
#################################################

# we enter constant random seed to avoid different results each run
id3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
id3 = id3.fit(train_data, train_labels)
predicted = id3.predict(test_data)

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
print([[tp, fp], [fn, tn]])

#################################################
############# Question 3 ########################
#################################################


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


#################################################
############# Question 7 b ######################
#################################################

def error(predicted, test_labels):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, predicted).ravel()
    return 4*fn + fp

error_not_cut = error(predicted, test_labels)
error_cut = error(predicted_27, test_labels)
print("error cut: " + str(error_cut))
print("error not cut: " + str(error_not_cut))

#################################################
############# Question 9 c ######################
#################################################

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

#plt.plot(prob, [error(predictions_arr[0], test_labels), error(predictions_arr[1], test_labels),
#                error(predictions_arr[2], test_labels)])
#plt.ylabel('error_W')
#plt.xlabel('p')
#plt.ylim(bottom=120, top=140)
#plt.show()
