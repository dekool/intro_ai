from tools import load_data
from KNN2 import KNN2_classifier
import matplotlib.pyplot as plt


train_data, train_labels = load_data("./train.csv")
test_data, test_labels = load_data("./test.csv")
knn = KNN2_classifier(9)
knn.train(train_data, train_labels)
predicted = knn.predict(test_data)
errors = []
for k in [1 ,3 ,9, 27]:
    temp_knn = KNN2_classifier(k)
    temp_knn.train(train_data, train_labels)
    temp_predicted = temp_knn.predict(test_data)
    errors.append(temp_knn.error_w(temp_predicted, test_labels))

plt.plot([1, 3, 9, 27], errors)
plt.ylabel('error_w')
plt.xlabel('k')
plt.show()

"""
positive samples: 197
negative samples: 371
"""
