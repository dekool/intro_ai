from sklearn import metrics
from tools import load_data
from KNN1 import KNN_classifier
import matplotlib.pyplot as plt


class KNN2_classifier(KNN_classifier):
    def predict(self, new_data, features_list=None):
        """
        estimate the labels of the given data
        feature_list (optional value) - the indexes of the features to use to calculate the distance. by default -
        use all the features
        :return array of predicted labels
        """
        predict = []
        for data in new_data:
            distances_list = self.calculate_distance(data, features_list)
            distances_list = list(distances_list.keys())[: self.k]
            positive_count = 0
            for sample_index in distances_list:
                if self.train_labels[sample_index] == '1':
                    positive_count += 1
            if 4*positive_count > (self.k/2):
                predict.append('1')
            else:
                predict.append('0')
        return predict


if __name__ == '__main__':
    train_data, train_labels = load_data("./train.csv")
    test_data, test_labels = load_data("./test.csv")
    knn = KNN2_classifier(9)
    knn.train(train_data, train_labels)
    predicted = knn.predict(test_data)
    tn, fp, fn, tp = metrics.confusion_matrix(test_labels, predicted).ravel()
    print([[tp, fp], [fn, tn]])
