from sklearn import metrics
import numpy as np
from tools import load_data


class KNN_classifier():
    def __init__(self, k):
        self.minimums = []
        self.maximums = []
        self.train_data = []
        self.train_labels = []
        self.k = k

    def __normalize_data(self, data):
        """
        normalize the given data using (feature-minimum)/(maximum-minimum)
        :return: normalized data
        """
        for i in range(len(data[0])):
            minimum = np.inf
            maximum = -np.inf
            for sample in data:
                minimum = min(minimum, float(sample[i]))
                maximum = max(maximum, float(sample[i]))
            self.minimums.append(minimum)
            self.maximums.append(maximum)
        normalized_data = []
        for sample in data:
            new_sample = []
            for i in range(len(sample)):
                new_sample.append((float(sample[i]) - self.minimums[i]) / (self.maximums[i] - self.minimums[i]))
            normalized_data.append(new_sample)
        return normalized_data

    def train(self, train_data, train_labels):
        """
        in KNN training is just saving the data (normalized)
        """
        self.train_labels = train_labels
        self.train_data = self.__normalize_data(train_data)

    def calculate_distance(self, data, features_list=None):
        """
        calculate the distance from each sample in train data using L2
        feature_list (optional value) - the indexes of the features to use to calculate the distance. by default -
        use all the features
        :return dict of sample index to distance
        """
        if features_list is None:
            features_list = range(len(data))
        distances = {}
        for sample_index in range(len(self.train_data)):  # for each train sample
            distance = 0
            for feature_index in features_list:  # for each feature
                # first - normalize
                normalized_feature = (float(data[feature_index]) - self.minimums[feature_index]) / (
                            self.maximums[feature_index] - self.minimums[feature_index])
                distance += ((normalized_feature - self.train_data[sample_index][feature_index]) ** 2)
            distances[sample_index] = np.emath.sqrt(distance)
        return {example: distance for example, distance in sorted(distances.items(), key=lambda item: item[1])}

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
            if positive_count > (self.k/2):
                predict.append('1')
            else:
                predict.append('0')
        return predict

    def error_w(self, predicted_lables, real_labels):
        tn, fp, fn, tp = metrics.confusion_matrix(real_labels, predicted_lables).ravel()
        return 4 * fn + fp

    def accuracy(self, predicted_lables, real_labels):
        tn, fp, fn, tp = metrics.confusion_matrix(real_labels, predicted_lables).ravel()
        return (tn + tp) / (tn + fp + fn + tp)

# we need a main function, because we import this class in other files
if __name__ == '__main__':
    train_data, train_labels = load_data("./train.csv")
    test_data, test_labels = load_data("./test.csv")
    knn = KNN_classifier(9)
    knn.train(train_data, train_labels)
    predicted = knn.predict(test_data)
    tn, fp, fn, tp = metrics.confusion_matrix(test_labels, predicted).ravel()
    print([[tp, fp], [fn, tn]])
    print("error: " + str(knn.error_w(predicted, test_labels)))
