def load_data(file_path):
    """
    get file path of data file (in csv format) and returns both the data and the labels as an array
    """
    data_file = open(file_path, "r")
    data = data_file.readlines()
    samples_data = []
    samples_labels = []
    for sample in data[1:]:
        sample = sample.split(",")
        samples_data.append(sample[:-1])
        samples_labels.append(sample[-1].replace('\n', ''))
    return samples_data, samples_labels
