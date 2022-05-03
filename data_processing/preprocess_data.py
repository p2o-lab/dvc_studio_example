import pandas as pd


def preprocess_data(path_to_data, output_path):
    data = pd.read_pickle(path_to_data)
    data["Sex"].replace({"male": 0.0, "female": 1.0}, inplace=True)
    data.rename(columns={"Sex": "female"}, inplace=True)
    data_filtered = data[["Pclass", "female", "Fare", "Survived"]]
    data_filtered.to_pickle(output_path)


def preprocess_train_data():
    print('Preprocess train data...')
    preprocess_data('cache/data_train.pkl', 'cache/data_preprocessed_train.pkl')


def preprocess_test_data():
    print('Preprocess test data...')
    preprocess_data('cache/data_test.pkl', 'cache/data_preprocessed_test.pkl')


if __name__ == '__main__':
    preprocess_train_data()
    preprocess_test_data()
