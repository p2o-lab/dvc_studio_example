import pandas as pd


def import_data():
    data_file = 'data/raw_data.csv'
    print('Reading data from %s' % data_file)
    df = pd.read_csv(data_file)
    print(df.head())

    print('DATA SPLIT')
    dataset_length = len(df)

    train_dataset_length = int(dataset_length * 0.8)

    df_train = df.iloc[:train_dataset_length, :]
    df_test = df.iloc[train_dataset_length + 1:, :]

    df_train.to_pickle('cache/data_train.pkl')
    df_test.to_pickle('cache/data_test.pkl')


if __name__ == '__main__':
    import_data()
