import pandas as pd
import yaml


def import_data():
    data_file = 'data/raw_data.tab'
    print('Reading data from %s' % data_file)
    df = pd.read_csv(data_file, sep='\t')
    print(df.head())

    print('DATA SPLIT')
    dataset_length = len(df)

    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    train_dataset_length = int(dataset_length * params['data']['split_value'])
    print(f"Split value = {params['data']['split_value']}")

    df_train = df.iloc[:train_dataset_length, :]
    df_test = df.iloc[train_dataset_length + 1:, :]

    df_train.to_pickle('cache/data_train.pkl')
    df_test.to_pickle('cache/data_test.pkl')


if __name__ == '__main__':
    import_data()
