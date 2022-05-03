import pandas as pd
from sklearn import svm
import joblib
from pathlib import Path
import yaml


def train_model():
    train_data = pd.read_pickle('cache/data_preprocessed_train.pkl')
    print('FEATURE SELECTION')
    x_train = train_data.drop(columns="Survived")
    y_train = train_data["Survived"]

    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)

    print('MODEL TRAINING')
    svm_model = svm.SVC(C=params['model']['c-value'])
    svm_model.fit(x_train, y_train)

    Path('training/model').mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_model, 'training/model/trained_model.joblib')


if __name__ == '__main__':
    print('Train model...')
    train_model()
