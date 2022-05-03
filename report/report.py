from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path


def create_model_report():
    model_path = os.path.join('training/model/trained_model.joblib')
    model = joblib.load(model_path)

    test_data = pd.read_pickle('cache/data_preprocessed_test.pkl')

    print('COMPOSING REPORT')
    x_test = test_data.drop(columns='Survived')
    y_true = test_data["Survived"].to_numpy()
    y_pred = model.predict(x_test)

    labels = ['Survived', 'Not survived']

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    Path('report/report').mkdir(parents=True, exist_ok=True)
    with open('report/report/report.json', 'w') as f:
        f.write(json.dumps(report, indent=4))

    metrics = {'weighted_f1_test': report['weighted avg']['f1-score'],
               'accuracy_test': report['accuracy']}

    with open('metrics.json', 'w') as f:
        f.write(json.dumps(metrics, indent=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print('Confusion matrix:')
    print(cm)

    plot_confusion_matrix(model, x_test, y_true)
    plt.tight_layout()
    img_path = 'report/confusion_matrix.png'
    plt.savefig(img_path, dpi=120)


if __name__ == '__main__':
    print('Compose report...')
    create_model_report()
