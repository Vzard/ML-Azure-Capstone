
from azureml.data.dataset_factory import TabularDatasetFactory
import pandas as pd
from azureml.core.run import Run
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib


def main():

    url_path = "https://raw.githubusercontent.com/Vzard/ML-Azure-Capstone/master/StrokeData_V1.csv"
    ds = TabularDatasetFactory.from_delimited_files(url_path)
    stroke_df = ds.to_pandas_dataframe().dropna()

    object_columns = stroke_df.select_dtypes(include=['object']).columns


    labelencoder = LabelEncoder()
    for col in object_columns:
        stroke_df[col] = labelencoder.fit_transform(stroke_df[col])

    y_data = stroke_df.pop("stroke")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(stroke_df, y_data, test_size=0.2, random_state=24)
    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="indicates regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    #use model to predict probability that given y value is 1
    y_pred_proba = model.predict_proba(x_test)[::,1]

    #calculate AUC of model
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    run.log("AUC", np.float(auc))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/hd-model.joblib')

if __name__ == '__main__':
    main()
