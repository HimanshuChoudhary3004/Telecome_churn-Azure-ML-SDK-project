import joblib
import json
import pandas as pd
from azureml.core import Model

def init():
    global ref_col, predictor
    model_path = Model.get_model_path('Telecom_voting_classifier')

    ref_col,predictor = joblib.load(model_path)



def run(row_data):
    data_dict = json.loads(row_data)['data']
    data = pd.DataFrame.from_dict(data_dict)

    data_enc = pd.get_dummies(data)
    data_cols = data_enc.columns

    missing_cols = ref_col.difference(data_cols)

    for col in missing_cols:
        data_enc[col] = 0

    data_enc = data_enc[ref_col]

    predictions = predictor.predict(data_enc)

    classes = ['Not churn','Churn']
    predicted_class = []
    for pred in predictions:
        predicted_class.append(classes[pred])

    
    return json.dumps(predicted_class)