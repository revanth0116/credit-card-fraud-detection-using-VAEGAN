from flask import *
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
from io import StringIO

xgb_model_path = './xgb_model.joblib'
xgb_model = joblib.load(xgb_model_path)

catboost_model_path = './catboost_model.joblib'
catboost_model = joblib.load(catboost_model_path)

# Function to load new data from CSV file
def load_new_data(csv_file):
    # Load the new data from CSV file
    new_data = pd.read_csv("fraud1.csv")
    return new_data

# Function to make predictions using both models
def predict_fraud(new_data):
    # Assuming 'fraud_label' is the target variable
    X_new = new_data.drop(columns=['Class'])

    # Predict using XGBoost model
    xgb_predictions = xgb_model.predict(X_new)

    # Predict using CatBoost model
    catboost_predictions = catboost_model.predict(X_new)

    # Combine predictions from both models
    predictions = pd.DataFrame({
        'XGBoost_Predictions': xgb_predictions,
        'CatBoost_Predictions': catboost_predictions
    })

    return predictions


app = Flask(__name__)


# @app.route('/')
# def hello():
#   return "Hello World"
@app.route('/')
def upload():
  return render_template('prediction.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
  if request.method == 'POST':
    f = request.files['file']
    
    new_csv_file = f.read().decode("utf-8")

    new_data = pd.read_csv(StringIO(new_csv_file))
    predictions = predict_fraud(new_data)
    data = predictions.to_dict()
    indices = list(data['XGBoost_Predictions'].keys())
    return render_template('prediction.html',data=data,indices=indices)
    # df_string = predictions.to_string(index=False)
    # return df_string

if __name__ == '__main__':
  app.run(debug=True)