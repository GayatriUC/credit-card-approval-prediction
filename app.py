import pickle
from flask import Flask,request,app,jsonify,url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

#Load  the Model
rfmodel=pickle.load(open('rfmodel.pkl','rb'))

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
  # Get data from request
  data = request.get_json()
  # Convert data to DataFrame
  df = pd.DataFrame(data['features'])  # Access features array
  # Preprocess categorical data (if necessary)
  # ... (e.g., one-hot encoding)
  # Make predictions
  predictions = rfmodel.predict(df)
  # ...

