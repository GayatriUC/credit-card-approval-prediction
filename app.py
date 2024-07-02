from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('rfmodel.pkl', 'rb') as file:
    rf = pickle.load(file)

# Load the dataset
data = pd.read_csv("traineddata.csv")

@app.route('/')
def home():
    Type_Income = data['Type_Income'].unique()
    Type_Occupation = data['Type_Occupation'].unique()
    Marital_status = data['Marital_status'].unique()
    Housing_type = data['Housing_type'].unique()
  
    return render_template('index.html', Type_Income=Type_Income, Type_Occupation=Type_Occupation, Marital_status=Marital_status , Housing_type=Housing_type)

@app.route('/predict', methods=['POST'])
def predict():
     Type_Income = request.form['Type_Income']
     Type_Occupation = request.form['Type_Occupation']
     Marital_status = request.form['Marital_status']
     Housing_type = request.form['Housing_type']
     Annual_income = float(request.form['Annual_income'])
     age = float(request.form['age'])
     experience = float(request.form['experience'])
    

query = pd.DataFrame(input).T
prediction = rf.predict(query)[0]

result = f"prediction is {prediction}"
render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
