from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('rfmodel.pkl', 'rb') as file:
    rf = pickle.load(file)

# Load the dataset
data = pd.read_csv("traindata.csv")

@app.route('/')
def home():
    Type_Income = data['Type_Income'].unique()
    Type_Occupation = data['Type_Occupation'].unique()
    Marital_status = data['Marital_status'].unique()
    Housing_type = data['Housing_type'].unique()
  
    return render_template('index.html', Type_Income=Type_Income, Type_Occupation=Type_Occupation, Marital_status=Marital_status , Housing_type=Housing_type)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    Type_Income = request.form['Type_Income']
    Type_Occupation = request.form['Type_Occupation']
    Marital_status = request.form['Marital_status']
    Housing_type = request.form['Housing_type']
    Annual_income = float(request.form['Annual_income'])
    age = float(request.form['age'])
    experience = float(request.form['experience'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Type_Income': [Type_Income],
        'Type_Occupation': [Type_Occupation],
        'Marital_status': [Marital_status],
        'Housing_type': [Housing_type],
        'Annual_income': [Annual_income],
        'age': [age],
        'experience': [experience]
    })

    # Make prediction
    prediction = rf.predict(input_data)[0]

    # Create result message
    result = "Credid Card Application Rejected" if prediction == 1 else "Credid Card Application Approved "

    # Render the template with the prediction result
    return render_template('index.html', prediction_text=result, Type_Income=data['Type_Income'].unique(), Type_Occupation=data['Type_Occupation'].unique(), Marital_status=data['Marital_status'].unique(), Housing_type=data['Housing_type'].unique())

if __name__ == '__main__':
    app.run(debug=True)
