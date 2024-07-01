from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
with open('rfmodel.pkl', 'rb') as f:
    pipe_RF = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json  # Assuming the data is sent as JSON
    new_data = pd.DataFrame(data)
    
    # Ensure the new data contains all the required columns
    required_columns = pipe_RF.named_steps['encoding'].transformers_[0][2] + \
                       list(new_data.columns[len(pipe_RF.named_steps['encoding'].transformers_[0][2]):])
    new_data = new_data.reindex(columns=required_columns, fill_value=0)
    
    # Preprocess and predict
    y_new_pred = pipe_RF.predict(new_data)
    
    # Return the prediction as JSON
    return jsonify({'prediction': y_new_pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
