from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the best model (assuming it's Random Forest)
model = joblib.load("C:/Users/NISHTHA GUPTA/OneDrive/Desktop/Postpartum Depression/random_forest_model.pkl")

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    age = int(request.form['age'])
    feeling_sad = int(request.form['feeling_sad'])
    irritable = int(request.form['irritable'])
    trouble_sleeping = int(request.form['trouble_sleeping'])
    feeling_anxious = int(request.form['feeling_anxious'])
    bonding_problems = int(request.form['bonding_problems'])
    
    # Arrange the data in the required order for prediction
    features = np.array([[age, feeling_sad, irritable, trouble_sleeping, feeling_anxious, bonding_problems]])
    
    # Predict the probability
    prediction_prob = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]

    # Interpret the result based on probability
    if prediction == 1 and prediction_prob[1] >= 0.5:
        result = f"High risk of postpartum depression (Confidence: {prediction_prob[1]:.2f})"
    else:
        result = f"Low risk of postpartum depression (Confidence: {prediction_prob[0]:.2f})"
    
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
