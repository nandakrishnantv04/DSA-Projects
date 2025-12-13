from flask import Flask, request, render_template
import joblib
import pandas as pd 
# Initialize the Flask application
app = Flask(__name__)

# --- LOAD MODELS/FILES HERE ---
model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    """Renders the main page (index.html)."""
    # Pass an empty string so the template doesn't error on first load
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, makes a prediction, and returns it."""
    
    try:
    # --- 1. Get data from the form ---
       input_data = {}

       for col in model_columns:
           input_data[col] = float(request.form[col])

    # --- 2. Convert to DataFrame ---
       input_df = pd.DataFrame([input_data])

    # --- 3. Make prediction ---
       prediction = model.predict(input_df)[0]

       prediction_text = f"Predicted Normalized Points: {round(prediction, 4)}"
    except Exception as e:
        prediction_text = f"An error occurred: {e}"

    # --- 3. Render the page again with the prediction ---
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)