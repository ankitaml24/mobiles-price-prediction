from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Trained Model ko Load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Form inputs fetch karo
        battery_power = float(request.form['battery_power'])
        clock_speed = float(request.form['clock_speed'])
        int_memory = float(request.form['int_memory'])
        n_cores = int(request.form['n_cores'])
        px_height = float(request.form['px_height'])
        px_width = float(request.form['px_width'])
        ram = float(request.form['ram'])

        # Features array match sequence with model input
        features = np.array([[battery_power, clock_speed, int_memory, n_cores, px_height, px_width, ram]])
        
        # Model Prediction
        prediction = model.predict(features)[0]

        # Map numerical prediction to label
        price_labels = {
            0: "Low Cost Category 🟢",
            1: "Medium Cost Category 🟡",
            2: "High Cost Category 🟠",
            3: "Very High (Premium) Category 🔴"
        }
        
        result = price_labels.get(prediction, "Unknown")

        return render_template('index.html', prediction_text=f"Predicted Range: {result}")

if __name__ == '__main__':
    app.run(debug=True)