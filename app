from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("gold_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        spx = float(request.form['spx'])
        uso = float(request.form['uso'])
        slv = float(request.form['slv'])
        eurusd = float(request.form['eurusd'])

        # Convert into array
        input_data = np.array([[spx, uso, slv, eurusd]])

        # Predict
        result = model.predict(input_data)[0]
        result = round(result, 2)

        return render_template("index.html", prediction_text=f"Predicted Gold Price (GLD): {result}")

if __name__ == "__main__":
    app.run(debug=True)