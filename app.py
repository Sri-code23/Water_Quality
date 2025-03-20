from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/Hybrid_GRU_LSTM_Model.h5")

# Load training data to get min/max values for normalization
X_train = pd.read_csv("outputs/X_train.csv")
min_values = X_train.min().values  # Convert Pandas Series to NumPy array
max_values = X_train.max().values  # Convert Pandas Series to NumPy array

# Define route for the homepage
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # Default no result

    if request.method == "POST":
        try:
            # Get user input from the form
            pH = float(request.form["pH"])
            Hardness = float(request.form["Hardness"])
            Solids = float(request.form["Solids"])
            Chloramines = float(request.form["Chloramines"])
            Sulfate = float(request.form["Sulfate"])
            Conductivity = float(request.form["Conductivity"])
            Organic_Carbon = float(request.form["Organic_Carbon"])
            Trihalomethanes = float(request.form["Trihalomethanes"])
            Turbidity = float(request.form["Turbidity"])

            # Convert input to NumPy array and normalize
            test_sample = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_Carbon, Trihalomethanes, Turbidity]], dtype=np.float32)
            test_sample = (test_sample - min_values) / (max_values - min_values)

            # Reshape data for model input
            test_sample = test_sample.reshape((1, test_sample.shape[1], 1))

            # Make prediction
            prediction_prob = model.predict(test_sample)[0][0]
            prediction = "Safe to Drink ✅" if prediction_prob > 0.5 else "Unsafe to Drink ⚠️"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
