import matplotlib
matplotlib.use("Agg")  # Fix Matplotlib threading error

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/Hybrid_GRU_LSTM_Model.h5")

# Load training data for normalization
X_train = pd.read_csv("outputs/X_train.csv")
min_values = X_train.min().values
max_values = X_train.max().values

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    graph_url = None

    if request.method == "POST":
        try:
            # Get user input
            user_input = [float(request.form[key]) for key in request.form]

            # Convert input to NumPy array and normalize
            test_sample = np.array([user_input], dtype=np.float32)
            test_sample = (test_sample - min_values) / (max_values - min_values)
            test_sample = test_sample.reshape((1, test_sample.shape[1], 1))

            # Make prediction
            prediction_prob = model.predict(test_sample)[0][0]
            probability = round(prediction_prob * 100, 2)

            # Correct prediction logic
            prediction = "⚠️ Unsafe to Drink ❌" if prediction_prob < 0.5 else "🚰 Safe to Drink ✅"

            # Generate graph
            labels = ["Unsafe", "Safe"]
            values = [100 - probability, probability]
            colors = ["red", "green"]

            plt.figure(figsize=(6, 4))
            plt.bar(labels, values, color=colors)
            plt.xlabel("Water Safety")
            plt.ylabel("Probability (%)")
            plt.title("Water Quality Prediction")
            plt.ylim(0, 100)

            graph_url = "static/prediction_chart.png"
            plt.savefig(graph_url)
            plt.close()

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, probability=probability, graph_url=graph_url)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)




# from flask import Flask, render_template, request
# import numpy as np
# import tensorflow as tf
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model("models/Hybrid_GRU_LSTM_Model.h5")

# # Load training data to get min/max values for normalization
# X_train = pd.read_csv("outputs/X_train.csv")
# min_values = X_train.min().values  # Convert Pandas Series to NumPy array
# max_values = X_train.max().values  # Convert Pandas Series to NumPy array

# # Define route for the homepage
# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = None  # Default no result

#     if request.method == "POST":
#         try:
#             # Get user input from the form
#             pH = float(request.form["pH"])
#             Hardness = float(request.form["Hardness"])
#             Solids = float(request.form["Solids"])
#             Chloramines = float(request.form["Chloramines"])
#             Sulfate = float(request.form["Sulfate"])
#             Conductivity = float(request.form["Conductivity"])
#             Organic_Carbon = float(request.form["Organic_Carbon"])
#             Trihalomethanes = float(request.form["Trihalomethanes"])
#             Turbidity = float(request.form["Turbidity"])

#             # Convert input to NumPy array and normalize
#             test_sample = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_Carbon, Trihalomethanes, Turbidity]], dtype=np.float32)
#             test_sample = (test_sample - min_values) / (max_values - min_values)

#             # Reshape data for model input
#             test_sample = test_sample.reshape((1, test_sample.shape[1], 1))

#             # Make prediction
#             prediction_prob = model.predict(test_sample)[0][0]
#             prediction = "Safe to Drink ✅" if prediction_prob > 0.5 else "Unsafe to Drink ⚠️"

#         except Exception as e:
#             prediction = f"Error: {e}"

#     return render_template("index.html", prediction=prediction)

# # Run Flask App
# if __name__ == "__main__":
#     app.run(debug=True)
