import numpy as np
import tensorflow as tf
import pandas as pd

# Step 1: Load the Best Model
model = tf.keras.models.load_model("models/Hybrid_GRU_LSTM_Model.h5")

# Step 2: Define New Sample Data for Testing
# Format: [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity]
test_sample = np.array([[8.5, 150, 2000, 6.9, 350, 400, 15, 60, 3]], dtype=np.float32)  # Example input values

# Step 3: Normalize Data Using Training Set Scaling
X_train = pd.read_csv("outputs/X_train.csv")

# Convert min and max values to NumPy arrays
min_values = X_train.min().values  # Convert Pandas Series to NumPy array
max_values = X_train.max().values  # Convert Pandas Series to NumPy array

# Normalize the test sample
test_sample = (test_sample - min_values) / (max_values - min_values)

# Step 4: Reshape Data for Model Input
test_sample = test_sample.reshape((1, test_sample.shape[1], 1))

# Step 5: Make Prediction
prediction = model.predict(test_sample)

# Step 6: Interpret Result
potability = int(prediction[0][0] > 0.5)  # Convert probability to 0 or 1
print("\nPredicted Water Quality:")
print("🚰 Safe to Drink ✅" if potability == 1 else "⚠️ Unsafe to Drink ❌")
