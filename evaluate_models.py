import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Test Data
X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv")

# Convert y_test to 1D array
y_test = y_test.values.ravel()

# Reshape for Deep Learning Models
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 2: Load Trained Models
models = {
    "LSTM_Model": tf.keras.models.load_model("models/LSTM_Model.h5"),
    "GRU_Model": tf.keras.models.load_model("models/GRU_Model.h5"),
    "Hybrid_GRU_LSTM_Model": tf.keras.models.load_model("models/Hybrid_GRU_LSTM_Model.h5")
}

# Step 3: Evaluate Each Model
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=1))



# import numpy as np
# import tensorflow as tf
# import pandas as pd
# from sklearn.metrics import accuracy_score, classification_report

# # Step 1: Load Test Data
# X_test = pd.read_csv("outputs/X_test.csv")
# y_test = pd.read_csv("outputs/y_test.csv")

# # Convert y_test to 1D array
# y_test = y_test.values.ravel()

# # Reshape for Deep Learning Models
# X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# # Step 2: Load Trained Models
# models = {
#     "LSTM_Model": tf.keras.models.load_model("models/LSTM_Model.h5"),
#     "GRU_Model": tf.keras.models.load_model("models/GRU_Model.h5"),
#     "Hybrid_GRU_LSTM_Model": tf.keras.models.load_model("models/Hybrid_GRU_LSTM_Model.h5")
# }

# # Step 3: Evaluate Each Model
# for model_name, model in models.items():
#     print(f"\nEvaluating {model_name}...")
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1
    
#     print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
#     print(classification_report(y_test, y_pred))
