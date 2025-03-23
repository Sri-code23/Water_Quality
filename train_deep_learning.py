import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# Step 1: Load Balanced & Normalized Data
X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")
y_train = pd.read_csv("outputs/y_train.csv")
y_test = pd.read_csv("outputs/y_test.csv")

# Convert y_train and y_test to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Step 2: Reshape Data for Deep Learning Models (LSTM/GRU requires 3D input)
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 3: Define the Hybrid Model with Dropout
hybrid_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
hybrid_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
hybrid_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Step 5: Save the Model
hybrid_model.save("models/Hybrid_GRU_LSTM_Model.h5")
print("\n✅ Model trained and saved successfully!")






# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# import pandas as pd

# # Step 1: Load Balanced & Normalized Data
# X_train = pd.read_csv("outputs/X_train.csv")
# X_test = pd.read_csv("outputs/X_test.csv")
# y_train = pd.read_csv("outputs/y_train.csv")
# y_test = pd.read_csv("outputs/y_test.csv")

# # Convert y_train and y_test to 1D arrays
# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

# # Step 2: Reshape Data for Deep Learning Models (LSTM/GRU requires 3D input)
# X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# # Step 3: Define a function to train a model
# def train_model(model_name, model):
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#     print(f"\nTraining {model_name} model...")
#     model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

#     # Save the trained model
#     model.save(f"models/{model_name}.h5")
#     print(f"{model_name} model saved successfully!")

# # Step 4: Train LSTM Model
# lstm_model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     LSTM(32, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# train_model("LSTM_Model", lstm_model)

# # Step 5: Train GRU Model
# gru_model = Sequential([
#     GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     GRU(32, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# train_model("GRU_Model", gru_model)

# # Step 6: Train Hybrid GRU+LSTM Model
# hybrid_model = Sequential([
#     GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     LSTM(32, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# train_model("Hybrid_GRU_LSTM_Model", hybrid_model)












# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# # Step 1: Load Split Data
# X_train = pd.read_csv("outputs/X_train.csv")
# X_test = pd.read_csv("outputs/X_test.csv")
# y_train = pd.read_csv("outputs/y_train.csv")
# y_test = pd.read_csv("outputs/y_test.csv")

# # Convert y_train and y_test to 1D arrays
# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

# # Step 2: Reshape Data for Deep Learning Models
# X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# # Step 3: Define a Function to Train a Model
# def train_model(model_name, model):
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     print(f"\nTraining {model_name} model...")
#     model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

#     # Save Model
#     model.save(f"models/{model_name}.h5")
#     print(f"{model_name} model saved successfully!")

# # Step 4: Train LSTM Model
# lstm_model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     LSTM(32, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# train_model("LSTM_Model", lstm_model)

# # Step 5: Train GRU Model
# gru_model = Sequential([
#     GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     GRU(32, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# train_model("GRU_Model", gru_model)

# # Step 6: Train Hybrid GRU+LSTM Model
# hybrid_model = Sequential([
#     GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     LSTM(32, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# train_model("Hybrid_GRU_LSTM_Model", hybrid_model)
