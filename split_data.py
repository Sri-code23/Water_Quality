import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
file_name = "water_potability.csv"
data = pd.read_csv(file_name)

# Step 2: Check for missing values
print("\nMissing Values Before Handling:")
print(data.isnull().sum())

# Step 3: Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

# Step 4: Check missing values after filling
print("\nMissing Values After Handling:")
print(data.isnull().sum())

# Step 5: Define Features (X) and Target (y)
X = data.drop("Potability", axis=1)
y = data["Potability"]

# Step 6: Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 7: Check new class distribution
print("\nBalanced Class Distribution:")
print(y_resampled.value_counts())

# Step 8: Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 9: Normalize Data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Save Balanced & Normalized Data
pd.DataFrame(X_train).to_csv("outputs/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("outputs/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("outputs/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("outputs/y_test.csv", index=False)

print("\n✅ Balanced & Normalized training & testing data saved in 'outputs' folder!")





# import pandas as pd
# """ print missing values before processing"""
# # Step 1: Load the dataset
# file_name = "water_potability.csv"
# data = pd.read_csv(file_name)

# # Step 2: Check for missing values
# print("\nMissing Values Before Handling:")
# print(data.isnull().sum())















# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# # Step 1: Load the dataset
# file_name = "water_potability.csv"
# data = pd.read_csv(file_name)

# print("\nMissing Values Before Handling:")
# print(data.isnull().sum())

# # Step 2: Check class imbalance
# print("\nOriginal Class Distribution:")
# print(data["Potability"].value_counts())

# # Step 3: Define Features (X) and Target (y)
# X = data.drop("Potability", axis=1)
# y = data["Potability"]

# # Step 4: Apply SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Step 5: Check new class distribution
# print("\nBalanced Class Distribution:")
# print(y_resampled.value_counts())

# # Step 6: Split Data (80% Training, 20% Testing)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Step 7: Normalize Data
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Step 8: Save Balanced & Normalized Data
# pd.DataFrame(X_train).to_csv("outputs/X_train.csv", index=False)
# pd.DataFrame(X_test).to_csv("outputs/X_test.csv", index=False)
# pd.DataFrame(y_train).to_csv("outputs/y_train.csv", index=False)
# pd.DataFrame(y_test).to_csv("outputs/y_test.csv", index=False)

# print("\n✅ Balanced & Normalized training & testing data saved in 'outputs' folder!")


















# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Step 1: Load the dataset
# file_name = "water_potability.csv"  # Ensure this matches your actual file name
# data = pd.read_csv(file_name)

# # Step 2: Handle Missing Values (If not done already)
# data.fillna(data.mean(), inplace=True)  # Replace NaN values with column mean

# # Step 3: Define Features (X) and Target (y)
# X = data.drop("Potability", axis=1)  # Features (all columns except 'Potability')
# y = data["Potability"]               # Target variable (Safe or Unsafe Water)

# # Step 4: Split Data (80% Training, 20% Testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Print Split Summary
# print("Dataset Split Summary:")
# print(f"Total Samples: {len(data)}")
# print(f"Training Samples: {len(X_train)}")
# print(f"Testing Samples: {len(X_test)}")

# # Step 6: Save Split Data (Optional)
# X_train.to_csv("outputs/X_train.csv", index=False)
# X_test.to_csv("outputs/X_test.csv", index=False)
# y_train.to_csv("outputs/y_train.csv", index=False)
# y_test.to_csv("outputs/y_test.csv", index=False)

# print("Training & Testing data saved in 'outputs' folder.")




# ########### Check class distribution ###############
# print("\nOriginal Class Distribution:")
# print(data["Potability"].value_counts())
