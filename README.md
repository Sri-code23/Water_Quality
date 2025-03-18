# Water Quality Prediction Using Deep Learning
=====================================================

Project Overview
---------------

This project **predicts the quality of water** based on various chemical and physical parameters using **Machine Learning (ML) and Deep Learning (DL) models**. The model determines whether water is **safe (`1`) or unsafe (`0`) for drinking**.

### How It Works

1. **Data Preprocessing**  
   - **Load the dataset (`water_potability.csv`)**  
   - **Handle missing values** (fill NaNs with column means)  
   - **Balance the dataset using SMOTE** (fixes class imbalance)  
   - **Normalize the data** for deep learning models  

2. **Training Machine Learning & Deep Learning Models**  
   - **ML Models**: Support Vector Machine (SVM), Random Forest, Decision Tree  
   - **DL Models**: LSTM, GRU, Hybrid GRU+LSTM  

3. **Evaluating Model Performance**  
   - Models are evaluated on **accuracy, precision, recall, and F1-score**  

4. **Testing Model with New Data**  
   - Users can input custom water parameters to check **safe or unsafe**  

Project Folder Structure
------------------------

### WaterQualityProject

* `water_potability.csv` # Dataset file
* `split_data.py` # Prepares and normalizes dataset
* `train_deep_learning.py` # Trains deep learning models
* `evaluate_models.py` # Evaluates model performance
* `test_model.py` # Predicts water quality for new samples
* `README.md` # Documentation
* `outputs` # Stores preprocessed & normalized data
* `models` # Stores trained deep learning models

Setup Guide
-----------

### 1. Install Dependencies

Make sure Python is installed (Recommended: Python **3.10 or 3.11**).

Run this command to install all required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing

Preprocess the dataset by running:
```bash
python split_data.py
```
This handles missing values, balances the dataset, and normalizes features.

### 3. Train the Deep Learning Models

```bash
python train_deep_learning.py
```
This trains LSTM, GRU, and Hybrid GRU+LSTM models.

### 4. Evaluate Model Performance

```bash
python evaluate_models.py
```
This prints accuracy, precision, recall, and F1-score for all models.

### 5. Test Model with New Data

```bash
python test_model.py
```
Enter a custom water sample, and the model will predict if it is safe to drink.

Understanding the Dataset
-------------------------

The dataset contains water quality parameters used to classify water as safe (1) or unsafe (0).

| Feature | Description |
| --- | --- |
| pH | Acidity/Alkalinity level (6.5–8.5 is ideal) |
| Hardness | Amount of dissolved calcium & magnesium |
| Solids | Total Dissolved Solids (TDS) in water |
| Chloramines | Chlorine-based disinfectant level |
| Sulfate | Sulfur compound (affects taste & health) |
| Conductivity | Ability of water to conduct electricity |
| Organic_carbon | Organic contamination in water |
| Trihalomethanes | Harmful byproducts of chlorination |
| Turbidity | Cloudiness of water |
| Potability | Target Column (0 = Unsafe, 1 = Safe) |

How the Model Works
--------------------

### 1. Data Preprocessing

* Handles missing values (NaN replaced with column mean).
* Balances dataset using SMOTE (equal safe/unsafe samples).
* Normalizes features using MinMaxScaler.

### 2. Deep Learning Models

| Model | Description |
| --- | --- |
| LSTM Model | Learns long-term patterns |
| GRU Model | Faster than LSTM, similar performance |
| Hybrid GRU+LSTM | Combines both models for best accuracy |

The Hybrid GRU+LSTM Model performed best with 91% accuracy.

### 3. Model Testing

A user inputs new water data (e.g., pH = 8.5, Hardness = 150, etc.).
The model predicts if the water is safe or unsafe.

Example Test Run
-----------------

### Input
```python
test_sample = np.array([[8.5, 150, 20000, 6.5, 350, 450, 15, 60, 3]])
```

### Expected Output

Predicted Water Quality:
 Safe to Drink 
or
 Unsafe to Drink 

Frequently Asked Questions (FAQ)
---------------------------------

### Q1: I get a "TensorFlow DLL load failed" error. How do I fix it?

Ensure Python 3.10 or 3.11 is installed (TensorFlow doesn't support Python 3.12 yet).
Run:
```bash
pip install tensorflow
```

### Q2: The model always predicts "Unsafe". What should I do?

Ensure `split_data.py` was run correctly (this fixes dataset imbalance).
Try retraining models:
```bash
python train_deep_learning.py
```

### Q3: Can I deploy this model as a web app?

Yes! The model can be deployed using Flask API or FastAPI for real-time predictions.

Future Improvements
-------------------

* Deploy as a web app (Flask or FastAPI)
* Improve accuracy using hyperparameter tuning
* Collect real sensor data instead of CSV

Conclusion
----------

This project successfully predicts water quality using deep learning models.
The Hybrid GRU+LSTM model achieved 91% accuracy.
Users can input new data and get real-time water safety predictions.

Next step: Deploy this model as a web app! 

Credits & References
--------------------

* Dataset Source: Kaggle - Water Potability Dataset
* Libraries Used: TensorFlow, Pandas, NumPy, Scikit-Learn, imbalanced-learn (SMOTE)