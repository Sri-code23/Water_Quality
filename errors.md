## Error faced 

```bash

Uninstalling numpy-2.1.3:
      Successfully uninstalled numpy-2.1.3
Successfully installed contourpy-1.3.1 cycler-0.12.1 fonttools-4.56.0 joblib-1.3.2 kiwisolver-1.4.8 matplotlib-3.8.2 numpy-1.26.3 pandas-2.1.4 pillow-11.1.0 pyparsing-3.2.1 python-dateutil-2.9.0.post0 pytz-2025.1 scikit-learn-1.3.2 scipy-1.11.4 seaborn-0.12.2 threadpoolctl-3.6.0 tzdata-2025.1
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\train_deep_learning.py
2025-03-18 23:19:30.339756: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:19:37.137053: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:19:53.514846: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\keras\src\layers\rnn\rnn.py:200: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)

Training LSTM_Model model...
Epoch 1/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 10s 27ms/step - accuracy: 0.5919 - loss: 0.6759 - val_accuracy: 0.6280 - val_loss: 0.6608
Epoch 2/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.6012 - loss: 0.6729 - val_accuracy: 0.6280 - val_loss: 0.6695
Epoch 3/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6170 - loss: 0.6718 - val_accuracy: 0.6280 - val_loss: 0.6609
Epoch 4/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6086 - loss: 0.6701 - val_accuracy: 0.6280 - val_loss: 0.6643
Epoch 5/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6002 - loss: 0.6736 - val_accuracy: 0.6280 - val_loss: 0.6607
Epoch 6/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6111 - loss: 0.6688 - val_accuracy: 0.6280 - val_loss: 0.6628
Epoch 7/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5990 - loss: 0.6740 - val_accuracy: 0.6280 - val_loss: 0.6622
Epoch 8/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.5974 - loss: 0.6749 - val_accuracy: 0.6280 - val_loss: 0.6605
Epoch 9/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6041 - loss: 0.6721 - val_accuracy: 0.6280 - val_loss: 0.6605
Epoch 10/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6068 - loss: 0.6704 - val_accuracy: 0.6280 - val_loss: 0.6631
Epoch 11/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6172 - loss: 0.6666 - val_accuracy: 0.6280 - val_loss: 0.6647
Epoch 12/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6031 - loss: 0.6725 - val_accuracy: 0.6280 - val_loss: 0.6601
Epoch 13/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.5937 - loss: 0.6778 - val_accuracy: 0.6280 - val_loss: 0.6601
Epoch 14/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6177 - loss: 0.6652 - val_accuracy: 0.6280 - val_loss: 0.6629
Epoch 15/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6050 - loss: 0.6714 - val_accuracy: 0.6280 - val_loss: 0.6637
Epoch 16/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.5982 - loss: 0.6734 - val_accuracy: 0.6280 - val_loss: 0.6614
Epoch 17/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6098 - loss: 0.6704 - val_accuracy: 0.6280 - val_loss: 0.6622
Epoch 18/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6279 - loss: 0.6618 - val_accuracy: 0.6280 - val_loss: 0.6638
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using 
instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
LSTM_Model model saved successfully!

Training GRU_Model model...
Epoch 1/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 8s 27ms/step - accuracy: 0.6067 - loss: 0.6786 - val_accuracy: 0.6280 - val_loss: 0.6700
Epoch 2/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6104 - loss: 0.6814 - val_accuracy: 0.6280 - val_loss: 0.6616
Epoch 3/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6059 - loss: 0.6718 - val_accuracy: 0.6280 - val_loss: 0.6621
Epoch 4/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6135 - loss: 0.6699 - val_accuracy: 0.6280 - val_loss: 0.6606
Epoch 5/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 17ms/step - accuracy: 0.6103 - loss: 0.6704 - val_accuracy: 0.6280 - val_loss: 0.6658
Epoch 6/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6047 - loss: 0.6725 - val_accuracy: 0.6280 - val_loss: 0.6648
Epoch 7/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 18ms/step - accuracy: 0.6205 - loss: 0.6670 - val_accuracy: 0.6280 - val_loss: 0.6810
Epoch 8/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6175 - loss: 0.6731 - val_accuracy: 0.6280 - val_loss: 0.6617
Epoch 9/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.5939 - loss: 0.6759 - val_accuracy: 0.6280 - val_loss: 0.6600
Epoch 10/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.6179 - loss: 0.6654 - val_accuracy: 0.6280 - val_loss: 0.6609
Epoch 11/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.6094 - loss: 0.6700 - val_accuracy: 0.6280 - val_loss: 0.6599
Epoch 12/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.5986 - loss: 0.6745 - val_accuracy: 0.6280 - val_loss: 0.6630
Epoch 13/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - accuracy: 0.6167 - loss: 0.6664 - val_accuracy: 0.6280 - val_loss: 0.6606
Epoch 14/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.5896 - loss: 0.6781 - val_accuracy: 0.6280 - val_loss: 0.6598
Epoch 15/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.6001 - loss: 0.6737 - val_accuracy: 0.6280 - val_loss: 0.6617
Epoch 16/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.5965 - loss: 0.6748 - val_accuracy: 0.6280 - val_loss: 0.6616
Epoch 17/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.6013 - loss: 0.6766 - val_accuracy: 0.6280 - val_loss: 0.6608
Epoch 18/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - accuracy: 0.5999 - loss: 0.6736 - val_accuracy: 0.6280 - val_loss: 0.6644
Epoch 19/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.6192 - loss: 0.6665 - val_accuracy: 0.6280 - val_loss: 0.6631
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using 
instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
GRU_Model model saved successfully!

Training Hybrid_GRU_LSTM_Model model...
Epoch 1/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 7s 26ms/step - accuracy: 0.5840 - loss: 0.6902 - val_accuracy: 0.6280 - val_loss: 0.6619
Epoch 2/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 18ms/step - accuracy: 0.6148 - loss: 0.6690 - val_accuracy: 0.6280 - val_loss: 0.6641
Epoch 3/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 15ms/step - accuracy: 0.6016 - loss: 0.6738 - val_accuracy: 0.6280 - val_loss: 0.6608
Epoch 4/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 4s 49ms/step - accuracy: 0.6069 - loss: 0.6746 - val_accuracy: 0.6280 - val_loss: 0.6632
Epoch 5/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 4s 27ms/step - accuracy: 0.5996 - loss: 0.6732 - val_accuracy: 0.6280 - val_loss: 0.6604
Epoch 6/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 22ms/step - accuracy: 0.6056 - loss: 0.6716 - val_accuracy: 0.6280 - val_loss: 0.6625
82/82 ━━━━━━━━━━━━━━━━━━━━ 3s 22ms/step - accuracy: 0.6050 - loss: 0.6728 - val_accuracy: 0.6280 - val_loss: 0.6608
Epoch 8/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 20ms/step - accuracy: 0.6086 - loss: 0.6699 - val_accuracy: 0.6280 - val_loss: 0.6606
Epoch 9/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 4s 30ms/step - accuracy: 0.5950 - loss: 0.6773 - val_accuracy: 0.6280 - val_loss: 0.6615
Epoch 10/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 20ms/step - accuracy: 0.6128 - loss: 0.6684 - val_accuracy: 0.6280 - val_loss: 0.6600
Epoch 11/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 2s 20ms/step - accuracy: 0.6033 - loss: 0.6721 - val_accuracy: 0.6280 - val_loss: 0.6599
Epoch 12/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 3s 17ms/step - accuracy: 0.6104 - loss: 0.6693 - val_accuracy: 0.6280 - val_loss: 0.6643
Epoch 13/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 3s 17ms/step - accuracy: 0.6161 - loss: 0.6675 - val_accuracy: 0.6280 - val_loss: 0.6600
Epoch 14/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.5951 - loss: 0.6765 - val_accuracy: 0.6280 - val_loss: 0.6612
Epoch 15/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.6083 - loss: 0.6700 - val_accuracy: 0.6280 - val_loss: 0.6647
Epoch 16/50
82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.6089 - loss: 0.6703 - val_accuracy: 0.6280 - val_loss: 0.6627
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using 
instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Hybrid_GRU_LSTM_Model model saved successfully!
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\evaluate_models.py    
2025-03-18 23:23:45.057801: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:23:47.068442: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:23:52.188513: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

Evaluating LSTM_Model...
21/21 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step  
Accuracy: 0.6280487804878049
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\metrics\_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\metrics\_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\metrics\_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

           0       0.63      1.00      0.77       412
           1       0.00      0.00      0.00       244

    accuracy                           0.63       656
   macro avg       0.31      0.50      0.39       656
weighted avg       0.39      0.63      0.48       656


Evaluating GRU_Model...
21/21 ━━━━━━━━━━━━━━━━━━━━ 1s 41ms/step 
Accuracy: 0.6280487804878049
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\metrics\_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\metrics\_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\metrics\_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

           0       0.63      1.00      0.77       412
           1       0.00      0.00      0.00       244

    accuracy                           0.63       656
   macro avg       0.31      0.50      0.39       656
weighted avg       0.39      0.63      0.48       656
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\main.py
         ph    Hardness        Solids  Chloramines     Sulfate  Conductivity  Organic_carbon  Trihalomethanes  Turbidity  Potability
0       NaN  204.890455  20791.318981     7.300212  368.516441    564.308654       10.379783        86.990970   2.963135           0
1  3.716080  129.422921  18630.057858     6.635246         NaN    592.885359       15.180013        56.329076   4.500656           0
2  8.099124  224.236259  19909.541732     9.275884         NaN    418.606213       16.868637        66.420093   3.055934           0
3  8.316766  214.373394  22018.417441     8.059332  356.886136    363.266516       18.436524       100.341674   4.628771           0
4  9.092223  181.101509  17978.986339     6.546600  310.135738    398.410813       11.558279        31.997993   4.075075           0
ph                 0
Hardness           0
Solids             0
Chloramines        0
Sulfate            0
Conductivity       0
Organic_carbon     0
Trihalomethanes    0
Turbidity          0
Potability         0
dtype: int64        
Potability
0    1998
1    1278
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\main.py
         ph    Hardness        Solids  Chloramines     Sulfate  Conductivity  Organic_carbon  Trihalomethanes  Turbidity  Potability
0       NaN  204.890455  20791.318981     7.300212  368.516441    564.308654       10.379783        86.990970   2.963135           0
1  3.716080  129.422921  18630.057858     6.635246         NaN    592.885359       15.180013        56.329076   4.500656           0
2  8.099124  224.236259  19909.541732     9.275884         NaN    418.606213       16.868637        66.420093   3.055934           0
3  8.316766  214.373394  22018.417441     8.059332  356.886136    363.266516       18.436524       100.341674   4.628771           0
4  9.092223  181.101509  17978.986339     6.546600  310.135738    398.410813       11.558279        31.997993   4.075075           0
ph                 0
Hardness           0
Solids             0
Chloramines        0
Sulfate            0
Conductivity       0
Organic_carbon     0
Trihalomethanes    0
Turbidity          0
Potability         0
dtype: int64
Potability
0    1998
1    1278
4  9.092223  181.101509  17978.986339     6.546600  310.135738    398.410813       11.558279        31.997993   4.075075           0
ph                 0
Hardness           0
Solids             0
Chloramines        0
Sulfate            0
Conductivity       0
Organic_carbon     0
Trihalomethanes    0
Turbidity          0
Potability         0
dtype: int64
Potability
0    1998
1    1278
Conductivity       0
Organic_carbon     0
Trihalomethanes    0
Turbidity          0
Potability         0
dtype: int64
Potability
0    1998
1    1278
Turbidity          0
Potability         0
dtype: int64
Potability
0    1998
1    1278
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py         
Dataset Split Summary:
Total Samples: 3276   
Training Samples: 2620
Testing Samples: 656  
Training & Testing data saved in 'outputs' folder.

Original Class Distribution:
Potability
0    1998
1    1278
Name: count, dtype: int64   
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\split_data.py", line 11, in <module>
    from imblearn.over_sampling import SMOTE
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> pip install imblearn
Collecting imblearn
  Downloading imblearn-0.0-py2.py3-none-any.whl.metadata (355 bytes)
Collecting imbalanced-learn (from imblearn)
  Downloading imbalanced_learn-0.13.0-py3-none-any.whl.metadata (8.8 kB)
mblearn) (1.26.3)
Requirement already satisfied: scipy<2,>=1.10.1 in c:\users\vaath\onedrive\desktop\water quality contamination\tensorflow_env\lib\site-packages (from imbalanced-learn->imblearn) (1.11.4)
Requirement already satisfied: scikit-learn<2,>=1.3.2 in c:\users\vaath\onedrive\desktop\water quality contamination\tensorflow_env\lib\site-packages (from imbalanced-learn->imblearn) (1.3.2)
Collecting sklearn-compat<1,>=0.1 (from imbalanced-learn->imblearn)
  Downloading sklearn_compat-0.1.3-py3-none-any.whl.metadata (18 kB)
Requirement already satisfied: joblib<2,>=1.1.1 in c:\users\vaath\onedrive\desktop\water quality contamination\tensorflow_env\lib\site-packages (from imbalanced-learn->imblearn) (1.3.2)
Requirement already satisfied: threadpoolctl<4,>=2.0.0 in c:\users\vaath\onedrive\desktop\water quality contamination\tensorflow_env\lib\site-packages (from imbalanced-learn->imblearn) (3.6.0)
Downloading imblearn-0.0-py2.py3-none-any.whl (1.9 kB)
Downloading imbalanced_learn-0.13.0-py3-none-any.whl (238 kB)
Downloading sklearn_compat-0.1.3-py3-none-any.whl (18 kB)
Installing collected packages: sklearn-compat, imbalanced-learn, imblearn
Successfully installed imbalanced-learn-0.13.0 imblearn-0.0 sklearn-compat-0.1.3
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\split_data.py", line 15, in <module>
    X = data.drop("Potability", axis=1)
        ^^^^
NameError: name 'data' is not defined
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\split_data.py", line 26, in <module>
    X_resampled, y_resampled = smote.fit_resample(X, y)
                               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 202, in fit_resample
    return super().fit_resample(X, y, **params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 99, in fit_resample
    X, y, binarize_y = self._check_X_y(X, y)
                       ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 157, in _check_X_y
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\utils\_sklearn_compat.py", line 426, in validate_data      
    return _estimator._validate_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 622, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 1146, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 957, in check_array
    _assert_all_finite(
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 122, in _assert_all_finite       
    _assert_all_finite_element_wise(
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
SMOTE does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py

Original Class Distribution:
Potability
0    1998
1    1278
Name: count, dtype: int64
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\split_data.py", line 30, in <module>
    X_resampled, y_resampled = smote.fit_resample(X, y)
                               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 202, in fit_resample
    return super().fit_resample(X, y, **params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 99, in fit_resample
    X, y, binarize_y = self._check_X_y(X, y)
                       ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 157, in _check_X_y
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\utils\_sklearn_compat.py", line 426, in validate_data      
    return _estimator._validate_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 622, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 1146, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 957, in check_array
    _assert_all_finite(
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 122, in _assert_all_finite       
    _assert_all_finite_element_wise(
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
SMOTE does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py

Missing Values Before Handling:
ph                 491
Hardness             0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
Potability           0
dtype: int64

Original Class Distribution:
Potability
0    1998
1    1278
Name: count, dtype: int64
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\split_data.py", line 33, in <module>
    X_resampled, y_resampled = smote.fit_resample(X, y)
                               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 202, in fit_resample
    return super().fit_resample(X, y, **params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 99, in fit_resample
    X, y, binarize_y = self._check_X_y(X, y)
                       ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 157, in _check_X_y
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\utils\_sklearn_compat.py", line 426, in validate_data      
    return _estimator._validate_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 622, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 1146, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 957, in check_array
    _assert_all_finite(
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 122, in _assert_all_finite       
    _assert_all_finite_element_wise(
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\utils\validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
SMOTE does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the fol(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\main.py
         ph    Hardness        Solids  Chloramines     Sulfate  Conductivity  Organic_carbon  Trihalomethanes  Turbidity  Potability
0       NaN  204.890455  20791.318981     7.300212  368.516441    564.308654       10.379783        86.990970   2.963135           0
1  3.716080  129.422921  18630.057858     6.635246         NaN    592.885359       15.180013        56.329076   4.500656           0
2  8.099124  224.236259  19909.541732     9.275884         NaN    418.606213       16.868637        66.420093   3.055934           0
3  8.316766  214.373394  22018.417441     8.059332  356.886136    363.266516       18.436524       100.341674   4.628771           0
4  9.092223  181.101509  17978.986339     6.546600  310.135738    398.410813       11.558279        31.997993   4.075075           0
ph                 0
Hardness           0
Solids             0
Chloramines        0
Sulfate            0
Conductivity       0
Organic_carbon     0
Trihalomethanes    0
Turbidity          0
Potability         0
dtype: int64
Potability
0    1998
1    1278
Name: count, dtype: int64
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py     

Missing Values Before Handling:
ph                 491
Hardness             0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
Potability           0
dtype: int64

Original Class Distribution:
Potability
0    1998
1    1278
Name: count, dtype: int64
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\split_data.py", line 33, in <module>
    X_resampled, y_resampled = smote.fit_resample(X, y)
                               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 202, in fit_resample
    return super().fit_resample(X, y, **params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\sklearn\base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 99, in fit_resample
    X, y, binarize_y = self._check_X_y(X, y)
                       ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\imblearn\base.py", line 157, in _check_X_y
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py   

Missing Values Before Handling:
ph                 491
Hardness             0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
Potability           0
dtype: int64
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\split_data.py

Missing Values Before Handling:
ph                 491
Hardness             0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
dtype: int64

Missing Values After Handling:
ph                 0
Hardness           0
Solids             0
Chloramines        0
Sulfate            0
Conductivity       0
Organic_carbon     0
Trihalomethanes    0
Turbidity          0
Potability         0
dtype: int64

Balanced Class Distribution:
Potability
0    1998
1    1998
Name: count, dtype: int64

✅ Balanced & Normalized training & testing data saved in 'outputs' folder!
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\train_deep_learning.py
2025-03-18 23:46:21.404414: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:46:23.116948: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:46:27.779319: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\keras\src\layers\rnn\rnn.py:200: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)

Training LSTM_Model model...
Epoch 1/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - accuracy: 0.4978 - loss: 0.6939 - val_accuracy: 0.4925 - val_loss: 0.6937
Epoch 2/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 15ms/step - accuracy: 0.5042 - loss: 0.6934 - val_accuracy: 0.5225 - val_loss: 0.6931
Epoch 3/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.4869 - loss: 0.6933 - val_accuracy: 0.5075 - val_loss: 0.6931
Epoch 4/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.4889 - loss: 0.6933 - val_accuracy: 0.5075 - val_loss: 0.6931
Epoch 5/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.4982 - loss: 0.6932 - val_accuracy: 0.5075 - val_loss: 0.6931
Epoch 6/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 11ms/step - accuracy: 0.4998 - loss: 0.6932 - val_accuracy: 0.4925 - val_loss: 0.6933
Epoch 7/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4888 - loss: 0.6933 - val_accuracy: 0.4925 - val_loss: 0.6936
Epoch 8/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4963 - loss: 0.6935 - val_accuracy: 0.4925 - val_loss: 0.6934
Epoch 9/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5152 - loss: 0.6930 - val_accuracy: 0.4925 - val_loss: 0.6932
Epoch 10/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5087 - loss: 0.6931 - val_accuracy: 0.5075 - val_loss: 0.6931
Epoch 11/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 11ms/step - accuracy: 0.5072 - loss: 0.6932 - val_accuracy: 0.4938 - val_loss: 0.6932
Epoch 12/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5096 - loss: 0.6932 - val_accuracy: 0.4825 - val_loss: 0.6932
Epoch 13/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.4986 - loss: 0.6932 - val_accuracy: 0.4950 - val_loss: 0.6932
Epoch 14/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5097 - loss: 0.6931 - val_accuracy: 0.4950 - val_loss: 0.6932
Epoch 15/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5046 - loss: 0.6932 - val_accuracy: 0.4925 - val_loss: 0.6932
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using 
instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
LSTM_Model model saved successfully!

Training GRU_Model model...
Epoch 1/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 7s 19ms/step - accuracy: 0.5029 - loss: 0.6939 - val_accuracy: 0.5075 - val_loss: 0.6931
Epoch 2/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 14ms/step - accuracy: 0.5229 - loss: 0.6932 - val_accuracy: 0.4925 - val_loss: 0.6952
Epoch 3/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5010 - loss: 0.6930 - val_accuracy: 0.5138 - val_loss: 0.6931
Epoch 4/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - accuracy: 0.5115 - loss: 0.6934 - val_accuracy: 0.5075 - val_loss: 0.6930
Epoch 5/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 19ms/step - accuracy: 0.5089 - loss: 0.6932 - val_accuracy: 0.4925 - val_loss: 0.6934
Epoch 6/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 14ms/step - accuracy: 0.5155 - loss: 0.6928 - val_accuracy: 0.5075 - val_loss: 0.6931
Epoch 7/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.4940 - loss: 0.6938 - val_accuracy: 0.5025 - val_loss: 0.6930
Epoch 8/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5103 - loss: 0.6931 - val_accuracy: 0.5013 - val_loss: 0.6931
Epoch 9/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 11ms/step - accuracy: 0.5076 - loss: 0.6929 - val_accuracy: 0.5075 - val_loss: 0.6929
Epoch 10/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5172 - loss: 0.6928 - val_accuracy: 0.5213 - val_loss: 0.6926
Epoch 11/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - accuracy: 0.5308 - loss: 0.6917 - val_accuracy: 0.5188 - val_loss: 0.6934
Epoch 12/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 11ms/step - accuracy: 0.5104 - loss: 0.6923 - val_accuracy: 0.5000 - val_loss: 0.6929
Epoch 13/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 15ms/step - accuracy: 0.4946 - loss: 0.6927 - val_accuracy: 0.5150 - val_loss: 0.6957
Epoch 14/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 20ms/step - accuracy: 0.5466 - loss: 0.6869 - val_accuracy: 0.5200 - val_loss: 0.6928
Epoch 15/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.5391 - loss: 0.6887 - val_accuracy: 0.5312 - val_loss: 0.6920
Epoch 16/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5347 - loss: 0.6875 - val_accuracy: 0.5163 - val_loss: 0.6903
Epoch 17/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5425 - loss: 0.6840 - val_accuracy: 0.5625 - val_loss: 0.6879
Epoch 18/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5546 - loss: 0.6854 - val_accuracy: 0.5550 - val_loss: 0.6898
Epoch 19/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.5810 - loss: 0.6782 - val_accuracy: 0.5550 - val_loss: 0.6804
Epoch 20/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 15ms/step - accuracy: 0.5707 - loss: 0.6764 - val_accuracy: 0.5700 - val_loss: 0.6826
Epoch 21/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 18ms/step - accuracy: 0.5851 - loss: 0.6716 - val_accuracy: 0.5725 - val_loss: 0.6839
Epoch 22/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.5701 - loss: 0.6705 - val_accuracy: 0.5800 - val_loss: 0.6789
Epoch 23/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - accuracy: 0.5958 - loss: 0.6627 - val_accuracy: 0.5425 - val_loss: 0.6804
Epoch 24/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.5739 - loss: 0.6688 - val_accuracy: 0.5813 - val_loss: 0.6826
Epoch 25/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5667 - loss: 0.6737 - val_accuracy: 0.5775 - val_loss: 0.6781
Epoch 26/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5849 - loss: 0.6675 - val_accuracy: 0.5713 - val_loss: 0.6729
Epoch 27/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - accuracy: 0.5823 - loss: 0.6685 - val_accuracy: 0.5700 - val_loss: 0.6741
Epoch 28/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - accuracy: 0.5962 - loss: 0.6617 - val_accuracy: 0.5688 - val_loss: 0.6739
Epoch 29/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - accuracy: 0.5814 - loss: 0.6623 - val_accuracy: 0.5700 - val_loss: 0.6761
Epoch 30/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.5777 - loss: 0.6644 - val_accuracy: 0.5800 - val_loss: 0.6706
Epoch 31/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 16ms/step - accuracy: 0.5946 - loss: 0.6587 - val_accuracy: 0.5763 - val_loss: 0.6700
Epoch 32/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.5988 - loss: 0.6536 - val_accuracy: 0.5650 - val_loss: 0.6786
Epoch 33/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.5919 - loss: 0.6559 - val_accuracy: 0.5813 - val_loss: 0.6708
Epoch 34/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5905 - loss: 0.6572 - val_accuracy: 0.5638 - val_loss: 0.6688
Epoch 35/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6021 - loss: 0.6516 - val_accuracy: 0.5738 - val_loss: 0.6674
Epoch 36/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5831 - loss: 0.6558 - val_accuracy: 0.5725 - val_loss: 0.6720
Epoch 37/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5913 - loss: 0.6518 - val_accuracy: 0.5775 - val_loss: 0.6666
Epoch 38/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5920 - loss: 0.6536 - val_accuracy: 0.5775 - val_loss: 0.6656
Epoch 39/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - accuracy: 0.5959 - loss: 0.6550 - val_accuracy: 0.5975 - val_loss: 0.6689
Epoch 40/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.6076 - loss: 0.6590 - val_accuracy: 0.5750 - val_loss: 0.6710
Epoch 41/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5831 - loss: 0.6628 - val_accuracy: 0.5625 - val_loss: 0.6678
Epoch 42/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5899 - loss: 0.6545 - val_accuracy: 0.5813 - val_loss: 0.6670
Epoch 43/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.5972 - loss: 0.6505 - val_accuracy: 0.5763 - val_loss: 0.6630
Epoch 44/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6101 - loss: 0.6497 - val_accuracy: 0.5663 - val_loss: 0.6677
Epoch 45/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - accuracy: 0.5973 - loss: 0.6526 - val_accuracy: 0.5950 - val_loss: 0.6661
Epoch 46/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6084 - loss: 0.6490 - val_accuracy: 0.5688 - val_loss: 0.6635
Epoch 47/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6106 - loss: 0.6492 - val_accuracy: 0.5763 - val_loss: 0.6636
Epoch 48/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6027 - loss: 0.6441 - val_accuracy: 0.5850 - val_loss: 0.6620
Epoch 49/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5948 - loss: 0.6525 - val_accuracy: 0.5825 - val_loss: 0.6642
Epoch 50/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.6053 - loss: 0.6450 - val_accuracy: 0.5700 - val_loss: 0.6661
Epoch 51/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.6003 - loss: 0.6555 - val_accuracy: 0.5813 - val_loss: 0.6643
Epoch 52/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6010 - loss: 0.6457 - val_accuracy: 0.5850 - val_loss: 0.6717
Epoch 53/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6072 - loss: 0.6486 - val_accuracy: 0.5750 - val_loss: 0.6627
Epoch 54/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.5928 - loss: 0.6563 - val_accuracy: 0.5950 - val_loss: 0.6621
Epoch 55/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - accuracy: 0.6082 - loss: 0.6467 - val_accuracy: 0.5813 - val_loss: 0.6659
Epoch 56/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 15ms/step - accuracy: 0.6186 - loss: 0.6424 - val_accuracy: 0.5850 - val_loss: 0.6625
Epoch 57/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.6207 - loss: 0.6415 - val_accuracy: 0.5987 - val_loss: 0.6660
Epoch 58/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.6154 - loss: 0.6403 - val_accuracy: 0.6025 - val_loss: 0.6633
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using 
instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
GRU_Model model saved successfully!

Training Hybrid_GRU_LSTM_Model model...
Epoch 1/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.5160 - loss: 0.6937 - val_accuracy: 0.4925 - val_loss: 0.6939
Epoch 2/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.4893 - loss: 0.6939 - val_accuracy: 0.4925 - val_loss: 0.6951
Epoch 3/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4824 - loss: 0.6949 - val_accuracy: 0.4925 - val_loss: 0.6941
Epoch 4/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5028 - loss: 0.6933 - val_accuracy: 0.4950 - val_loss: 0.6932
Epoch 5/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5064 - loss: 0.6931 - val_accuracy: 0.4925 - val_loss: 0.6937
Epoch 6/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5027 - loss: 0.6932 - val_accuracy: 0.4875 - val_loss: 0.6932
Epoch 7/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4891 - loss: 0.6933 - val_accuracy: 0.4925 - val_loss: 0.6933
Epoch 8/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5147 - loss: 0.6930 - val_accuracy: 0.5075 - val_loss: 0.6930
Epoch 9/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4955 - loss: 0.6932 - val_accuracy: 0.5188 - val_loss: 0.6931
Epoch 10/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4852 - loss: 0.6933 - val_accuracy: 0.5000 - val_loss: 0.6931
Epoch 11/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5119 - loss: 0.6931 - val_accuracy: 0.4875 - val_loss: 0.6932
Epoch 12/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5071 - loss: 0.6929 - val_accuracy: 0.5063 - val_loss: 0.6926
Epoch 13/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5178 - loss: 0.6931 - val_accuracy: 0.5125 - val_loss: 0.6926
Epoch 14/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5184 - loss: 0.6912 - val_accuracy: 0.5200 - val_loss: 0.6925
Epoch 15/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5372 - loss: 0.6891 - val_accuracy: 0.5462 - val_loss: 0.6890
Epoch 16/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5306 - loss: 0.6902 - val_accuracy: 0.5263 - val_loss: 0.7029
Epoch 17/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4919 - loss: 0.6944 - val_accuracy: 0.5250 - val_loss: 0.6910
Epoch 18/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5529 - loss: 0.6830 - val_accuracy: 0.5200 - val_loss: 0.6886
Epoch 19/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5532 - loss: 0.6835 - val_accuracy: 0.5612 - val_loss: 0.6878
Epoch 20/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5758 - loss: 0.6793 - val_accuracy: 0.5587 - val_loss: 0.6905
Epoch 21/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5844 - loss: 0.6754 - val_accuracy: 0.5512 - val_loss: 0.6850
Epoch 22/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - accuracy: 0.5756 - loss: 0.6696 - val_accuracy: 0.5437 - val_loss: 0.6849
Epoch 23/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5905 - loss: 0.6710 - val_accuracy: 0.5537 - val_loss: 0.6816
Epoch 24/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 11ms/step - accuracy: 0.5913 - loss: 0.6668 - val_accuracy: 0.5625 - val_loss: 0.6831
Epoch 25/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.5867 - loss: 0.6677 - val_accuracy: 0.5675 - val_loss: 0.6770
Epoch 26/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 11ms/step - accuracy: 0.5723 - loss: 0.6716 - val_accuracy: 0.5625 - val_loss: 0.6840
Epoch 27/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5964 - loss: 0.6630 - val_accuracy: 0.5700 - val_loss: 0.6801
Epoch 28/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.6026 - loss: 0.6617 - val_accuracy: 0.5675 - val_loss: 0.6866
Epoch 29/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5877 - loss: 0.6646 - val_accuracy: 0.5738 - val_loss: 0.6767
Epoch 30/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5776 - loss: 0.6682 - val_accuracy: 0.5800 - val_loss: 0.6714
Epoch 31/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5939 - loss: 0.6623 - val_accuracy: 0.5825 - val_loss: 0.6740
Epoch 32/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5837 - loss: 0.6662 - val_accuracy: 0.5838 - val_loss: 0.6700
Epoch 33/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5926 - loss: 0.6595 - val_accuracy: 0.5850 - val_loss: 0.6722
Epoch 34/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5899 - loss: 0.6620 - val_accuracy: 0.5462 - val_loss: 0.6795
Epoch 35/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6006 - loss: 0.6624 - val_accuracy: 0.5575 - val_loss: 0.6743
Epoch 36/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5961 - loss: 0.6596 - val_accuracy: 0.5763 - val_loss: 0.6730
Epoch 37/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.6008 - loss: 0.6582 - val_accuracy: 0.5800 - val_loss: 0.6735
Epoch 38/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6081 - loss: 0.6547 - val_accuracy: 0.5838 - val_loss: 0.6726
Epoch 39/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6068 - loss: 0.6547 - val_accuracy: 0.5713 - val_loss: 0.6692
Epoch 40/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6056 - loss: 0.6614 - val_accuracy: 0.5863 - val_loss: 0.6713
Epoch 41/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5750 - loss: 0.6607 - val_accuracy: 0.5800 - val_loss: 0.6674
Epoch 42/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 14ms/step - accuracy: 0.5974 - loss: 0.6588 - val_accuracy: 0.5700 - val_loss: 0.6700
Epoch 43/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.6026 - loss: 0.6625 - val_accuracy: 0.5838 - val_loss: 0.6702
Epoch 44/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6137 - loss: 0.6577 - val_accuracy: 0.5863 - val_loss: 0.6810
Epoch 45/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.6064 - loss: 0.6553 - val_accuracy: 0.5850 - val_loss: 0.6721
Epoch 46/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.6037 - loss: 0.6555 - val_accuracy: 0.5825 - val_loss: 0.6731
Epoch 47/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5972 - loss: 0.6477 - val_accuracy: 0.5775 - val_loss: 0.6759
Epoch 48/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.6119 - loss: 0.6515 - val_accuracy: 0.5863 - val_loss: 0.6685
Epoch 49/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5901 - loss: 0.6575 - val_accuracy: 0.5888 - val_loss: 0.6674
Epoch 50/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6026 - loss: 0.6553 - val_accuracy: 0.5863 - val_loss: 0.6659
Epoch 51/100
Epoch 52/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6174 - loss: 0.6490 - val_accuracy: 0.5738 - val_loss: 0.6712
Epoch 53/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 14ms/step - accuracy: 0.6020 - loss: 0.6562 - val_accuracy: 0.5650 - val_loss: 0.6714
Epoch 54/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.5986 - loss: 0.6553 - val_accuracy: 0.5813 - val_loss: 0.6675
Epoch 55/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.6233 - loss: 0.6433 - val_accuracy: 0.5850 - val_loss: 0.6678
Epoch 56/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.6197 - loss: 0.6501 - val_accuracy: 0.5738 - val_loss: 0.6691
Epoch 57/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6152 - loss: 0.6442 - val_accuracy: 0.5900 - val_loss: 0.6723
Epoch 58/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6063 - loss: 0.6525 - val_accuracy: 0.5825 - val_loss: 0.6675
Epoch 59/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 14ms/step - accuracy: 0.6224 - loss: 0.6462 - val_accuracy: 0.5763 - val_loss: 0.6663
Epoch 60/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.6209 - loss: 0.6434 - val_accuracy: 0.5800 - val_loss: 0.6734
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using 
instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Hybrid_GRU_LSTM_Model model saved successfully!
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python evaluate_models.py
>>
2025-03-18 23:51:05.488896: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:51:07.225293: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:51:11.993553: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

Evaluating LSTM_Model...
25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step  
Accuracy: 0.5075
              precision    recall  f1-score   support

           0       1.00      0.00      0.00       394
           1       0.51      1.00      0.67       406

    accuracy                           0.51       800
   macro avg       0.75      0.50      0.34       800
weighted avg       0.75      0.51      0.34       800


Evaluating GRU_Model...
25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step   
Accuracy: 0.585
              precision    recall  f1-score   support

           0       0.60      0.48      0.53       394
           1       0.58      0.69      0.63       406

    accuracy                           0.58       800
   macro avg       0.59      0.58      0.58       800
weighted avg       0.59      0.58      0.58       800


           0       0.60      0.48      0.53       394
           1       0.58      0.69      0.63       406

    accuracy                           0.58       800
   macro avg       0.59      0.58      0.58       800
weighted avg       0.59      0.58      0.58       800

           1       0.58      0.69      0.63       406

    accuracy                           0.58       800
   macro avg       0.59      0.58      0.58       800
weighted avg       0.59      0.58      0.58       800

   macro avg       0.59      0.58      0.58       800
weighted avg       0.59      0.58      0.58       800


Evaluating Hybrid_GRU_LSTM_Model...
25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step
Accuracy: 0.58625
              precision    recall  f1-score   support

           0       0.59      0.51      0.55       394
           1       0.58      0.66      0.62       406

    accuracy                           0.59       800
   macro avg       0.59      0.59      0.58       800
weighted avg       0.59      0.59      0.58       800

(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:52:16.344246: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:52:17.953274: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:52:22.606286: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Traceback (most recent call last):
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\test_model.py", line 16, in <module>
    test_sample = (test_sample - min_values) / (max_values - min_values)
                   ~~~~~~~~~~~~^~~~~~~~~~~~
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\pandas\core\generic.py", line 2102, in __array_ufunc__
    return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\pandas\core\arraylike.py", line 273, in array_ufunc
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "ops_dispatch.pyx", line 113, in pandas._libs.ops_dispatch.maybe_dispatch_ufunc_to_dunder_op
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\pandas\core\ops\common.py", line 76, in new_method
    return method(self, other)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\pandas\core\arraylike.py", line 198, in __rsub__
    return self._arith_method(other, roperator.rsub)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vaath\OneDrive\Desktop\Water Quality contamination\tensorflow_env\Lib\site-packages\pandas\core\series.py", line 5819, in _arith_method
2025-03-18 23:55:40.824921: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:55:42.459786: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:55:47.051465: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.      
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step

Predicted Water Quality:
🚰 Safe to Drink ✅
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.      
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 470ms/step

(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.      
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-cri(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-cri(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point r(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point r(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination> python .\test_model.py
2025-03-18 23:56:10.872649: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:12.530201: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 23:56:17.081040: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.      
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 470ms/step

Predicted Water Quality:
🚰 Safe to Drink ✅
(tensorflow_env) C:\Users\vaath\OneDrive\Desktop\Water Quality contamination>```