# file objective: to view the csv file and to clear the missing data values


import pandas as pd

# Load CSV file
file_name = "water_potability.csv"  # Ensure this matches your actual file name
data = pd.read_csv(file_name)

# Display first 5 rows
print(data.head())

################ output ####################
"""
         ph    Hardness        Solids  Chloramines  ...  Organic_carbon  Trihalomethanes  Turbidity  Potability
0       NaN  204.890455  20791.318981     7.300212  ...       10.379783        86.990970   2.963135           0
1  3.716080  129.422921  18630.057858     6.635246  ...       15.180013        56.329076   4.500656           0
2  8.099124  224.236259  19909.541732     9.275884  ...       16.868637        66.420093   3.055934           0
3  8.316766  214.373394  22018.417441     8.059332  ...       18.436524       100.341674   4.628771           0
4  9.092223  181.101509  17978.986339     6.546600  ...       11.558279        31.997993   4.075075           0

[5 rows x 10 columns]

"""

######################## check for missing values ###############################
# print(data.isnull().sum())  # Count missing values in each column

#output 
"""
ph                 491
ph                 491
Hardness             0
Solids               0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
Potability           0
dtype: int64

"""

###################### to handle missing values ##############################
# Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

# Check again if missing values are handled
print(data.isnull().sum())  # Should print all 0s

#output 
"""
[5 rows x 10 columns]
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
"""


########### to check how many safe and unsafe potability ############
import pandas as pd

# Load dataset
data = pd.read_csv("water_potability.csv")

# Count values of each class
print(data["Potability"].value_counts())






