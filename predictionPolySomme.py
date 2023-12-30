import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data from a csv file
pa =pd.read_csv('pa.txt', sep =';')

Q1 = pa.quantile(0.25)
Q3 = pa.quantile(0.75)
IQR = Q3 - Q1

paPropre = pa[~((pa < (Q1 - 1.5 * IQR)) | (pa > (Q3 + 1.5 * IQR))).any(axis=1)]

dataDate=paPropre.groupby(['DateOperation']).sum()
dataDate['DateOperation']=dataDate.index


# Split the data into training and testing sets
train_size = 0.7
split_index = int(train_size * len(dataDate))
df_train, df_test = dataDate[:split_index], dataDate[split_index:]

# Select the relevant features and target variable
X_train = df_train[['DateOperation']].values
y_train = df_train['Montant'].values.reshape(-1, 1)
X_test = df_test[['DateOperation']].values
y_test = df_test['Montant'].values.reshape(-1, 1)

# Create polynomial features
degree = 2
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

def predictPolySomme(x, poly, model):
    # Create a DataFrame with the input feature(s)
    data = pd.DataFrame({'Feature1': [x]})  # Replace 'Feature1' with the actual column name

    # Transform the input data using the polynomial features
    x_new_poly = pd.DataFrame(poly.transform(data), columns=poly.get_feature_names_out(data.columns))

    # Make the prediction using the model
    y_pred = model.predict(x_new_poly)

    # Return the absolute value of the predicted result
    return abs(y_pred[0])
