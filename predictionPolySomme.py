import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

# Load data from a csv file
pa =pd.read_csv('pa.txt', sep = ';')
paiement_dataEncoded=pa.copy()
labelencoder = LabelEncoder()
paiement_dataEncoded['RIBBeneficiaire'] = labelencoder.fit_transform(pa['RIBBeneficiaire'])
paiement_dataEncoded['BanqueBeneficiaire'] = labelencoder.fit_transform(pa['BanqueBeneficiaire'])
paiement_dataEncoded['AgenceBeneficiaire'] = labelencoder.fit_transform(pa['AgenceBeneficiaire'])
paiement_dataEncoded['RIBPayeur'] = labelencoder.fit_transform(pa['RIBPayeur'])
paiement_dataEncoded['BanquePayeur'] = labelencoder.fit_transform(pa['BanquePayeur'])
paiement_dataEncoded['AgencePayeur'] = labelencoder.fit_transform(pa['AgencePayeur'])
paiement_dataEncoded['DateOperation'] = labelencoder.fit_transform(pa['DateOperation'])
paiement_dataEncoded['NumeroLigne'] = labelencoder.fit_transform(pa['NumeroLigne'])



Q1 = paiement_dataEncoded.quantile(0.25)
Q3 = paiement_dataEncoded.quantile(0.75)
IQR = Q3 - Q1
paPropre = paiement_dataEncoded[~((paiement_dataEncoded < (Q1 - 1.5 * IQR)) | (paiement_dataEncoded > (Q3 + 1.5 * IQR))).any(axis=1)]
dataDate=paPropre.groupby(['DateOperation']).sum()
dataDate['DateOperation']=dataDate.index
train_size = 0.7
split_index = int(train_size * len(dataDate))
df_train, df_test = dataDate[:split_index], dataDate[split_index:]
X_train = df_train[['DateOperation']].values
y_train = df_train['Montant'].values.reshape(-1, 1)
X_test = df_test[['DateOperation']].values
y_test = df_test['Montant'].values.reshape(-1, 1)
degree = 2
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)
def predictPolySomme(x):
    data = pd.DataFrame({'Feature1': [x]})  # Replace 'Feature1' with the actual column name
    x_new_poly = pd.DataFrame(poly.transform(data), columns=poly.get_feature_names_out(data.columns))
    y_pred = model.predict(x_new_poly)
    return float(abs(y_pred[0]))
