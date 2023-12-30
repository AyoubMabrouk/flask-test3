import pandas as pd
from sklearn.preprocessing import LabelEncoder

paiement_data=pd.read_csv('pa.txt', sep = ';')

paiement_dataEncoded=paiement_data.copy()
labelencoder = LabelEncoder()
paiement_dataEncoded['Etat'] = labelencoder.fit_transform(paiement_data['Etat'])
paiement_dataEncoded['IdTypeLot'] = labelencoder.fit_transform(paiement_data['IdTypeLot'])
paiement_dataEncoded['BanquePayeur'] = labelencoder.fit_transform(paiement_data['BanquePayeur'])
paiement_dataEncoded['MotifRejet'] = labelencoder.fit_transform(paiement_data['MotifRejet'])
paiement_dataEncoded['RejetCCE'] = labelencoder.fit_transform(paiement_data['RejetCCE'])

X = paiement_dataEncoded.loc[:, ['Etat', 'IdTypeLot','BanquePayeur','MotifRejet']]
y = paiement_dataEncoded['RejetCCE']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, y_train)
    return trained_model
