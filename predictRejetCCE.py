import pandas as pd
paiement_data=pd.read_csv('pa.txt', sep = ';')

X = paiement_data.loc[:, ['Etat', 'IdTypeLot','BanquePayeur','MotifRejet']]
y = paiement_data['RejetCCE']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, y_train)
    return trained_model