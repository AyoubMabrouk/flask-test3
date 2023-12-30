import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt



pa=pd.read_csv('pa.txt', sep = ';')
paiement_dataEncoded=pa
labelencoder = LabelEncoder()
paiement_dataEncoded['RIBBeneficiaire'] = labelencoder.fit_transform(pa['RIBBeneficiaire'])
paiement_dataEncoded['BanqueBeneficiaire'] = labelencoder.fit_transform(pa['BanqueBeneficiaire'])
paiement_dataEncoded['AgenceBeneficiaire'] = labelencoder.fit_transform(pa['AgenceBeneficiaire'])
paiement_dataEncoded['RIBPayeur'] = labelencoder.fit_transform(pa['RIBPayeur'])
paiement_dataEncoded['BanquePayeur'] = labelencoder.fit_transform(pa['BanquePayeur'])
paiement_dataEncoded['AgencePayeur'] = labelencoder.fit_transform(pa['AgencePayeur'])
paiement_dataEncoded['DateOperation'] = labelencoder.fit_transform(pa['DateOperation'])
paiement_dataEncoded['NumeroLigne'] = labelencoder.fit_transform(pa['NumeroLigne'])
Q1 = pa.quantile(0.25)
Q3 = pa.quantile(0.75)
IQR = Q3 - Q1
paPropre = pa[~((pa < (Q1 - 1.5 * IQR)) | (pa > (Q3 + 1.5 * IQR))).any(axis=1)]
dataDate=paPropre.groupby(['DateOperation']).sum()
dataDate['DateOperation']=dataDate.index
x = dataDate.DateOperation
y = dataDate.Montant
slope, intercept, r, p, std_err = stats.linregress(x, y)
def predict(x):
    return slope * x + intercept
 #def courbe ():
  #  mymodel = list(map(predict, x))
   # plt.scatter(x, y)
    #plt.plot(x, mymodel)
    #plt.savefig('/myplot.png')
#%%
