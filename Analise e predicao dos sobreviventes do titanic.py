# Loeads library

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Import file

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Read file

train_data_path = "/kaggle/input/titanic/train.csv"
train_data =  pd.read_csv(train_data_path)

test_data_path = "/kaggle/input/titanic/train.csv"
test_data =  pd.read_csv(train_data_path)

#Analisando os dados

train_data.columns
train_data.info()
train_data.describe()
train_data.head(10)

#limpeza dos dados nulos

train_data_clean = train_data.dropna(axis = 0)
train_data_clean.info()
train_data_clean.describe()

#Convertendo os dados de texto para numeros

train_data_clean['sex_number']=0
train_data_clean['Port_Embarked_number']=0

for index, row in train_data_clean.iterrows():
    if row['Embarked']  == 'C':
        train_data_clean.loc[index, 'Port_Embarked_number'] = 1
    elif row['Embarked']  == 'Q':
        train_data_clean.loc[index , 'Port_Embarked_number'] = 2
    else:
        train_data_clean.loc[index , 'Port_Embarked_number'] = 3

train_data_clean.head(10)

#Separando o alvo de precisão e o recurso

target_predction = train_data_clean.Survived
x = ['Pclass','sex_number','Age','SibSp','Parch','Fare','Port_Embarked_number']
train_data_clean_feature= train_data_clean[x]

#Verificando se existe corelação entre dados

fig, (axs1,axs2,axs3) = plt.subplots(ncols=3)
axs[0] = sns.jointplot(x='Pclass',y='Survived',data=train_data_clean, kind='reg')
axs[1] = sns.jointplot(x='Pclass',y='Survived',data=train_data_clean, kind='reg')
axs[2] = sns.jointplot(x='Pclass',y='Survived',data=train_data_clean, kind='reg')

#Treinando o algortimo de regressao linear

regr = linear_model.LinearRegression()
regr.fit(train_data_clean_feature,target_predction)

#Fazendo predição
preditc_train = regr.predict(train_data_clean[x])


#Analisando resultado
# The coefficients
print('coefficient of determination:', regr.score(train_data_clean_feature,target_predction))
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(train_data_clean.Survived, preditc_train))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(train_data_clean.Survived, preditc_train))