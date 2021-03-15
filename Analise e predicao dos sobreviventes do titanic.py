# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data_path = "/kaggle/input/titanic/train.csv"
train_data =  pd.read_csv(train_data_path)
train_data.columns
train_data.describe()

train_data_clean = train_data.dropna(axis=0)

train_data.head

target_predction = train_data_clean.Survived
x = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
train_data_clean_feature= train_data_clean[x]