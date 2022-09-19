from string import whitespace
from urllib import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import joblib
import json
from markupsafe import Markup
from xgboost import XGBRegressor
from sklearn import metrics
import pandas as pd
import csv
import seaborn as sns
from flask import Flask, jsonify
url='https://drive.google.com/file/d/1fL6MkQ1Sd76TJ5k6pSWUZRTRjK7PkIHo/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]

exercise_data = pd.read_csv(url)
url2 = 'https://drive.google.com/uc?id=1Vc5rl9TxXCuqNr5sSVZbzICKUbi1KsE3'
calories = pd.read_csv(url2)
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
print(calories_data)
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

sns.set()
sns.distplot(calories_data['Age'])
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = XGBRegressor()
res = model.fit(X_train, Y_train)
test_data_prediction = model.predict(X_test)
print(test_data_prediction)
df = pd.read_csv('C:/Users/abhij/Desktop/test.csv')
print(df)
test = model.predict(df)
print(test)
if(test<20):
    print('Unhealthy!!!!! Visit a doctor.')
elif(test>20 and test<=50):
    print('Weight_lifting')
elif(test>50 and test<=100):
    print('High Intensity Interval Training')
elif(test>100 and test<=200):
    print('Cardio')
elif(test>200):
    print('Walking, other low intensity exercise')
data_cols = ['Gender', 'Age', 'Height', 'Weight', 'Duration']
joblib.dump(res, 'model.pkl') 
joblib.dump(data_cols, 'model_cols.pkl')
