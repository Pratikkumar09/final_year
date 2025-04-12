import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
# import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("STUDENT.csv")
df.head()
df.dropna(inplace=True)

le = preprocessing.LabelEncoder()
df['Encoded_School_Type'] = le.fit_transform(df['SCHOOL TYPE'])
df.drop(columns = 'SCHOOL TYPE',inplace = True)

df['Encoded_location'] = le.fit_transform(df['LOCATION'])
df.drop(columns = 'LOCATION',inplace = True)

df['Encoded_Gender'] = le.fit_transform(df['GENDER'])
df.drop(columns = 'GENDER',inplace = True)

X = df[['AGE', 'FAMILY INCOME', 'ACADEMIC MARKS', 'NO OF SIBLINGS', 'CLASS' , 'Encoded_School_Type','Encoded_Gender','Encoded_location']]


# Find min and max income
# min_income = df['FAMILY INCOME'].min()
# max_income = df['FAMILY INCOME'].max()
#
# print(f"Minimum Family Income: {min_income}")
# print(f"Maximum Family Income: {max_income}")



#print(X.shape)
df['Encoded_Dropout'] = le.fit_transform(df['DROPOUT'])
df.drop(columns='DROPOUT', inplace=True)

y = df['Encoded_Dropout']
# print(y.type)
y = y.astype('int')
X = X.astype('int')
#print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
nv = GaussianNB()
nv

nv.fit(x_train, y_train)
#print(nv.predict(x_test))
print(nv.score(x_test, y_test))

pickle.dump(nv, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))