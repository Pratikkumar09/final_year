import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


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

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
# Predict on test data
y_pred = lr.predict(x_test)
print("Logistic Regression Accuracy (model.score):", round(lr.score(x_test, y_test) * 100, 2), "%")

#print("Logistic Regression Accuracy:", lr.score(x_test, y_test))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print results
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")


#pickle.dump(lr, open('logistic_model.pkl', 'wb'))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
print("Random Forest Accuracy:", rf.score(x_test, y_test))
#pickle.dump(rf, open('random_forest_model.pkl', 'wb'))

svm = SVC()
svm.fit(x_train, y_train)
print("SVM Accuracy:", svm.score(x_test, y_test))
#pickle.dump(svm, open('svm_model.pkl', 'wb'))


# from sklearn.model_selection import GridSearchCV
#
# params = {
#     'C': [0.01, 0.1, 1, 10],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']
# }
# grid = GridSearchCV(LogisticRegression(), params, cv=5)
# grid.fit(x_train, y_train)
# print("Best score:", grid.best_score_)



pickle.dump(nv, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))