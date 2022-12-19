import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# DATA COLLECTION & PROCESSING :-

# Loading data to dataframe :-
data_Full = pd.read_csv('cancer.csv')
print(data_Full)

# Numbers of rows and columns in the dataset :-
a = data_Full.shape
print(a)

# Getting some information about the data :-
data_Full.info()

# checking for missing values :-
b = data_Full.isnull().sum()
print(b)

# Statistical measures about the data
c = data_Full.describe()
print(c)

# Checking the distribution of Target variable :-
data = pd.read_csv('cancer.csv')
print(data['diagnosis'].value_counts())

# ENCODING :-
# LABEL ENCODER:-
# Converting the string data into numerical data:-
label = LabelEncoder()
# Seeing the types of string data:-
label.fit(data['diagnosis'])
print(label.classes_)
# Adding the converted numerical data as new column:-
data['diagnosis_label'] = label.transform(data['diagnosis'])
print(data['diagnosis_label'])


# SEPARATING THE FEATURES AND TARGET:-
x = data.drop(['id', 'Unnamed: 32', 'diagnosis', 'diagnosis_label'], axis=1)
y = data['diagnosis_label']
print(x)
print(y)

# SPLITTING THE DATA INTO TRAINING DATA  AND TESTING DATA :-
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x.shape, x_train.shape, x_test.shape)

# MODEL TRAINING :-
# SVM-(support vector classifier)
svm = SVC()
# Training the SVM model by using the Training data :-
svm.fit(x_train, y_train)

# MODEL EVALUATION :-
# ACCURACY SCORE :-
pre = svm.predict(x_test)
pre_t = svm.predict(x_train)
print(classification_report(y_test, pre))
print(classification_report(y_train, pre_t))
# Accuracy on testing data
print(accuracy_score(y_test, pre))
# Accuracy on training data
print(accuracy_score(y_train, pre_t))
print('*****************************************************************')


# THIS IS ENSEMBLE TECHNIQUE :-
# BaggingClassifier :-

from sklearn.ensemble import BaggingClassifier

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
bag = BaggingClassifier(base_estimator=SVC(),
                        n_estimators=20,
                        max_samples=1.0,
                        max_features=1.0)
bag.fit(x_train, y_train)
pre1 = bag.predict(x_test)
pre_t1 = bag.predict(x_train)
print(classification_report(y_test, pre1))
print(classification_report(y_train, pre_t1))
print('*****************************************************************')

# Boosting :-

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
ada = AdaBoostClassifier(n_estimators=20)
ada.fit(x_train, y_train)
pre2 = ada.predict(x_test)
pre_t2 = ada.predict(x_train)
print(classification_report(y_test, pre2))
print(classification_report(y_train, pre_t2))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
gb = AdaBoostClassifier(n_estimators=20)
gb.fit(x_train, y_train)
pre3 = gb.predict(x_test)
pre_t3 = gb.predict(x_train)
print(classification_report(y_test, pre3))
print(classification_report(y_train, pre_t3))
print('*****************************************************************')


# BUILDING A PREDICTIVE SYSTEM :-

input_data = (
    12.31, 16.52, 79.19, 470.9, 0.09172, 0.06829, 0.03372, 0.02272, 0.172, 0.05914, 0.2505, 1.025, 1.74, 19.68,
    0.004854,
    0.01819, 0.01826, 0.007965, 0.01386, 0.002304, 14.11, 23.21, 89.71, 611.1, 0.1176, 0.1843, 0.1703, 0.0866, 0.2618,
    0.07609)
# Changing the input data into numpy array:-
input_data_as_numpy_array = np.asarray(input_data)
# Reshaping the numpy array as we are predicting for one data point:-
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = svm.predict(input_data_reshaped)
if prediction[0] == 0:
    print('The Breast Cancer is Benign : No Cancer')
else:
    print("The breast cancer is Malignant : Cancer")

