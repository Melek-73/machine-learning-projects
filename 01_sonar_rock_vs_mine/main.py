#%% Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path

#%% Data Collection and Data Processing
#loading the dataset to a pandas Dataframe
BASE_DIR = Path(r"C:\Personnel\ML\01_sonar_rock_vs_mine")
DATA_PATH = BASE_DIR / "sonar data.csv"
sonar_data = pd.read_csv(DATA_PATH, header=None)
print(sonar_data.head())
print("Shape:", sonar_data.shape)
print(sonar_data.describe())
print("Target counts:")
print(sonar_data[60].value_counts())

sonar_data.groupby(60).mean()

# separating data and Labels
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]
print("X=",X)
print("Y=",Y)

X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)
print(X_train)
print(Y_train)

model = LogisticRegression()
#training the logistic Regression model with training data
model.fit(X_train,Y_train)

#accuracy on training data
Y_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train , Y_train_prediction)
print('Accuracy on training data :' , training_data_accuracy)

#accuracy on test data
Y_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test , Y_test_prediction)
print('Accuracy on test data :' , test_data_accuracy)

#%%Making Predictive System
input_data = (0.0180,0.0444,0.0476,0.0698,0.1615,0.0887,0.0596,0.1071,0.3175,0.2918,0.3273,0.3035,0.3033,0.2587,0.1682,0.1308,0.2803,0.4519,0.6641,0.7683,0.6960,0.4393,0.2432,0.2886,0.4974,0.8172,1.0000,0.9238,0.8519,0.7722,0.5772,0.5190,0.6824,0.6220,0.5054,0.3578,0.3809,0.3813,0.3359,0.2771,0.3648,0.3834,0.3453,0.2096,0.1031,0.0798,0.0701,0.0526,0.0241,0.0117,0.0122,0.0122,0.0114,0.0098,0.0027,0.0025,0.0026,0.0050,0.0073,0.0022)

#changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

predection = model.predict(input_data_reshaped)
print(predection)

if predection[0] == 'R' :
  print('The object is Rock')
else :
  print('The object is a mine')