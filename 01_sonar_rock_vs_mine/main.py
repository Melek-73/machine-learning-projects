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
