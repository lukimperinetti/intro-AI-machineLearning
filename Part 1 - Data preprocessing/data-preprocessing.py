# Data preprocessing

# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data set
dataset = pd.read_csv('Data.csv')  # import cvs data file
X = dataset.iloc[:, :-1].values # Matrice X : that get all column except the last one
y = dataset.iloc[:, -1].values # Vector y : Get the dependente var that is the last one

# Fix empty data lines
from sklearn.impute import SimpleImputer # sickitLearn
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # mean strat because of clean data however use : Median
imputer.fit(X[:, 1:3]) # get idex where i need to create mean on empty values
X[:, 1:3] = imputer.transform(X[:, 1:3]) # create values

# Generate categoric variables



