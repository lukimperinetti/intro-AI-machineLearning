# ----- Data preprocessing

# ----- Import library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer  # sickitLearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# ----- Import Data set
dataset = pd.read_csv('Data.csv')  # import cvs data file
X = dataset.iloc[:, :-1].values # Matrice X : that get all column except the last one
y = dataset.iloc[:, -1].values # Vector y : Get the dependente var that is the last one


# ----- Fix empty data lines
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # mean strat because of clean data however use : Median
imputer.fit(X[:, 1:3])  # get idex where i need to create mean on empty values
X[:, 1:3] = imputer.transform(X[:, 1:3])  # create values


# ----- Generate categoric variables
labelencoder_X = LabelEncoder() # for country
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Fit & transform !
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], 
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))


labelencoder_y = LabelEncoder() # for dep var
y = labelencoder_y.fit_transform(y) 

