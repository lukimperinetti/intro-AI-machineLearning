# ----- Data preprocessing

# ----- Import library
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer  # sickitLearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# ----- Import Data set
dataset = pd.read_csv('Data.csv')  # import cvs data file
# Matrice X : that get all column except the last one
X = dataset.iloc[:, :-1].values
# Vector y : Get the dependente var that is the last one
y = dataset.iloc[:, -1].values


# ----- Fix empty data lines
# mean strat because of clean data however use : Median
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])  # get idex where i need to create mean on empty values
X[:, 1:3] = imputer.transform(X[:, 1:3])  # create values


# ----- Generate categoric variables
labelencoder_X = LabelEncoder()  # for country
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # Fit & transform !
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

labelencoder_y = LabelEncoder()  # for dep var
y = labelencoder_y.fit_transform(y)


# ----- Divide the dataSet in Training set and test Set
# random_state is just for the course to have the same data as the theacher
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# ----- Feature Scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
