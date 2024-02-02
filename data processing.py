#importing librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#reading the csv file.
dataset=pd.read_csv("Data.csv")
x=dataset.iloc[:, :-1].values #independent variable dataset.iloc[rows(indices),columns(indices)]
y=dataset.iloc[:, -1] .values#dependent variable
#print(x,y,sep='\n')#for 
#Taking care of missing data.
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
#print(x)
#Encoding categorial data.
#Encoding the independent variable.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
#print(x)#prints transformed array of vectors of the csv file
#encoding dependent variables
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y=le.fit_transform(y)
print(y)


