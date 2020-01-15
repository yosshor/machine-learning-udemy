# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values



# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = np.round(imputer.transform(X[:, 1:3]), 2)
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
X[:, 0] = label_x.fit_transform(X[:, 0])
print(X)
onehot = OneHotEncoder(categorical_features = [0])
X = np.round(onehot.fit_transform(X).toarray(),1)
print(X)
label_y = LabelEncoder()
y = label_y.fit_transform(y)
print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
x = StandardScaler()
x_train = x.fit_transform(x_train)
print(x)
x_test = x.transform(x_test)




v = 2*0.5 +3
s = np.linspace(0,5,num =50)
f = np.linspace(0,10, num =30)
plt.scatter(f,s, v)
plt.figure()
plt.Line2D(f,s,v)






