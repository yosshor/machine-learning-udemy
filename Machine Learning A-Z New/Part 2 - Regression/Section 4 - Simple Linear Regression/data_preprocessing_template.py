
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
(x_train), (x_test), (y_train), (y_test ) = train_test_split(x, y, test_size = 1/3, random_state = 0)
#x_train = x_train.reshape(1,-1)
x_train = x_train.reshape(len(x_train),1)
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 1)
y_test = y_test.reshape(len(y_test), 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.reshape(len(x_train),1), y_train.reshape(len(y_train), 1))
#x_test = x_test.reshape(1, -1)
pred = model.predict(x_test.reshape(len(x_test),1))

#
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, pred)

plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train,model.predict(x_train))
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, pred)
plt.show()


#for multiply models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
x[:, -1] = label_x.fit_transform(x[:, -1])

hot_encoder = OneHotEncoder(categorical_features=[3])
x = hot_encoder.fit_transform(x).toarray()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_yesy = train_test_split(x, y, test_size = 0.25, random_state = 0 )
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

#
#plt.scatter(x_train[:,-1], y_train, color = 'red')
#plt.plot(x_train[:,-1], model.predict(x_train), color = 'blue')
#plt.show()

























