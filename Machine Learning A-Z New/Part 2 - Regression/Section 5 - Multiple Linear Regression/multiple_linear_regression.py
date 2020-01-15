# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
x = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
print(y_test, y_pred)


#find p-value that p < ls = 0.05

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
#x = x[:, 1:]
regresor_OLS = sm.OLS(endog = y, exog = x).fit()
regresor_OLS.summary()

x_out = x
regresor_OLS = sm.OLS(endog = y, exog = x).fit()
regresor_OLS.summary()

x_out = x[:,[0, 1, 3, 4, 5]]
regresor_OLS = sm.OLS(endog = y, exog = x_out).fit()
regresor_OLS.summary()

x_out = x[:,[0, 3, 4, 5]]
regresor_OLS = sm.OLS(endog = y, exog = x_out).fit()
regresor_OLS.summary()

x_out = x[:,[0, 3, 5]]
regresor_OLS = sm.OLS(endog = y, exog = x_out).fit()
regresor_OLS.summary()

x_out = x[:,[0, 3]]
regresor_OLS = sm.OLS(endog = y, exog = x_out).fit()
regresor_OLS.summary()




import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
sl = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(X_opt, SL)

x_plt1 = x_train[:,3].reshape(40, 1)
model = LinearRegression()
model.fit(x_plt1, y_train)

plt.scatter(x_train[:,3], y_train, color = 'red')
plt.plot(x_plt1.reshape(len(x_plt1),1), model.predict(x_plt1), color = 'blue' )
plt.show()

x_plt = x_test[:,3].reshape(10,1)
plt.scatter(x_plt, y_test, color = 'blue')
plt.plot(x_plt , model.predict(x_plt), color = 'red')
plt.show()








