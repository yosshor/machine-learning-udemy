# Random Forest Regression

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
x = np.atleast_2d(x).T
y = np.atleast_2d(y)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0 )
regressor.fit(x, y)
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
print('The prediction is {} for {} estimators'.format(np.round(y_pred, 2), regressor.n_estimators))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bloof ( Random Foreset Regressor)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()




regressor = RandomForestRegressor(n_estimators = 300, random_state = 0 )
regressor.fit(x, y)
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
print('The prediction is {} for {} estimators'.format(np.round(y_pred, 2), regressor.n_estimators))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bloof ( Random Foreset Regressor)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()