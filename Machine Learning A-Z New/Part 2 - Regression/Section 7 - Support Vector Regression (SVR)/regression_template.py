import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x.reshape(len(x), 1))
y = sc_y.fit_transform(y.reshape(len(y), 1))

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', degree = 3)
regressor.fit(x, y)

# regressor.predict()
y_pred = sc_y.inverse_transform(regressor.predict(sc_y.transform(np.array([[6.5]]))))
print(y_pred)

plt.scatter(x, y, color ='red')
#plt.plot(x, y)
#plt.plot(x, y, marker='o', markersize=3, color="red")
#plt.plot(x, y_pred, color = 'black')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bloof (SVR module) ')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






























