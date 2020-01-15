# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = np.asarray(dataset.iloc[:, 1])
y = dataset.iloc[:, 2].values
x = np.atleast_2d(x).T
y = np.atleast_2d(y).T



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
model2 = PolynomialFeatures(degree = 3)
x_poly = model2.fit_transform(x)
model2.fit(x_poly, y)
linear2 = LinearRegression()
linear2.fit(x_poly, y)


#if we want a better curve we need to change the degree in PolynomialFeature
plt.scatter(x, y, color = 'blue')
plt.plot(x, model.predict(x),color = 'red')
plt.title("Truth or bluff (Linear Regression)")
plt.xlabel('Position lavel')
plt.ylabel('Salary')
plt.show()


#visualising the polynomial regression result
plt.scatter(x, y, color = 'red')
plt.plot(x, linear2.predict(x_poly), color = 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel('Position lavel')
plt.ylabel('Salary')
plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, linear2.predict(model2.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

model.predict(np.array(6.5).reshape(1,1))
linear2.predict(np.array(model2.fit_transform(np.atleast_2d(6.5))))#.reshape(len(model2.fit_transform(6.5)), 1))#.array(6.5).reshape())








