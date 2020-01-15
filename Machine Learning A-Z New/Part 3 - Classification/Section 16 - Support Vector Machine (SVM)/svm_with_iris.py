# -*- coding: utf-8 -*-
"""
@author: yoss
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = sns.load_dataset("iris")
print(iris.head())
y = iris.species
x = iris.drop('species',axis = 1)
sns.pairplot(iris, hue="species",palette="bright")

df=iris[(iris['species'] != 'virginica')]
df=df.drop(['sepal_length','sepal_width'], axis=1)
df.head()

#let's convert categorical values to numerical target
df=df.replace('setosa', 0)
df=df.replace('versicolor', 1)
x=df.iloc[:,0:2]
y=df['species']
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=50, cmap='autumn')

from sklearn.svm import SVC
model = SVC(kernel='linear', C = 1E10)
model.fit(x, y)
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=50, cmap='autumn')
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1])


ax = plt.gca()
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=50, cmap='autumn')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(xx.shape)
ax.contour(xx, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

red_sample=df.sample(frac=0.7)
x=red_sample.iloc[:,0:2]
y=red_sample['species']
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=50, cmap='autumn')



from sklearn.datasets.samples_generator import make_circles
x, y = make_circles(100, factor=.1, noise=.1)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')

model=SVC(kernel='linear').fit(x, y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(xx.shape)
# plot decision boundary and margins
ax.contour(xx, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

from mpl_toolkits import mplot3d
#setting the 3rd dimension with RBF centered on the middle clump
r = np.exp(-(x ** 2).sum(1))
ax = plt.subplot(projection='3d')
ax.scatter3D(x[:, 0], x[:, 1], r, c=y, s=50, cmap='autumn')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')



model=SVC(kernel='rbf').fit(x, y)

if isinstance(x, pd.DataFrame):
    print('yes the type of x is DataFrame')
    x = x.values  
    
    
ax = plt.gca()
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(xx.shape)
ax.contour(xx, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


from mpl_toolkits import mplot3d
#setting the 3rd dimension with RBF centered on the middle clump
r = np.exp(-(x ** 2).sum(1))
ax = plt.subplot(projection='3d')
ax.scatter3D(x[:, 0], x[:, 1], r, c=y, s=50, cmap='autumn')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')


















