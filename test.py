import sklearn.linear_model
import numpy as np
import matplotlib.pyplot as plt

reg = sklearn.linear_model.Ridge()
X = np.array([[1, 2], [3, 4], [5, 6], [2, 3], [4, 5], [6, 7]])
Y = np.array([2, 4, 6, 8, 10, 13]).reshape(-1, 1)
print(X)
reg.fit(X, Y)

print(reg.coef_)
print(reg.predict(np.array([8, 9]).reshape(1, -1)))
print(reg.score(X, Y))
yPred = reg.predict(X)

print(X.shape)
print(Y.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.scatter(X[:, 0], X[:, 1], Y)
ax.plot_surface(X[:, 0], X[:, 1], yPred)
#plt.plot(X[:, 0], X[:, 1], yPred)
plt.show()
