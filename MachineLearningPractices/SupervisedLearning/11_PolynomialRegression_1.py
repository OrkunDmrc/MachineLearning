import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = 4 * np.random.rand(100,1)
y = 2 + 3*X**3 + np.random.rand(100,1)

poly_feat = PolynomialFeatures(degree=3)
X_poly = poly_feat.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly,y)

X_test = np.linspace(0,4,100).reshape(-1,1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.scatter(X,y,color="blue")
plt.plot(X_test, y_pred, color = "red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression Model")
plt.show()
