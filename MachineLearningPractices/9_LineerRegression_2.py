from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
#diabetes = load_diabetes()

diabetes_X, diabetes_y = load_diabetes(return_X_y = True)
diabetes_X = diabetes_X[:,np.newaxis,2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

lin_reg = LinearRegression()
lin_reg.fit(diabetes_X_train,diabetes_y_train)

y_pred = lin_reg.predict(diabetes_X_test)
mse = mean_squared_error(diabetes_y_test,y_pred)
r2 = r2_score(diabetes_y_test,y_pred)
print("mse:",mse)
print("r2:",r2)
plt.scatter(diabetes_X_test,diabetes_y_test,color="black")
plt.plot(diabetes_X_test,y_pred,color="blue")
plt.show()
print(diabetes_X_test[0])
print(lin_reg.predict([[0.5]]))



