import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

dataFrame = pd.read_excel("merc.xlsx")
print(dataFrame.head())
print(dataFrame.describe())
print(dataFrame.isnull().sum())
dataFrameCorr = dataFrame.drop("transmission", axis=1)  # axis=1 means column
print(dataFrameCorr.corr())
print(dataFrameCorr.corr()["price"].sort_values())

sbn.displot(dataFrame["price"])
sbn.scatterplot(x = dataFrame["mileage"], y = dataFrame["price"])
dataFrame.sort_values("price", ascending=False).head(20)
dataFrame.sort_values("price", ascending=True).head(20)
print(len(dataFrame) * 0.01) # 1% remove

#dataFrame2 = dataFrame.drop(dataFrame.sort_values("price", ascending=False).head(131).index)
dataFrame2 = dataFrame.sort_values("price", ascending=False).iloc[131:]
print(dataFrame2.describe())
sbn.displot(dataFrame2["price"])
dataFrameCorr2 = dataFrame2.drop("transmission", axis=1)
dataFrameCorr2.groupby("year").mean()["price"]
dataFrame = dataFrameCorr2[dataFrameCorr2["year"] != 1970]
y = dataFrame["price"].values
x = dataFrame.drop("price",axis = 1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(12, activation = "relu"))
model.add(Dense(12, activation = "relu"))
model.add(Dense(12, activation = "relu"))
model.add(Dense(12, activation = "relu"))
model.add(Dense(1))
model.compile(optimizer = "adam", loss = "mse")
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300)

loss = pd.DataFrame(model.history.history)
print(loss.head())
loss.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error
y_pred = model.predict(x_test)
print(mean_absolute_error(y_test, y_pred))





