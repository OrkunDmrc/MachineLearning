import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
print(dataFrame.head())
sbn.pairplot(dataFrame)
plt.show()

from sklearn.model_selection import train_test_split
#y -> wx + b
#y -> label
y = dataFrame["Fiyat"].values
#x feature
x = dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=15)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#scalling tüm verileri 0 ile 1 arasına alır daha kolay işlem yapabilmek için
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="rmsprop", loss="mse")
model.fit(x_train, y_train, epochs=250)
loss = model.history.history['loss']
sbn.lineplot(x = range(len(loss)), y = loss)
plt.show()

trainLoss = model.evaluate(x_train, y_train, verbose=0)
testLoss = model.evaluate(x_test, y_test, verbose=0)
print(trainLoss, testLoss)

testTahminleri = model.predict(x_test)
tahminDf = pd.DataFrame(y_test, columns=["gercek"])
tahminDf["tahmin"] = testTahminleri
print(tahminDf)

from sklearn.metrics import mean_absolute_error, mean_squared_error
absoluteError = mean_absolute_error(tahminDf["gercek"], tahminDf["tahmin"])
squiredError = mean_squared_error(tahminDf["gercek"], tahminDf["tahmin"])
print(absoluteError, squiredError)
description = dataFrame.describe()
print(description)

bisikletOzellikleri = [[1751,1750]]
bisikletOzellikleri = scaler.transform(bisikletOzellikleri)
output = model.predict(bisikletOzellikleri)
print(output)

from tensorflow.keras.models import load_model
#model.save("bisiklet_modeli.h5")
model.save('bisiklet_modeli.keras')
savedModel = load_model("bisiklet_modeli.keras")

output = savedModel.predict(bisikletOzellikleri)
print(output)

