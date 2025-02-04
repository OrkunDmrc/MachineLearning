import numpy as np
import pandas as pd

dataFrame = pd.read_excel("maliciousornot.xlsx")
print(dataFrame.head())
print(dataFrame.info())
print(dataFrame.describe())

import seaborn as sbn
sbn.countplot(x= "Type" , data = dataFrame)

y = dataFrame["Type"].values
x = dataFrame.drop(["Type"], axis = 1).values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 15)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(units = 30, activation = "relu"))
model.add(Dropout(0.6))#yüzde kaçında turn off yapıcağını belirtir
model.add(Dense(units = 15, activation = "relu"))
model.add(Dropout(0.6))
model.add(Dense(units = 15, activation = "relu"))
model.add(Dropout(0.6))
model.add(Dense(units = 1, activation = "sigmoid"))#sınıflandırmada çıkış layerı sigmoid
model.compile(loss = "binary_crossentropy", optimizer = "adam")
earlyStopping = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 50)
model.fit(x = x_train, y = y_train, epochs = 700, validation_data = (x_test, y_test), verbose = 1, callbacks = [earlyStopping])

loss = pd.DataFrame(model.history.history)
loss.plot()

predictions = model.predict(x_test)  # Get raw probabilities
predicted_classes = np.argmax(predictions, axis=1)  # Get class labels
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predicted_classes))
print(confusion_matrix(y_test, predicted_classes))
