#sklearn: ML library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
#1 dataset investigation
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target
#print(df)
#2 picking the model - KNN classification
#3 tranning the model
knn = KNeighborsClassifier(n_neighbors=9)#model | dont forget neighbor parameter
X = cancer.data #features
y = cancer.target #target
#train test sprit
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
#standardization
scaller = StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)


knn.fit(X_train, y_train) # this function using data samples + target trains KNN algorithm
#4 Assesment of results
y_pred = knn.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("confision metrix:", confusion_matrix(y_test, y_pred))
#5 Adjustment of Hyperparameters
"""
KNN: Hyperparameter = K
K: 1,2,3 ... N
Accuracy: %A, %B, %C ...,
"""
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(k, accuracy_score(y_test, y_pred))
