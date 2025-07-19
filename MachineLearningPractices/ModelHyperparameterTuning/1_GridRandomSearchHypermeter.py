from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#KNN
knn = KNeighborsClassifier()
knn_param = {"n_neighbors": np.arange(2,31)}
knn_grid_search = GridSearchCV(knn,knn_param)
knn_grid_search.fit(X_train, y_train)
print("KNN Grid Search Best Parameters:",knn_grid_search.best_params_)
print("KNN Grid Best Accuracy:",knn_grid_search.best_score_)

knn_random_search = RandomizedSearchCV(knn,knn_param, n_iter = 10)
knn_random_search.fit(X_train,y_train)
print("KNN Random Search Best Parameters:",knn_random_search.best_params_)
print("KNN Random Best Accuracy:",knn_random_search.best_score_)

#DT
dt = DecisionTreeClassifier()
tree_param = {"max_depth":[3,5,7],
              "max_leaf_nodes":[None,5,10,20,30,50]}

tree_grid_serach = GridSearchCV(dt, tree_param)
tree_grid_serach.fit(X_train,y_train)
print("DT Grid Search Best Parameters:",tree_grid_serach.best_params_)
print("DT Grid Best Accuracy:",tree_grid_serach.best_score_)

tree_random_search = RandomizedSearchCV(dt,tree_param,n_iter = 10)
tree_random_search.fit(X_train,y_train)
print("DT Random Search Best Parameters:",tree_random_search.best_params_)
print("DT Random Best Accuracy:",tree_random_search.best_score_)

