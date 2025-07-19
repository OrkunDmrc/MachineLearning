from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

dt = DecisionTreeClassifier()
tree_param = {"max_depth":[3,5,7]}

#KFOLD
kf = KFold(n_splits = 10)
tree_grid_serach_kf = GridSearchCV(dt, tree_param,cv = kf)
tree_grid_serach_kf.fit(X_train,y_train)
print("KF Grid Search Best Parameters:",tree_grid_serach_kf.best_params_)
print("KF Grid Best Accuracy:",tree_grid_serach_kf.best_score_)

#LOO
loo = LeaveOneOut()
tree_grid_serach_loo = GridSearchCV(dt, tree_param,cv = loo)
tree_grid_serach_loo.fit(X_train,y_train)
print("LOO Grid Search Best Parameters:",tree_grid_serach_loo.best_params_)
print("LOO Grid Best Accuracy:",tree_grid_serach_loo.best_score_)
