
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pickle
import os

os.chdir(os.path.dirname(__file__))

iris=load_iris()

print(iris.keys())


x= iris.data
y=iris.target

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=40)


sscaler= StandardScaler()
x_trn_scaled=sscaler.fit_transform(x_train)
x_tst_scaled= sscaler.transform(x_test)


knn_clas= KNeighborsClassifier(n_neighbors=5)
knn_clas.fit(x_trn_scaled,y_train)

y_pred=knn_clas.predict(x_tst_scaled)

accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


with open("model.pkl", "wb") as f:
    pickle.dump(knn_clas, f)