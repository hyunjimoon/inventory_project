import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.neighbors import KNeighborsRegressor


sales = pd.read_csv("sales.csv")


X0=sales.ix[:,'year':'date']
X1=sales.ix[:,'year':'month']
X2=sales.ix[:,'month':'date']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, sales.target, random_state=5)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("prediction of test set :\n{}".format(reg.predict(X_test)))
print("accuracy = r^2 : {}".format(reg.score(X_test,y_test)))

fig, axes = plt.subplots(1,3, figsize = (15,4))


for n_neighbor, ax in zip([1,3,9],axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbor)
    reg.fit(X_train, y_train)
    ax.plot(X2, reg.predict(X2))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=2)
    ax.plot(X_test, y_test, 'o', c=mglearn.cm2(1), markersize=2)
    ax.set_title("{} trainscore : {:.2f} testscore : {:.2f}".format(n_neighbor, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("feature")
    ax.set_xlabel("target")

axes[0].legend(["predict", "train data/target", "test data/target"], loc="best")

plt.show()