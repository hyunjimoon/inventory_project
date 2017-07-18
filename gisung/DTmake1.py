import graphviz
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

sales = pd.read_csv("sales.csv")
# use_sales = pd.DataFrame(sales)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#
# sl_data = (np.array)(pd.concat([sales.year, sales.month, sales.date], axis=1))
# print(sl_data)
# print(sl_data.shape)
# print(sales.target.shape)

Xreshape = (pd.DataFrame)((pd.Series)([i for i in range(1,398)]))
print(Xreshape)
X0=sales.ix[:,'year':'date']
X1=sales.ix[:,'year':'month']
X2=sales.ix[:,'month':'date']
X_year = (pd.DataFrame)(sales.year)
X_month = (pd.DataFrame)(sales.month)
X_date = (pd.DataFrame)(sales.date)

X_curuse = Xreshape

X_train, X_test, y_train, y_test = train_test_split(X_curuse, sales.target, random_state=0)

tree=DecisionTreeRegressor(max_depth=5)
tree.fit(X_train,y_train)
allpredicted=tree.predict(X_curuse)

print("accuracy of train : {}".format(tree.score(X_train, y_train)))
print("accuracy of test : {} ".format(tree.score(X_test,y_test)))

print("특성 중요도:\n{}".format(tree.feature_importances_))

# 만약 1변수만 트리에 들어간다면 아래를 해제하여 시각화 가능. 다른경우는 없음. 애초에 평면이니까...
plt.figure()
plt.scatter(X_curuse, sales.target, c="darkorange", label ="origin")
plt.plot(X_curuse, allpredicted, c="cornflowerblue", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
plt.show()