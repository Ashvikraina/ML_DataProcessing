import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
# x=np.linspace(1,10,50).reshape(-1,1)
# y=np.linspace(1,10,50)+np.random.random(50)
# # plt.scatter(x,y,alpha=.7)
# # plt.show()
# model=LinearRegression()
# model.fit(x,y)
# y_predict=model.predict(x)
# # plt.scatter(x,y,alpha=.7)
# # plt.plot(x,y_predict,color="red",linewidth=1)
# # plt.show()
# print(model.coef_,model.intercept_)
# value=model.predict([[20]])
# print(value)
# print('Mean squared error',mean_squared_error(y,y_predict))
# print('Coefficient of determination/R2 score',r2_score(y,y_predict))
# x=np.append(x,[20])
# y_predict=np.append(y_predict,value)
# y=np.append(y,value)
# plt.scatter(x,y,alpha=.7)
# plt.plot(x,y_predict,color="red",linewidth=1)
# plt.show()

# population=pd.read_csv('population_total_long.csv')
# world=population.groupby(["Year"])["Count"].sum()
# world=world.reset_index()
# x=world['Year']
# y=world["Count"]
# x=x.to_numpy()
# x=np.reshape(x,(-1,1))
# model=LinearRegression()
# model.fit(x,y)
# y_predict=model.predict(x)
# value=model.predict([[2050]])
# x=np.append(x,[2050])
# y_predict=np.append(y_predict,value)
# y=np.append(y,value)
# plt.scatter(x,y,alpha=.7)
# plt.plot(x,y_predict,color="red",linewidth=1)
# plt.show()

file = pd.read_csv('diabetes.csv')
x=np.array(file.drop(["Outcome"],axis=1))
y=np.array(file["Outcome"])
# model = LogisticRegression()
# model.fit(X_train,Y_train)