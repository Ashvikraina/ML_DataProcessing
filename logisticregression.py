from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
file = pd.read_csv('diabetes.csv')
x=np.array(file.drop(["Outcome"],axis=1))
y=np.array(file["Outcome"])
train_data,test_data,train_labels,test_labels=train_test_split(x,y,test_size=.1)
model=LogisticRegression(max_iter=1000)
model.fit(train_data,train_labels)
prediction=model.predict(test_data)
# print(prediction)
# print(test_labels)

# h=0
# g=0
# for e in range(len(prediction)):
#     if prediction[e]==test_labels[e]:
#         h=h+1
#         g=g+1
#     else:
#         g=g+1
# print(h/g)

accuracy = metrics.accuracy_score(test_labels,prediction)
# print(accuracy)

cf=metrics.confusion_matrix(test_labels,prediction)
print(cf)
