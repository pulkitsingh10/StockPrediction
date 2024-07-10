import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


import quandl
data = quandl.get("NSE/TATAGLOBAL")
print(data.head(10))


plt.figure(figsize=(16,8))
plt.plot(data['Close'],label='Closing Price')
plt.show()

data['Open - Close']=data['Open']-data['Close']
data['High - Low']=data['High']-data['Low']
data=data.dropna()
X=data[['Open - Close','High - Low']]
X.head()

Y=np.where(data['Close'].shift(-1)>data['Close'],1,-1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=44)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


params={ 'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn=neighbors.KNeighborsClassifier()
model=GridSearchCV(knn, params, cv=5)

model.fit(X_train, y_train)

accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

print('Train_data Accuracy: %.2f' %accuracy_train)
print('Test_data Accuracy: %.2f' %accuracy_test)

predictions_classification = model.predict(X_test)
actual_predicted_data = pd.DataFrame({'Actual class':y_test,'predicted class':predictions_classification})
print(actual_predicted_data.head(10))

y=data['Close']

from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25,random_state=44)
params={ 'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg=neighbors.KNeighborsRegressor()
model_reg=GridSearchCV(knn_reg, params, cv=5)

model_reg.fit(X_train_reg, y_train_reg)
predictions=model_reg.predict(X_test_reg)

print(predictions)

rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
rms

valid=pd.DataFrame({'Actual Close':y_test_reg, 'Predicted Close':predictions})
print(valid.head(10))



                               







