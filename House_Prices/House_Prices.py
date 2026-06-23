

import numpy as np 
import pandas as pd 
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

train_data=pd.read_csv('train.csv')
train_data.head()
test_data=pd.read_csv('test.csv')
model=Sequential([
    Dense(units=32,activation='relu'),
    Dense(units=16,activation='relu'),
    Dense(units=1,activation='linear'),
])
y_scaler=StandardScaler()
X_train=train_data[['MSSubClass','LotArea','ScreenPorch','PoolArea','MiscVal']]
Y_train=train_data['SalePrice']
Y_scaled = y_scaler.fit_transform(Y_train.values.reshape(-1,1))
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_train)
model.compile(
    optimizer='Adam',
    loss='mean_squared_error',
)
model.fit(X_scaled,Y_scaled,epochs=10)
loss=model.evaluate(X_scaled,Y_scaled)
X_test=test_data[['MSSubClass','LotArea','ScreenPorch','PoolArea','MiscVal']]
X_test_scaled=scaler.transform(X_test)
predict=model.predict(X_test_scaled)
pred=y_scaler.inverse_transform(predict)
submission=pd.DataFrame({
    'Id':test_data['Id'],
    'SalePrice':pred.flatten()
})
submission.to_csv('submission.csv',index=False)
print(submission.head())