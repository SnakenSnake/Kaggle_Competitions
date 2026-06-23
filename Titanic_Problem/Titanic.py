
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

train_data=pd.read_csv("train.csv")
model=Sequential([
    Dense(units=128,activation='relu'),
    Dense(units=64,activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=16,activation='relu'),
    Dense(units=8,activation='relu'),
    Dense(units=4,activation='relu'),
    Dense(units=2,activation='relu'),
    Dense(units=1,activation='sigmoid')
])

X=train_data.drop(columns=['Survived','Name','Ticket','Cabin','Embarked'])
X = X.fillna(X.mean(numeric_only=True))
X['Sex'] = X['Sex'].map({
    'male': 0,
    'female': 1
})
Y=train_data[['Survived']].to_numpy()
X=X.to_numpy()


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X_scaled,Y,epochs=10)
loss , acc =model.evaluate(X_scaled,Y)
train_predict=model.predict(X)
test_data=pd.read_csv('test.csv')
X_test=test_data.drop(columns=['Name','Ticket','Cabin','Embarked'])
X_test['Sex']=X_test['Sex'].map({
    "male":0,
    "female":1
})
X_test=X_test.fillna(X_test.mean())
X_test=X_test.to_numpy()
X_test_scaled=scaler.fit_transform(X_test)
predict=model.predict(X_test_scaled)
predict=(predict>0.5).astype(int)
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predict.flatten()
})

submission.to_csv('submission.csv', index=False)
print(submission.head())
