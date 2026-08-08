import numpy as np
import pandas as pd
# import tensorflow as tf
# from keras.layers import Dense
# from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# combined = pd.concat([train_data, test_data])

# combined["Group"] = combined["PassengerId"].str.split("_").str[0]

# combined["GroupSize"] = combined.groupby("Group")["Group"].transform("count")
# print(train_data.head())
# print(train_data["HomePlanet"].nunique())
# train_data = combined.iloc[:len(train_data)].copy()
# test_data = combined.iloc[len(train_data):].copy()
train_data = pd.get_dummies(train_data, columns=["HomePlanet"], dtype=int)
# print(train_data.head())

train_data = train_data.drop(columns=["PassengerId", "Name"])
# print(train_data["Cabin"].nunique())
train_data[['Deck', 'Num', 'Side']] = (
    train_data["Cabin"].str.split("/", expand=True))
train_data["Num"] = pd.to_numeric(train_data["Num"])
train_data.drop(columns=["Cabin"], inplace=True)
# print(train_data.tail())
train_data = pd.get_dummies(train_data, columns=['Deck'], dtype=int)
train_data = pd.get_dummies(train_data, columns=['Side'], dtype=int)
train_data = pd.get_dummies(train_data, columns=["Destination"], dtype=int)
y = train_data["Transported"].astype(int)
X = train_data.drop(columns=["Transported"])
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# print(pd.crosstab(
#     X_train["Destination_TRAPPIST-1e"],
#     y_train
# ))
# print(X_train["Destination_TRAPPIST-1e"].value_counts())
# print(y_train.value_counts())
X_train['CryoSleep'] = X_train["CryoSleep"].fillna(False)
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_train["VIP"] = X_train["VIP"].fillna(False)
X_train['RoomService'] = X_train['RoomService'].fillna(
    X_train['RoomService'].mean())
X_train['FoodCourt'] = X_train['FoodCourt'].fillna(X_train['FoodCourt'].mean())
X_train['ShoppingMall'] = X_train['ShoppingMall'].fillna(
    X_train['ShoppingMall'].mean())
X_train['Spa'] = X_train['Spa'].fillna(X_train['Spa'].mean())
X_train['VRDeck'] = X_train['VRDeck'].fillna(X_train['VRDeck'].mean())
X_train['Num'] = X_train['Num'].fillna(X_train['Num'].mean())
X_train["TotalSpending"] = (X_train["RoomService"]+X_train["FoodCourt"] +
                            X_train["ShoppingMall"]+X_train["Spa"]+X_train["VRDeck"])
X_train["NoSpending"] = (X_train["TotalSpending"] == 0).astype(int)
# print(X_train.isna().sum())
# plt.scatter(y_train,X_train['Age'])
# plt.xlabel('Transported')
# plt.ylabel('Age')
# plt.show()
X_test=test_data.copy()
X_test = pd.get_dummies(X_test, columns=["HomePlanet"], dtype=int)
X_test.drop(columns=["PassengerId", "Name"], inplace=True)
X_test[['Deck', 'Num', 'Side']] = (X_test["Cabin"].str.split("/", expand=True))
X_test["Num"] = pd.to_numeric(X_test["Num"])
X_test.drop(columns=["Cabin"], inplace=True)
X_test = pd.get_dummies(X_test, columns=['Deck'], dtype=int)
X_test = pd.get_dummies(X_test, columns=['Side'], dtype=int)
X_test = pd.get_dummies(X_test, columns=["Destination"], dtype=int)
X_test['CryoSleep'] = X_test["CryoSleep"].fillna(False)
X_test['Age'] = X_test['Age'].fillna(X_train['Age'].mean())
X_test["VIP"] = X_test["VIP"].fillna(False)
X_test['RoomService'] = X_test['RoomService'].fillna(
    X_train['RoomService'].mean())
X_test['FoodCourt'] = X_test['FoodCourt'].fillna(X_train['FoodCourt'].mean())
X_test['ShoppingMall'] = X_test['ShoppingMall'].fillna(
    X_train['ShoppingMall'].mean())
X_test['Spa'] = X_test['Spa'].fillna(X_train['Spa'].mean())
X_test['VRDeck'] = X_test['VRDeck'].fillna(X_train['VRDeck'].mean())
X_test['Num'] = X_test['Num'].fillna(X_train['Num'].mean())
X_test["TotalSpending"] = (X_test["RoomService"]+X_test["FoodCourt"] +
                           X_test["ShoppingMall"]+X_test["Spa"]+X_test["VRDeck"])
X_test["NoSpending"] = (X_test["TotalSpending"] == 0).astype(int)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

X_val['CryoSleep'] = X_val["CryoSleep"].fillna(False)
X_val['VIP'] = X_val["VIP"].fillna(False)

X_val['Age'] = X_val['Age'].fillna(X_train['Age'].mean())
X_val['RoomService'] = X_val['RoomService'].fillna(
    X_train['RoomService'].mean())
X_val['FoodCourt'] = X_val['FoodCourt'].fillna(X_train['FoodCourt'].mean())
X_val['ShoppingMall'] = X_val['ShoppingMall'].fillna(
    X_train['ShoppingMall'].mean())
X_val['Spa'] = X_val['Spa'].fillna(X_train['Spa'].mean())
X_val['VRDeck'] = X_val['VRDeck'].fillna(X_train['VRDeck'].mean())
X_val['Num'] = X_val['Num'].fillna(X_train['Num'].mean())
X_val["TotalSpending"] = (
    X_val["RoomService"]
    + X_val["FoodCourt"]
    + X_val["ShoppingMall"]
    + X_val["Spa"]
    + X_val["VRDeck"]
)
X_val["NoSpending"] = (X_val["TotalSpending"] == 0).astype(int)
X_train, X_val = X_train.align(
    X_val,
    join="left",
    axis=1,
    fill_value=0
)

X_train, X_test = X_train.align(
    X_test,
    join='left',
    axis=1,
    fill_value=0
)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# print(X_train)

# model = Sequential([
#     Dense(units=64,activation="relu"),
#     Dense(units=32,activation="relu"),
#     Dense(units=1,activation="sigmoid"),
# ])
# for col in X_train.columns:
#     if X_train[col].equals(y_train):
#         print("LEAKAGE:", col)
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=8,
    min_samples_leaf=4
)

# model.fit(X_train,y_train,epochs=50)

model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

print("Training accuracy:", train_acc)
print("Validation accuracy:", val_acc)

# model.compile(
#     optimizer='Adam',
#     loss='binary_crossentropy',
#     metrics=["accuracy"]
# )

# X_test=scaler.transform(X_test)
predict = model.predict(X_test).astype(bool)
# predict=(predict>=0.5).astype(bool)
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Transported': predict.flatten()
})

submission.to_csv('submission.csv', index=False)
print(submission.head())
