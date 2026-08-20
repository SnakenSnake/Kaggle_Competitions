import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

numeric_cols = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time"
]


def preprocessing(data):
    data=data.copy()
    for col in numeric_cols:
        data[col]=data[col].fillna(train_means[col])
    data["gender"]=data["gender"].fillna("Other")
    data=pd.get_dummies(data, columns=["gender"],dtype=int)
    ratio = data["daily_screen_time_hours"] / data["sleep_hours"]
    data.loc[
    (ratio <= 0.5) & data["stress_level"].isna(),
    "stress_level"
] = "Low"
    data.loc[
    (ratio > 0.5) & (ratio <= 0.9) & data["stress_level"].isna(),
    "stress_level"
    ] = "Medium"
    data.loc[
    (ratio > 0.9) & data["stress_level"].isna(),
    "stress_level"
    ] = "High"
    data=pd.get_dummies(data,columns=["stress_level"],dtype=int)
    impact_ratio=data["work_study_hours"]/data["daily_screen_time_hours"]
    data.loc[(impact_ratio>0.5)&data["academic_work_impact"].isna(),"academic_work_impact"]="No"    
    data.loc[(impact_ratio<=0.5)&data["academic_work_impact"].isna(),"academic_work_impact"]="Yes"    
    data=pd.get_dummies(data,columns=["academic_work_impact"],dtype=int)
    
    return data

y=train_data["addicted_label"]
X=train_data.drop(columns=["addicted_label","id"])
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
train_means=X_train[numeric_cols].mean()
X_train=preprocessing(X_train)
X_test=preprocessing(test_data)
X_val=preprocessing(X_val)

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

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_depth=10,
    min_samples_leaf=6
)

model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

print("Training accuracy:", train_acc)
print("Validation accuracy:", val_acc)

predict = model.predict(X_test).astype(int)

submission = pd.DataFrame({
    'id': pd.read_csv('test.csv')['id'],
    'addicted_label': predict.flatten()
})

submission.to_csv('submission.csv', index=False)
print(submission.head())