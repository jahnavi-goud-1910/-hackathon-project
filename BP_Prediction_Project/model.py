import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Dummy but valid medical dataset
data = {
    "gender": [0,1,0,1,0,1,0,1],
    "age_group": [1,2,3,4,2,3,4,1],
    "systolic": [118,145,160,180,150,170,190,125],
    "diastolic": [78,92,100,120,95,105,125,80],
    "stage": [0,1,2,3,1,2,3,0]  # 0-Normal,1-Stage1,2-Stage2,3-Crisis
}

df = pd.DataFrame(data)

X = df[["gender", "age_group", "systolic", "diastolic"]]
y = df["stage"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("hypertension_model.pkl", "wb"))

print("Model trained & saved successfully")
