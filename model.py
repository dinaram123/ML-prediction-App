import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
data = {
    "age": [22, 25, 30, 35, 40, 45, 50, 55],
    "bmi": [18, 22, 25, 28, 30, 26, 32, 29],
    "charges": [2200, 2500, 3000, 3500, 4000, 3800, 4500, 4200]
}
df = pd.DataFrame(data)

X = df[["age", "bmi"]]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained and saved as model.pkl")
