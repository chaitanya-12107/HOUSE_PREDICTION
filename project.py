import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data.csv")

print("")


print("Shape:", df.shape)


df = df.drop(columns=["date", "street", "city", "statezip", "country"])


df = df.dropna()

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)


accuracy = model.score(X_test, y_test)
print("Model Accuracy (R2 Score):", accuracy)


plt.scatter(X_test["sqft_living"], y_test, alpha=0.3, label="Actual Price")
plt.scatter(X_test["sqft_living"], predictions, alpha=0.3, label="Predicted Price")
plt.xlabel("Square Feet Living")
plt.ylabel("House Price")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()
input("Press any key to exit...")

