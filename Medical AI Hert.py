import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

path = "Heart data.csv"
df = pd.read_csv(path)
print("Data ", df)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Features :", x)
print("Target :", y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

print("Model Accuracy:", accuracy)
