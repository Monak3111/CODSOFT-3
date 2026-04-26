import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df= pd.read_csv("advertising.csv")

df.dtypes

df.info()

df.isnull().sum()

df.dropna()

X= df[['TV', 'Radio', 'Newspaper']]

y= df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

model= LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, "advertising.csv.pkl")
print("Model is saved")

