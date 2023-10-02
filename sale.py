import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv(r"C:/project/ramsha/ramsha/New folder/Advertising.csv")
print(data)

# Assuming you have a DataFrame named 'data' and you want to drop the 'Sales' column
X = data.drop(["Sales"], axis=1)

Y=data["Sales"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
r2=r2_score(Y_test,Y_pred)
print("R2_score:",r2)
Y_pred=model.predict([[7,57.5,32.8,23.5]])
print(Y_pred)