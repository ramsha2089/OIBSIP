import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv(r"C:/project/ramsha/ramsha/New folder/car data.csv")
print
# Replace categorical variables with numerical values
df.replace({'Selling_type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = df['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
random_forest.fit(X_train, Y_train)

# Make predictions on the test data
predictions = random_forest.predict(X_test)

# Calculate the R-squared score
r2 = r2_score(Y_test, predictions)
print("R-squared:", r2)

# Visualize Predictions (Optional)
plt.scatter(Y_test, predictions)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs. Predicted Selling Price")
plt.show()
