import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
data = pd.read_csv("data/wastewater.csv")


# Select input features
X = data[
    [
        "pH",
        "Turbidity",
        "Temperature",
        "BOD",
        "COD",
        "TSS",
        "Dissolved_Oxygen"
    ]
]


# Select target
y = data["Treatment_Efficiency"]


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate errors
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Display results
print("WASTE WATER TREATMENT EFFICIENCY PREDICTION")
print("--------------------------------------------")

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nMean Absolute Error:")
print(mae)

print("\nR2 Score:")
print(r2)