import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load wastewater dataset

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


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create models

linear_model = LinearRegression()

decision_tree_model = DecisionTreeRegressor(
    random_state=42
)

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train models

linear_model.fit(X_train, y_train)

decision_tree_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)


# Make predictions

linear_prediction = linear_model.predict(X_test)

decision_tree_prediction = decision_tree_model.predict(X_test)

random_forest_prediction = random_forest_model.predict(X_test)


# Calculate evaluation metrics

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


decision_tree_mae = mean_absolute_error(
    y_test,
    decision_tree_prediction
)

decision_tree_r2 = r2_score(
    y_test,
    decision_tree_prediction
)


random_forest_mae = mean_absolute_error(
    y_test,
    random_forest_prediction
)

random_forest_r2 = r2_score(
    y_test,
    random_forest_prediction
)


# Display results

print("\nWASTEWATER ML MODEL COMPARISON")
print("--------------------------------")

print("\nTest Samples:")
print(len(y_test))


print("\nActual Values:")
print(y_test.values)


print("\nLinear Regression")
print("-----------------")
print("Predicted Values:")
print(linear_prediction)
print("MAE:", linear_mae)
print("R2 Score:", linear_r2)


print("\nDecision Tree Regression")
print("------------------------")
print("Predicted Values:")
print(decision_tree_prediction)
print("MAE:", decision_tree_mae)
print("R2 Score:", decision_tree_r2)


print("\nRandom Forest Regression")
print("------------------------")
print("Predicted Values:")
print(random_forest_prediction)
print("MAE:", random_forest_mae)
print("R2 Score:", random_forest_r2)