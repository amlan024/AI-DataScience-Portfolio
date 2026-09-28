import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv("data/wastewater.csv")


# Input features
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


# Target
y = data["Treatment_Efficiency"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Take user input
ph = float(input("Enter pH: "))
turbidity = float(input("Enter Turbidity: "))
temperature = float(input("Enter Temperature: "))
bod = float(input("Enter BOD: "))
cod = float(input("Enter COD: "))
tss = float(input("Enter TSS: "))
dissolved_oxygen = float(input("Enter Dissolved Oxygen: "))


# Create input data
new_data = pd.DataFrame(
    [
        [
            ph,
            turbidity,
            temperature,
            bod,
            cod,
            tss,
            dissolved_oxygen
        ]
    ],
    columns=[
        "pH",
        "Turbidity",
        "Temperature",
        "BOD",
        "COD",
        "TSS",
        "Dissolved_Oxygen"
    ]
)


# Predict treatment efficiency
prediction = model.predict(new_data)


print("\nPredicted Treatment Efficiency:")
print(round(prediction[0], 2), "%")