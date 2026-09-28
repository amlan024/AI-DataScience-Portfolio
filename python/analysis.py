import pandas as pd
import matplotlib.pyplot as plt


# Load wastewater dataset
data = pd.read_csv("data/wastewater.csv")

# Display complete dataset
print("WASTEWATER DATASET")
print(data)


# Display first 5 rows
print("\nFIRST 5 ROWS")
print(data.head())


# Display number of rows and columns
print("\nDATASET SHAPE")
print(data.shape)


# Display column names
print("\nCOLUMN NAMES")
print(data.columns)


# Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())


# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(data.describe())





# BOD vs Treatment Efficiency
plt.scatter(data["BOD"], data["Treatment_Efficiency"])

plt.xlabel("BOD")
plt.ylabel("Treatment Efficiency (%)")
plt.title("BOD vs Treatment Efficiency")

plt.savefig("../images/Images.png")

plt.show()