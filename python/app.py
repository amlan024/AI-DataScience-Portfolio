from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

load_dotenv()

app = Flask(__name__)
CORS(app)


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


# Create and train ML model
model = LinearRegression()
model.fit(X, y)


# MySQL connection settings
db_config = {
    "unix_socket": "/tmp/mysql.sock",
    "user": "root",
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": "wastewater_db"
}


@app.route("/")
def home():
    return "Waste Water Management System API is running"

@app.route("/predictions", methods=["GET"])
def get_predictions():

    connection = mysql.connector.connect(**db_config)

    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            id,
            ph,
            turbidity,
            temperature,
            bod,
            cod,
            tss,
            dissolved_oxygen,
            treatment_efficiency,
            prediction_time
        FROM predictions
        ORDER BY id DESC
    """

    cursor.execute(query)

    predictions = cursor.fetchall()

    cursor.close()
    connection.close()

    for prediction in predictions:
        prediction["prediction_time"] = prediction["prediction_time"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return jsonify(predictions)

@app.route("/dashboard", methods=["GET"])
def dashboard():

    connection = mysql.connector.connect(**db_config)

    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            COUNT(*) AS total_predictions,
            AVG(treatment_efficiency) AS average_efficiency,
            MAX(treatment_efficiency) AS highest_efficiency,
            MIN(treatment_efficiency) AS lowest_efficiency
        FROM predictions
    """

    cursor.execute(query)

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    return jsonify({
        "total_predictions": result["total_predictions"],
        "average_efficiency": round(float(result["average_efficiency"]), 2)
            if result["average_efficiency"] is not None else 0,
        "highest_efficiency": round(float(result["highest_efficiency"]), 2)
            if result["highest_efficiency"] is not None else 0,
        "lowest_efficiency": round(float(result["lowest_efficiency"]), 2)
            if result["lowest_efficiency"] is not None else 0
    })

@app.route("/predict", methods=["POST"])
def predict():

    values = request.json

    # Create input data
    new_data = pd.DataFrame(
        [[
            values["pH"],
            values["Turbidity"],
            values["Temperature"],
            values["BOD"],
            values["COD"],
            values["TSS"],
            values["Dissolved_Oxygen"]
        ]],
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


    # Make prediction
    prediction = model.predict(new_data)

    treatment_efficiency = round(float(prediction[0]), 2)


    # Connect to MySQL
    connection = mysql.connector.connect(**db_config)

    cursor = connection.cursor()


    # Save prediction
    query = """
        INSERT INTO predictions
        (
            ph,
            turbidity,
            temperature,
            bod,
            cod,
            tss,
            dissolved_oxygen,
            treatment_efficiency
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """


    values_to_insert = (
        values["pH"],
        values["Turbidity"],
        values["Temperature"],
        values["BOD"],
        values["COD"],
        values["TSS"],
        values["Dissolved_Oxygen"],
        treatment_efficiency
    )


    cursor.execute(query, values_to_insert)

    connection.commit()


    cursor.close()
    connection.close()


    # Send result back to website
    return jsonify({
        "predicted_treatment_efficiency": treatment_efficiency
    })


if __name__ == "__main__":
    app.run(debug=True)