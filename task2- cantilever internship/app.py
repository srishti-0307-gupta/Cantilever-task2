from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load model and encoders
with open("fraud_model.pkl", "rb") as f:
    label_encoders, model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get data from form
        form_data = {
            "Time": float(request.form["time"]),
            "Type of Card": request.form["card_type"],
            "Entry Mode": request.form["entry_mode"],
            "Amount": float(request.form["amount"]),
            "Type of Transaction": request.form["transaction_type"],
            "Merchant Group": request.form["merchant_group"],
            "Country of Transaction": request.form["country_transaction"],
            "Shipping Address": request.form["shipping_address"],
            "Country of Residence": request.form["country_residence"],
            "Gender": request.form["gender"],
            "Age": int(request.form["age"]),
            "Bank": request.form["bank"]
        }

        # Create DataFrame
        input_df = pd.DataFrame([form_data])

        # Encode categorical values
        for col in input_df.select_dtypes(include="object").columns:
            le = label_encoders.get(col)
            if le and input_df[col][0] in le.classes_:
                input_df[col] = le.transform(input_df[col])
            else:
                # Handle unseen label
                input_df[col] = le.transform(["Unknown"]) if "Unknown" in le.classes_ else 0

        # Prediction (0 = Not Fraud, 1 = Fraud)
        pred_proba = model.predict_proba(input_df)[0][1]  # Probability of fraud
        fraud_percent = round(pred_proba * 100, 2)
        prediction = int(pred_proba >= 0.5)  # threshold

        return render_template("result.html", probability=fraud_percent, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
