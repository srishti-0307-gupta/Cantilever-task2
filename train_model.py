import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("CreditCardData.csv")

# Drop rows with missing target
df = df.dropna(subset=["Fraud"])

# Select relevant features
features = [
    "Time", "Type of Card", "Entry Mode", "Amount", "Type of Transaction",
    "Merchant Group", "Country of Transaction", "Shipping Address",
    "Country of Residence", "Gender", "Age", "Bank"
]
target = "Fraud"

# Filter relevant columns
df = df[features + [target]]

# Fill missing values
df.fillna("Unknown", inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df[features]
y = df[target].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoders
with open("fraud_model.pkl", "wb") as f:
    pickle.dump((label_encoders, model), f)
