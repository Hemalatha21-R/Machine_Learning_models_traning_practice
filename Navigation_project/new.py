from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("SMART METER PREDICTION")

csv_path = Path(__file__).resolve().parent / "Household.csv"
if not csv_path.exists():
    st.error("Household.csv file not found!")
    st.stop()

df = pd.read_csv(csv_path)

st.subheader("DATASET PREVIEW")
st.dataframe(df.head(10))

# Feature selection
x = df[["power_watts", "voltage_v", "current_a", "duration_minutes"]]
y = df["smart_meter"]

# Splitting
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression()
fitted_model = model.fit(x_train, y_train)

# Generate Predictions (Fixes NameError)
y_pred = fitted_model.predict(x_test)

results = x_test.copy()
results["Actual_Results"] = y_test.values
results["Predicted_Results"] = y_pred

st.subheader("CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
st.dataframe(
    pd.DataFrame(
        cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]
    )
)

st.subheader("CLASSIFICATION REPORT")
# Fixes UndefinedMetricWarning
report = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T)

st.subheader("PREDICTION RESULT")
st.dataframe(results.head(10))