from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.title("Employee Data Prediction")

# 1. Load Data
csv_path = Path(__file__).resolve().parent / "HR-Employee-Attrition.csv"
if not csv_path.exists():
    st.error("HR-Employee-Attrition.csv file not found!")
    st.stop()

df = pd.read_csv(csv_path)

# Map target variable
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

st.subheader("DATASET PREVIEW")
st.dataframe(df.head(10))

# 2. Feature Selection
x = df.iloc[0:100][
    [
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StandardHours",
    ]
]
y = df.iloc[0:100]["OverTime"]

# 3. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 4. Model Training
dtc = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
)
dtc.fit(x_train, y_train)

# 5. Prediction & Metrics
y_prediction = dtc.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)

st.subheader("MODEL ACCURACY")
st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")

# 6. Decision Tree Visualization
st.subheader("DECISION TREE VISUALIZATION")
fig, ax = plt.subplots(figsize=(15, 8))
plot_tree(
    dtc,
    feature_names=x.columns.tolist(),
    class_names=["No OverTime", "OverTime"],
    filled=True,
    fontsize=10,
    ax=ax,
)
st.pyplot(fig)

# 7. Model Evaluation
st.subheader("CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_prediction)
st.dataframe(
    pd.DataFrame(
        cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]
    )
)

st.subheader("CLASSIFICATION REPORT")
report = classification_report(
    y_test, y_prediction, output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T)

# 8. Prediction Results Table
st.subheader("PREDICTION RESULT")
results = x_test.copy()
results["Actual_Results"] = y_test.values
results["Predicted_Results"] = y_prediction
st.dataframe(results.head(10))