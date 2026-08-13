import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solar Panel Predictive Maintenance",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffffff;
        border: 1px solid #dddddd;
        margin-top: 15px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 1px solid #81c784;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff8e1;
        border: 1px solid #ffcc80;
    }

    .danger-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffebee;
        border: 1px solid #ef9a9a;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rf_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

KMEANS_PATH = os.path.join(
    MODEL_DIR,
    "kmeans_model.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "threshold.pkl"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "image_class_names.pkl"
)

CNN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "solar_cnn_model.pth"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # SENSOR MODEL
    # --------------------------------------------------------

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    kmeans = joblib.load(
        KMEANS_PATH
    )

    anomaly_threshold = joblib.load(
        THRESHOLD_PATH
    )


    # --------------------------------------------------------
    # IMAGE CLASS NAMES
    # --------------------------------------------------------

    class_names = joblib.load(
        CLASS_NAMES_PATH
    )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # --------------------------------------------------------
    # RESNET-18
    # --------------------------------------------------------

    cnn_model = models.resnet18(
        weights=None
    )

    num_features = cnn_model.fc.in_features

    cnn_model.fc = nn.Linear(
        num_features,
        len(class_names)
    )


    # --------------------------------------------------------
    # LOAD TRAINED IMAGE MODEL
    # --------------------------------------------------------

    cnn_model.load_state_dict(
        torch.load(
            CNN_MODEL_PATH,
            map_location=device
        )
    )

    cnn_model = cnn_model.to(device)

    cnn_model.eval()


    # --------------------------------------------------------
    # IMAGE TRANSFORM
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


    return (
        rf_model,
        scaler,
        kmeans,
        anomaly_threshold,
        class_names,
        cnn_model,
        transform,
        device
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    RF_MODEL_PATH,
    SCALER_PATH,
    KMEANS_PATH,
    THRESHOLD_PATH,
    CLASS_NAMES_PATH,
    CNN_MODEL_PATH
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "Some required model files are missing."
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Run `python train_models.py` first to create the models."
    )

    st.stop()


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    (
        rf_model,
        scaler,
        kmeans,
        anomaly_threshold,
        class_names,
        cnn_model,
        transform,
        device
    ) = load_models()

except Exception as e:

    st.error(
        "Error while loading the trained models."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Panel Predictive Maintenance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based monitoring of solar panel condition and machine health'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a module",
    [
        "🏠 Dashboard",
        "📊 Sensor Prediction",
        "🖼️ Image Classification",
        "🔍 Anomaly Detection",
        "ℹ️ About Project"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("System Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sensor Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Image Model",
            "ResNet-18"
        )

    with col3:
        st.metric(
            "Image Classes",
            len(class_names)
        )

    st.markdown("---")

    st.subheader("Solar Panel Image Classes")

    cols = st.columns(3)

    for i, class_name in enumerate(class_names):

        with cols[i % 3]:
            st.info(
                f"**{i + 1}. {class_name}**"
            )

    st.markdown("---")

    st.subheader("System Components")

    st.write(
        """
        **Sensor-based predictive maintenance**
        
        Uses the AI4I 2020 predictive maintenance dataset and
        a Random Forest model to predict machine failure.

        **Anomaly detection**
        
        K-Means clustering identifies sensor conditions that
        differ significantly from normal operating patterns.

        **Solar panel image classification**
        
        A ResNet-18 deep-learning model classifies uploaded solar
        panel images into six condition categories.
        """
    )


# ============================================================
# SENSOR PREDICTION
# ============================================================

elif page == "📊 Sensor Prediction":

    st.header("Sensor-Based Machine Failure Prediction")

    st.write(
        "Enter the machine sensor readings below."
    )

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            [
                "L",
                "M",
                "H"
            ]
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=298.1,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=308.6,
            step=0.1
        )

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0.0,
            max_value=5000.0,
            value=1500.0,
            step=10.0
        )

    with col2:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=1.0
        )

        twf = st.selectbox(
            "TWF - Tool Wear Failure",
            [0, 1]
        )

        hdf = st.selectbox(
            "HDF - Heat Dissipation Failure",
            [0, 1]
        )

        pwf = st.selectbox(
            "PWF - Power Failure",
            [0, 1]
        )

        osf = st.selectbox(
            "OSF - Overstrain Failure",
            [0, 1]
        )

        rnf = st.selectbox(
            "RNF - Random Failure",
            [0, 1]
        )


    if st.button(
        "🔮 Predict Machine Condition",
        type="primary"
    ):

        try:

            # Encode machine type exactly like LabelEncoder
            type_mapping = {
                "H": 0,
                "L": 1,
                "M": 2
            }

            type_encoded = type_mapping[machine_type]


            # Feature engineering
            temp_difference = (
                process_temperature
                - air_temperature
            )

            power_index = (
                torque
                * rotational_speed
                / 1000
            )


            # IMPORTANT:
            # This column order matches the training dataset.

            input_data = pd.DataFrame(
                [[
                    type_encoded,
                    air_temperature,
                    process_temperature,
                    rotational_speed,
                    torque,
                    tool_wear,
                    twf,
                    hdf,
                    pwf,
                    osf,
                    rnf,
                    temp_difference,
                    power_index
                ]],
                columns=[
                    "Type",
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "TWF",
                    "HDF",
                    "PWF",
                    "OSF",
                    "RNF",
                    "Temp_Difference",
                    "Power_Index"
                ]
            )


            # Scale
            input_scaled = scaler.transform(
                input_data
            )


            # Prediction
            prediction = rf_model.predict(
                input_scaled
            )[0]


            # Probability
            if hasattr(
                rf_model,
                "predict_proba"
            ):

                probability = rf_model.predict_proba(
                    input_scaled
                )[0]

                failure_probability = (
                    probability[1]
                    * 100
                )

            else:

                failure_probability = None


            st.markdown("---")

            st.subheader("Prediction Result")


            if prediction == 1:

                st.markdown(
                    """
                    <div class="danger-box">
                    <h3>⚠️ Machine Failure Predicted</h3>
                    The sensor values indicate a possible
                    machine failure condition.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="success-box">
                    <h3>✅ Normal Condition</h3>
                    No machine failure is predicted from
                    the supplied sensor values.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            if failure_probability is not None:

                st.metric(
                    "Failure Probability",
                    f"{failure_probability:.2f}%"
                )


            # K-Means anomaly detection

            cluster_distance = np.min(
                kmeans.transform(
                    input_scaled
                ),
                axis=1
            )[0]


            st.subheader(
                "Anomaly Detection"
            )

            st.write(
                f"Distance from nearest cluster: "
                f"**{cluster_distance:.4f}**"
            )

            st.write(
                f"Anomaly threshold: "
                f"**{anomaly_threshold:.4f}**"
            )


            if (
                cluster_distance
                > anomaly_threshold
            ):

                st.warning(
                    "⚠️ The sensor pattern appears anomalous."
                )

            else:

                st.success(
                    "✅ The sensor pattern is within the normal range."
                )


        except Exception as e:

            st.error(
                "Prediction error:"
            )

            st.code(
                str(e)
            )


# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

elif page == "🖼️ Image Classification":

    st.header("Solar Panel Image Classification")

    st.write(
        """
        Upload an image of a solar panel. The ResNet-18 model
        will classify its condition.
        """
    )


    uploaded_file = st.file_uploader(
        "Upload Solar Panel Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # Display uploaded image
            st.image(
                image,
                caption="Uploaded Solar Panel Image",
                use_container_width=True
            )


            # Transform image
            image_tensor = transform(
                image
            )


            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(
                0
            ).to(device)


            # Prediction
            with torch.no_grad():

                outputs = cnn_model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted_index = torch.max(
                    probabilities,
                    1
                )


            predicted_index = predicted_index.item()

            confidence = confidence.item()


            predicted_class = class_names[
                predicted_index
            ]


            st.markdown("---")

            st.subheader(
                "Classification Result"
            )


            st.success(
                f"Predicted Condition: **{predicted_class}**"
            )


            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


            # ------------------------------------------------
            # ALL CLASS PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )


            probabilities_np = (
                probabilities[0]
                .cpu()
                .numpy()
            )


            probability_df = pd.DataFrame({
                "Condition": class_names,
                "Probability": (
                    probabilities_np * 100
                )
            })


            probability_df = (
                probability_df
                .sort_values(
                    "Probability",
                    ascending=False
                )
            )


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            # Bar chart
            st.bar_chart(
                probability_df.set_index(
                    "Condition"
                )
            )


            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            if predicted_class == "Clean":

                st.success(
                    "✅ The panel appears clean."
                )

            elif predicted_class == "Dusty":

                st.warning(
                    "⚠️ Dust detected. Cleaning or maintenance "
                    "may be required."
                )

            elif predicted_class == "Bird-drop":

                st.warning(
                    "⚠️ Bird-drop contamination detected. "
                    "Cleaning is recommended."
                )

            elif predicted_class == "Snow-Covered":

                st.warning(
                    "⚠️ Snow coverage detected. "
                    "Panel performance may be reduced."
                )

            elif predicted_class == "Physical-Damage":

                st.error(
                    "🚨 Physical damage detected. "
                    "Inspect the panel for cracks or structural damage."
                )

            elif predicted_class == "Electrical-damage":

                st.error(
                    "🚨 Electrical damage detected. "
                    "Professional inspection is recommended."
                )


        except Exception as e:

            st.error(
                "Image prediction error:"
            )

            st.code(
                str(e)
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🔍 Anomaly Detection":

    st.header("K-Means Anomaly Detection")

    st.write(
        """
        This module uses the trained K-Means model to determine
        whether sensor patterns are close to the learned normal
        operating clusters.
        """
    )


    st.metric(
        "Anomaly Threshold",
        f"{anomaly_threshold:.4f}"
    )


    st.write(
        """
        A distance greater than the threshold indicates that
        the current operating condition is potentially anomalous.
        """
    )


    st.info(
        "You can also view anomaly detection through the "
        "Sensor Prediction module after entering sensor values."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.header(
        "About the Project"
    )


    st.write(
        """
        ### Solar Panel Predictive Maintenance Using AI and Machine Learning

        This project combines machine learning and deep learning
        techniques for predictive maintenance and solar-panel
        condition monitoring.

        ### Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - PyTorch
        - Torchvision
        - Streamlit

        ### Machine Learning Models

        **Random Forest**
        
        Used for sensor-based machine failure prediction.

        **K-Means**
        
        Used for anomaly detection.

        **ResNet-18**
        
        Used to classify solar panel images into six categories:

        - Bird-drop
        - Clean
        - Dusty
        - Electrical-damage
        - Physical-Damage
        - Snow-Covered

        ### Datasets

        The sensor component uses the AI4I 2020 predictive
        maintenance dataset.

        The image component uses the solar-panel image dataset
        stored in:

        `data/solar_panel_images`
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Solar Panel Predictive Maintenance | "
    "Machine Learning + Deep Learning"
)