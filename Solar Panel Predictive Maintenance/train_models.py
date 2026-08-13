import os
import joblib
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 1. SENSOR MODULE
# ============================================================

print("\n" + "=" * 60)
print("1. TRAINING SENSOR MODELS")
print("=" * 60)

try:
    df = pd.read_csv("data/ai4i2020.csv")
except FileNotFoundError:
    print("ERROR: data/ai4i2020.csv not found.")
    exit(1)

df_clean = df.drop(["UDI", "Product ID"], axis=1)

le = LabelEncoder()
df_clean["Type"] = le.fit_transform(df_clean["Type"])

# Feature engineering
df_clean["Temp_Difference"] = (
    df_clean["Process temperature [K]"]
    - df_clean["Air temperature [K]"]
)

df_clean["Power_Index"] = (
    df_clean["Torque [Nm]"]
    * df_clean["Rotational speed [rpm]"]
    / 1000
)

X = df_clean.drop(["Machine failure"], axis=1)
y = df_clean["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_dict = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000, random_state=42),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=5, random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

best_model = None
best_acc = 0

for name, model in models_dict.items():

    model.fit(X_train_scaled, y_train)

    acc = model.score(
        X_test_scaled,
        y_test
    )

    print(f"{name} accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model

joblib.dump(
    best_model,
    "models/rf_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("✅ Sensor models saved.")
print(f"Best sensor accuracy: {best_acc:.4f}")


# ============================================================
# 2. K-MEANS ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 60)
print("2. TRAINING K-MEANS ANOMALY DETECTOR")
print("=" * 60)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)

distances = np.min(
    kmeans.transform(X_train_scaled),
    axis=1
)

threshold = np.percentile(
    distances,
    95
)

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    threshold,
    "models/threshold.pkl"
)

print("✅ K-Means model saved.")
print(f"Anomaly threshold: {threshold:.4f}")


# ============================================================
# 3. SOLAR PANEL IMAGE CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("3. TRAINING SOLAR PANEL IMAGE CLASSIFIER")
print("=" * 60)

data_dir = "data/solar_panel_images"

if not os.path.exists(data_dir):

    print("⚠️ Image dataset not found:")
    print(data_dir)

    joblib.dump(
        ["dummy_class"],
        "models/image_class_names.pkl"
    )

else:

    print("✅ Image dataset found.")
    print("Dataset path:", data_dir)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    try:

        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        full_dataset = datasets.ImageFolder(
            data_dir,
            transform=image_transform
        )

        class_names = full_dataset.classes

        print("\nDetected classes:")

        for i, name in enumerate(class_names):
            print(f"{i}: {name}")

        print(
            f"\nTotal images: {len(full_dataset)}"
        )

        # ----------------------------------------------------
        # SPLIT DATA
        # ----------------------------------------------------

        train_size = int(
            0.8 * len(full_dataset)
        )

        val_size = (
            len(full_dataset)
            - train_size
        )

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(
            f"Training images: {len(train_dataset)}"
        )

        print(
            f"Validation images: {len(val_dataset)}"
        )

        # ----------------------------------------------------
        # DATA LOADERS
        # ----------------------------------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )

        # Save classes
        joblib.dump(
            class_names,
            "models/image_class_names.pkl"
        )

        # ----------------------------------------------------
        # RESNET-18
        # ----------------------------------------------------

        print("\nLoading ResNet-18...")

        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze pretrained layers
        for param in model.parameters():
            param.requires_grad = False

        # Replace final layer
        num_features = model.fc.in_features

        model.fc = nn.Linear(
            num_features,
            len(class_names)
        )

        model = model.to(device)

        # ----------------------------------------------------
        # LOSS + OPTIMIZER
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.fc.parameters(),
            lr=0.001,
            momentum=0.9
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        num_epochs = 5

        print(
            f"\nStarting training for {num_epochs} epochs..."
        )

        best_val_accuracy = 0.0

        for epoch in range(num_epochs):

            print(
                f"\nEpoch {epoch + 1}/{num_epochs}"
            )

            print("-" * 40)

            # ==========================
            # TRAIN
            # ==========================

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

            train_loss = (
                running_loss
                / len(train_dataset)
            )

            train_accuracy = (
                correct / total
            )

            # ==========================
            # VALIDATION
            # ==========================

            model.eval()

            val_correct = 0
            val_total = 0
            val_loss_total = 0.0

            with torch.no_grad():

                for inputs, labels in val_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)

                    loss = criterion(
                        outputs,
                        labels
                    )

                    val_loss_total += (
                        loss.item()
                        * inputs.size(0)
                    )

                    _, predicted = torch.max(
                        outputs,
                        1
                    )

                    val_total += labels.size(0)

                    val_correct += (
                        predicted == labels
                    ).sum().item()

            val_loss = (
                val_loss_total
                / len(val_dataset)
            )

            val_accuracy = (
                val_correct / val_total
            )

            print(
                f"Train Loss: {train_loss:.4f}"
            )

            print(
                f"Train Accuracy: {train_accuracy:.4f}"
            )

            print(
                f"Validation Loss: {val_loss:.4f}"
            )

            print(
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

            # Save best model
            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                torch.save(
                    model.state_dict(),
                    "models/solar_cnn_model.pth"
                )

                print(
                    "✅ Best image model saved."
                )

        print("\n" + "=" * 60)
        print("IMAGE MODEL TRAINING COMPLETE")
        print("=" * 60)

        print(
            "Classes:",
            class_names
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.4f}"
        )

    except Exception as e:

        print(
            "\n⚠️ Error during image training:"
        )

        print(e)

        print(
            "\nPlease check your image dataset."
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("🎉 ALL TRAINING COMPLETED")
print("=" * 60)