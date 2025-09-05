import sys
import os
import logging
import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import kagglehub
import zipfile


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download dataset
logging.info("Downloading PlantVillage dataset using kagglehub...")
try:
    BASE_PATH = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    logging.info(f"Dataset downloaded to: {BASE_PATH}")
except Exception as e:
    logging.error(f"Failed to download dataset: {e}")
    sys.exit(1)


# Locate the "color" folder
def find_color_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        if "color" in dirs:
            return os.path.join(root, "color")
    return None

DATASET_PATH = find_color_folder(BASE_PATH)

if DATASET_PATH is None:
    logging.error("Could not find 'color' folder inside dataset.")
    sys.exit(1)

logging.info(f"Using dataset directory: {DATASET_PATH}")


# Feature extraction
def extract_features(image):
    try:
        img = cv2.resize(image, (64, 64))  # resize
        hist = cv2.calcHist([img], [0, 1, 2], None,
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    except Exception as e:
        logging.warning(f"Error extracting features: {e}")
        return None


# Load dataset
logging.info(f"Loading dataset from {DATASET_PATH}...")
X, y = [], []
for label_dir in Path(DATASET_PATH).iterdir():
    if label_dir.is_dir():
        label = label_dir.name
        logging.info(f"Processing directory: {label}")
        for img_path in label_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                continue
            features = extract_features(img)
            if features is not None:
                X.append(features)
                y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

if len(X) == 0 or len(y) == 0:
    logging.error("No valid data extracted. Check dataset structure.")
    sys.exit(1)

logging.info(f"Extracted features for {len(X)} images with {X.shape[1]} features each.")


# Encode + scale
le = LabelEncoder()
y = le.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save encoder and scaler
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

logging.info(f"Dataset sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")


# Model training
best_k, best_acc = 5, 0
for k in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    logging.info(f"K={k}, Validation Accuracy={acc:.2f}")
    if acc > best_acc:
        best_acc, best_k = acc, k

logging.info(f"Best K: {best_k} with Validation Accuracy: {best_acc:.2f}")

# Retrain with best K
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn.fit(X_train, y_train)

# Save final model
joblib.dump(knn, "knn_model.pkl")
logging.info("Final model saved as knn_model.pkl")


# Evaluation
y_pred = knn.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
logging.info(f"Test Accuracy: {test_acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

