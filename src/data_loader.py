import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.config import NUM_CLASSES

PROCESSED_DIR = "data/processed"

def load_data(test_size=0.2, random_state=42):
    # Load preprocessed features
    X = np.load(f"{PROCESSED_DIR}/X.npy")
    y = np.load(f"{PROCESSED_DIR}/y.npy")
    
    # One-hot encode labels
    y = to_categorical(y, num_classes=NUM_CLASSES)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
