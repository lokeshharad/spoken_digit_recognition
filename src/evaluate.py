import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.model import build_model
from src.config import MODEL_DIR

def main():
    # Load test data
    _, X_test, _, y_test = load_data()
    y_true = np.argmax(y_test, axis=1)

    # Load trained model
    model = tf.keras.models.load_model(f"{MODEL_DIR}/best_model.h5")

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
