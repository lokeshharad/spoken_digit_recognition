import numpy as np
import mlflow
import mlflow.tensorflow

# MLflow model path (replace RUN_ID with your actual run ID or use latest)
MLFLOW_MODEL_URI = "runs:/<RUN_ID>/model"  # e.g., "runs:/1234567890abcdef/model"

def load_model():
    model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)
    return model

def predict(model, sample):
    # Ensure shape: (1, 64, 64, 1)
    sample = np.expand_dims(sample, axis=0)
    pred = model.predict(sample)
    return np.argmax(pred)

if __name__ == "__main__":
    # Example: random dummy input
    dummy = np.random.rand(64, 64, 1).astype(np.float32)

    model = load_model()
    digit = predict(model, dummy)
    print(f"Predicted digit: {digit}")