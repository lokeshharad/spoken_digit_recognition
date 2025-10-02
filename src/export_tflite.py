import tensorflow as tf
from src.model import build_model
from src.config import MODEL_DIR

def main():
    # Load Keras model
    model = tf.keras.models.load_model(f"{MODEL_DIR}/best_model.h5")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = f"{MODEL_DIR}/digit_cnn.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ TFLite model saved at {tflite_path}")

if __name__ == "__main__":
    main()
