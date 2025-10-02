# import os
# import mlflow
# import mlflow.tensorflow
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from src.config import MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE
# from src.data_loader import load_data
# # from src.model import build_model
# from src.model import build_simple_model as build_model

# def main():
#     # Load data
#     X_train, X_test, y_train, y_test = load_data()

#     # Build model
#     model = build_model()

#     # Callbacks
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")
#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#         ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
#     ]

#     # Start MLflow run
#     mlflow.set_experiment("spoken_digit_recognition")
#     with mlflow.start_run():
#         # Log parameters
#         mlflow.log_param("epochs", EPOCHS)
#         mlflow.log_param("batch_size", BATCH_SIZE)
#         mlflow.log_param("learning_rate", LEARNING_RATE)

#         # Train model
#         history = model.fit(
#             X_train, y_train,
#             validation_data=(X_test, y_test),
#             epochs=EPOCHS,
#             batch_size=BATCH_SIZE,
#             callbacks=callbacks
#         )

#         # Log metrics
#         for i, val_acc in enumerate(history.history['val_accuracy']):
#             mlflow.log_metric("val_accuracy_epoch", val_acc, step=i)
        
#         # Log model
#         mlflow.tensorflow.log_model(model, "model")

#     print(f"✅ Training complete. Best model saved at {checkpoint_path}")

# if __name__ == "__main__":
#     main()

import os
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.data_loader import load_data
from src.model import build_simple_model as build_model

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Build model
    model = build_model()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    # Set MLflow experiment
    mlflow.set_experiment("spoken_digit_recognition")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )

        # Log metrics per epoch
        for i in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][i], step=i)
            mlflow.log_metric("train_loss", history.history['loss'][i], step=i)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][i], step=i)
            mlflow.log_metric("val_loss", history.history['val_loss'][i], step=i)

        # Log the trained model
        mlflow.tensorflow.log_model(model, artifact_path="model")

    print(f"✅ Training complete. Best model saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
