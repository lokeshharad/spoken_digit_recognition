import tensorflow as tf
from tensorflow.keras import layers, models

# def build_model(input_shape=(64, 64, 1), num_classes=10, dropout_rate=0.3):
#     """
#     Improved CNN model for spoken digit recognition.
#     """
#     inputs = tf.keras.Input(shape=input_shape)

#     # Conv Block 1
#     x = layers.Conv2D(32, (3,3), padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)
#     x = layers.MaxPooling2D((2,2))(x)
#     x = layers.Dropout(dropout_rate)(x)

#     # Conv Block 2
#     x = layers.Conv2D(64, (3,3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)
#     x = layers.MaxPooling2D((2,2))(x)
#     x = layers.Dropout(dropout_rate)(x)

#     # Conv Block 3
#     x = layers.Conv2D(128, (3,3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)
#     x = layers.MaxPooling2D((2,2))(x)
#     x = layers.Dropout(dropout_rate)(x)

#     # Global Average Pooling
#     x = layers.GlobalAveragePooling2D()(x)

#     # Dense layers
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(dropout_rate)(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs, outputs)
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

# if __name__ == "__main__":
#     model = build_model()
#     model.summary()

import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_model(input_shape=(64, 64, 1), num_classes=10):
    """
    Simple CNN model for spoken digit recognition
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Conv Block 1
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    # Conv Block 2
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Conv Block 3
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = build_simple_model()
    model.summary()
