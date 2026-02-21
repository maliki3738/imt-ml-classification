from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def build_model(img_size=(224, 224), num_classes=3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    inputs = keras.Input(shape=img_size + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = data_augmentation(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)