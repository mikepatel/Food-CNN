"""
Michael Patel
March 2021

Project description:
    Use TensorFlow to create an object detection model that runs on an Android device

File description:

"""
################################################################################
# Imports
from packages import *


################################################################################
# MobileNetV2
def build_model_mobilenet(num_classes):
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False,
        weights="imagenet"
    )

    mobilenet.trainable = False

    # add classification
    model = tf.keras.Sequential()
    model.add(mobilenet)
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu"
    ))

    model.add(tf.keras.layers.Dropout(
        rate=0.2
    ))

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation="softmax"
    ))

    return model


# MobileNetV2...v2
def build_model_mobilenet_2(num_classes):
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False,
        weights="imagenet"
    )

    mobilenet.trainable = False

    # add classification head
    model = tf.keras.Sequential()
    model.add(mobilenet)

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.Dropout(
        rate=0.4
    ))
    model.add(tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.Dropout(
        rate=0.4
    ))
    model.add(tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))

    return model


# VGG16
def build_cnn_vgg16(num_classes):
    vgg16 = tf.keras.applications.VGG16(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False
    )

    vgg16.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(
        units=256,
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))

    return model


# custom CNN
def build_cnn_custom(num_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    ))

    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=512 ,
        kernel_size=3,
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=1024
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(
        units=256
    ))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))

    return model
