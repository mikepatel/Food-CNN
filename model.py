"""
Michael Patel
March 2021

Project description:
    Use TensorFlow to create an object detection model that runs on an Android device

File description:
    For model definitions

"""
################################################################################
# Imports
from packages import *


################################################################################
# MobileNetV2
def build_model_mobilenet(num_classes):
    # use Functional API of Model()

    # define model layers
    data_augment_layers = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomRotation(
            factor=0.2
        )
        #tf.keras.layers.experimental.preprocessing.RandomTranslation(
        #    height_factor=0.2,
        #    width_factor=0.2
        #)
    ])

    preprocess_input_layer = tf.keras.applications.mobilenet_v2.preprocess_input
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )
    mobilenet.trainable = False

    # add classification layers
    global_pool_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout_layer = tf.keras.layers.Dropout(rate=0.2)
    fc_layer = tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu
    )
    output_layer = tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    )

    # build model
    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x = inputs
    x = data_augment_layers(x)
    x = preprocess_input_layer(x)
    x = mobilenet(x, training=False)
    x = global_pool_layer(x)
    x = dropout_layer(x)
    x = fc_layer(x)
    x = dropout_layer(x)
    x = output_layer(x)
    outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model

    """
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
    """


# VGG16
def build_cnn_vgg16(num_classes):
    """
    vgg16 = tf.keras.applications.VGG16(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False
    )

    vgg16.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg16)
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Output
    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))

    return model
    """

    inputs = tf.keras.layers.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    )

    x = inputs
    x = tf.keras.applications.vgg16.preprocess_input(x)
    vgg16 = tf.keras.applications.VGG16(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )

    vgg16.trainable = False
    x = vgg16(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    )(x)

    outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

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
