"""
Michael Patel
March 2021

Project description:
    Use TensorFlow to create an object detection model that runs on an Android device

File description:
    For model training

"""
################################################################################
# Imports
from packages import  *


################################################################################
# Main
if __name__ == "__main__":
    # TF version
    print(f'TensorFlow version: {tf.__version__}')

    # create a save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # ----- ETL ----- #
    # get labels / class names
    directories = os.listdir(TRAIN_DIR)
    num_classes = len(directories)
    print(f'Number of classes: {num_classes}')

    # create text file with labels
    if not os.path.exists(os.path.join(os.getcwd(), "labels.txt")):
        with open("labels.txt", "w") as f:
            for d in directories:
                f.write(d + "\n")

    # image generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        subset="training"
    )

    validation_generator = datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        subset="validation"
    )

    # ----- MODEL ----- #
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )
    mobilenet.trainable = False

    """
    model = tf.keras.Sequential([
        mobilenet,
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=num_classes, activation="softmax")
    ])
    """
    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x = inputs
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = mobilenet(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    outputs = x
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

    # training plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(os.path.join(os.getcwd(), "plots"))

    # save model
    model.save(SAVE_DIR)

    # ----- DEPLOY ----- #
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    tflite_model = converter.convert()

    with open(os.path.join(SAVE_DIR, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
