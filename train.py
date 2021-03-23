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
from packages import *
from model import build_model_mobilenet, build_cnn_vgg16, build_cnn_custom


################################################################################
# Main
if __name__ == "__main__":
    # TF version
    print(f'TensorFlow version: {tf.__version__}')

    # create a save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # ----- ETL ----- #
    # create text file with labels
    directories = os.listdir(TRAIN_DIR)
    with open("labels.txt", "w") as f:
        for d in directories:
            f.write(d + "\n")

    num_classes = len(directories)

    # image data generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        #rotation_range=90,
        #horizontal_flip=True,
        #vertical_flip=True,
        #width_shift_range=0.3,
        #height_shift_range=0.3,
        #brightness_range=[0.1, 1.3],
        #zoom_range=0.5,
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

    #print(len(list(pathlib.Path(TRAIN_DIR).glob("*/*.jpg"))))

    """
    # create dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        seed=DATASET_SEED
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        seed=DATASET_SEED
    )

    #print(train_dataset.class_names)

    # configure dataset for performance
    # cache() --> load images into memory
    # prefetch() --> overlap data preprocessing and model execution while training
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    """

    # ----- MODEL ----- #
    #model = build_model_mobilenet(num_classes=num_classes)
    #model = build_model_mobilenet_2(num_classes=num_classes)
    #model = build_cnn_vgg16(num_classes=num_classes)
    #model = build_cnn_custom(num_classes=num_classes)

    # get MobileNetV2 base
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )

    mobilenet.trainable = False

    # add classification head
    model = tf.keras.Sequential([
        mobilenet,
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            units=num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # label_mode = "int"
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
    """

    history = model.fit(
        x=train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=validation_dataset
    )
    """

    # save model
    model.save(SAVE_DIR)

    # fine-tune model
    # unfreeze base
    mobilenet.trainable = True
    fine_tune_at = 100
    for layer in mobilenet.layers[:fine_tune_at]:
        layer.trainable = False  # freeze early layers

    # re-compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=["accuracy"]
    )

    model.summary()

    # continue training
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

    # plot accuracy
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, "accuracy"))

    # plot loss
    plt.clf()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, "loss"))

    # save model
    model.save(SAVE_DIR)

    # ----- DEPLOY ----- #
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    tflite_model = converter.convert()

    with open(os.path.join(SAVE_DIR, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
