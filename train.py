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
from model import build_model_mobilenet, build_cnn_vgg16, build_cnn_custom, build_model_mobilenet_2


################################################################################
# Main
if __name__ == "__main__":
    # TF version
    print(f'TensorFlow version: {tf.__version__}')

    # ----- ETL ----- #
    # labels
    labels = []
    int2label = {}
    directories = os.listdir(TRAIN_DIR)
    for i in range(len(directories)):
        name = directories[i]
        labels.append(name)
        int2label[i] = name

    num_classes = len(labels)
    #print(labels)
    #print(num_classes)

    # create text file with labels
    with open("labels.txt", "w") as f:
        for d in directories:
            f.write(d + "\n")

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
        validation_split=0.05
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
    model = build_model_mobilenet(num_classes=num_classes)
    #model = build_model_mobilenet_2(num_classes=num_classes)
    #model = build_cnn_vgg16(num_classes=num_classes)
    #model = build_cnn_custom(num_classes=num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(os.getcwd(), "training"))

    # save model
    model.save(SAVE_DIR)

    # ----- DEPLOY ----- #
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
