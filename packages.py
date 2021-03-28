"""
Michael Patel
March 2021

Project description:
    Use TensorFlow to create an object detection model that runs on an Android device

File description:
    For imports and model/training hyperparameters

"""
################################################################################
# Imports
import os
import matplotlib.pyplot as plt
import tensorflow as tf


################################################################################
# directories
DATA_DIR = os.path.join(os.getcwd(), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
DATASET_SEED = 123

LEARNING_RATE = 1e-2  # default 0.001 for Adam, 0.01 for SGD
NUM_EPOCHS = 10

# fine-tuning
LEARNING_RATE_FINE_TUNING = 1e-5
NUM_EPOCHS_FINE_TUNING = 10
