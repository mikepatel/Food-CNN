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
import pathlib


################################################################################
# directories
DATA_DIR = os.path.join(os.getcwd(), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

NUM_EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
DATASET_SEED = 123

LEARNING_RATE = 0.001  # default 0.001
