import warnings
warnings.filterwarnings("ignore")

import os
import sys
from Code.ResidualAttentionNetwork import ResidualAttentionNetwork
import numpy as np
import pandas as pd 
from datetime import datetime
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers 

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# ============================ DATA WORK ============================
file_num = sys.argv[1]

training_filepath = "/home/deopha32/EyeDiseaseClassification/Data/training_data_{}".format(file_num)
validation_filepath = "/home/deopha32/EyeDiseaseClassification/Data/validatation_data_{}".format(file_num)

model_train_data = pd.read_csv(training_filepath)
model_val_data = pd.read_csv(validation_filepath)

# ============================ MODEL META ============================

IMAGE_WIDTH=32
IMAGE_HEIGHT=32
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=1
IMAGE_SHAPE=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

batch_size=100

num_classes = 4

epochs = 500

# Image Generators

train_datagen = ImageDataGenerator(rescale=1./255., 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=model_train_data,
    x_col="filepath",
    y_col="class",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    validate_filenames=False
)

valid_datagen = ImageDataGenerator(rescale=1./255.)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=model_val_data,
    x_col="filepath",
    y_col="class",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    validate_filenames=False
)

# Class Weights
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

weights = class_weight.compute_class_weight('balanced',
                                             class_names,
                                             model_train_data['class'])

# Callbacks
curr_time = f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
logger_path = "/pylon5/cc5614p/deopha32/Saved_Models/eye-model-history_cv{num}_{time}.csv" \
                .format(num=file_num, time=curr_time)
model_path = "/pylon5/cc5614p/deopha32/Saved_Models/eye-model_cv{num}_{time}.h5" \
                .format(num=file_num, time=curr_time)

csv_logger = CSVLogger(logger_path, append=True)
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True)

callbacks = [csv_logger, checkpoint]

# Generator Step Size
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# ============================ MODEL TRAINING ============================

with tf.device('/gpu:0'):
    model = ResidualAttentionNetwork(
                input_shape=IMAGE_SHAPE, 
                n_classes=num_classes, 
                activation='softmax').build_model()
    
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN, verbose=1, callbacks=callbacks,
                    validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                    epochs=epochs, use_multiprocessing=True, workers=40)
