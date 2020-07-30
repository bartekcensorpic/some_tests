# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf

#https://stackoverflow.com/questions/47279677/how-use-grid-search-with-fit-generator-in-keras


def get_gens(test_df,train_df):


    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        # samplewise_std_normalization = True,
    )
    batch_size = 16
    image_size = (512,512)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        # directory=images_folder_path,
        x_col="image_path",
        y_col="tags",
        class_mode="categorical",
        batch_size=batch_size,
        target_size=image_size,
        shuffle=False,  # THIS IS A MUST
        color_mode='rgb',
        seed=42,
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        # directory=images_folder_path,
        x_col="image_path",
        y_col="tags",
        class_mode="categorical",
        batch_size=batch_size,
        color_mode='rgb',
        target_size=image_size,
        shuffle=False,  # THIS IS A MUST
        seed=42
    )

    return train_generator, test_generator

def fbeta(y_true, y_pred, beta=2):
    """
    Fbeta metric for Keras
    :param y_true:
    :param y_pred:
    :param beta:
    :return:
    """
    backend = tf.keras.backend
    # clip predictions
    y_pred = tf.keras.backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score

def create_model(path_to_trained_model):

    if path_to_trained_model == '':
        raise ValueError()

    base_model = tf.keras.models.load_model(
        path_to_trained_model,
        custom_objects={"KerasLayer": hub.KerasLayer, "fbeta": fbeta},
    )
    base_model.trainable = True

    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=[fbeta, "acc"],
    )

    return base_model