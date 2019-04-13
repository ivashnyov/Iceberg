from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.callbacks import Callback
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score


# Manually created f1 score callback
class F1Score(Callback):

    def on_train_begin(self, logs={}):
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        predict = np.asarray(self.model.predict(self.model.validation_data[0]))
        label = self.model.validation_data[1]
        f1 = f1_score(label, predict)
        self.f1s.append(f1)
        print ('f1:', f1)
        return


def create_model(input_shape, optimizer, loss_function):
    """
    Creation model for image classification
    :param input_shape: the shape of input images
    :param optimizer: suitable optimiser
    :param loss_function: suitable loss function
    :return: model
    """

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])
    return model


def images_augmentation(data, size):
    """
    Creating image dataflow for CNN
    :param data: name of folder with folders 0 and 1
    :param size: shape of images
    :return: data generators for train and val
    """
    parent_dir = os.getcwd()
    data_path = os.path.join(parent_dir, data)

    # Creating image augmentation generator
    # We will allow horizontal flip, little rotation and shifts
    train_image_gen = image.ImageDataGenerator(horizontal_flip=True,
                                               rotation_range=15,
                                               validation_split=0.2)

    # Creating image data generator for test
    train_image_data = train_image_gen.flow_from_directory(data_path,
                                                           target_size=size,
                                                           batch_size=32,
                                                           class_mode='binary',
                                                           subset='training')
    # Creating image data generator for val
    validation_image_data = train_image_gen.flow_from_directory(data,
                                                                target_size=size,
                                                                batch_size=32,
                                                                class_mode='binary',
                                                                subset='validation')

    return train_image_data, validation_image_data


def train_model(model, train, val, num_epochs, callbacks=None):
    """
    Training model
    :param model: model to train
    :param train: train data generator
    :param val: val data generator
    :param num_epochs: number of epochs to train
    :param callbacks: suitable callback to evaluate model
    :return: history of training
    """

    history = model.fit_generator(train,
                                  samples_per_epoch=2000,
                                  validation_data=val,
                                  epochs=num_epochs,
                                  callbacks=callbacks)

    return history
