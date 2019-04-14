from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Lambda
from keras import optimizers
import os
import matplotlib.pyplot as plt


def create_model(input_shape=(128, 128, 3), learning_rate=0.001):
    """
    Creation model for image classification
    :param input_shape: the shape of input images
    :param learning_rate: suitable learning_rate for optimizer
    :return: model
    """

    # Define Sequential model with 3 conv layers and 3 pool layers
    model = Sequential()
    model.add(Lambda(lambda size: size / 255.0, input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    adam = optimizers.adam(lr=learning_rate)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_generators(data, size=(128, 128)):
    """
    Creating image dataflow for CNN
    :param data: name of folder with folders 0 and 1
    :param size: shape of images
    :return: data generators for train, val and test
    """
    parent_dir = os.getcwd()
    data_path = os.path.join(parent_dir, data)

    # Creating train image augmentation generator
    # We will allow horizontal flip, little rotation and shifts
    train_image_gen = image.ImageDataGenerator(horizontal_flip=True,
                                               rotation_range=30,
                                               width_shift_range=0.3,
                                               height_shift_range=0.3,
                                               validation_split=0.2)

    # Creating image data generator for train
    train_image_data = train_image_gen.flow_from_directory(data_path,
                                                           target_size=size,
                                                           batch_size=1,
                                                           class_mode='binary',
                                                           subset='training')
    # Creating image data generator for val
    validation_image_data = train_image_gen.flow_from_directory(data_path,
                                                                target_size=size,
                                                                batch_size=1,
                                                                class_mode='binary',
                                                                subset='validation')

    # Create test image generator
    test_image_gen = image.ImageDataGenerator()

    # Creating image data generator for evaluation on full dataset
    evaluation_image_data = test_image_gen.flow_from_directory(data_path,
                                                               target_size=size,
                                                               batch_size=1,
                                                               class_mode='binary')

    return train_image_data, validation_image_data, evaluation_image_data


def train_model(model, train, val, num_epochs=300):
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
                                  samples_per_epoch=135/1,
                                  validation_data=val,
                                  epochs=num_epochs)

    return history


def plot_history(history):
    """
    Draw accuracy and loss function graphics
    :param history: history of model training
    """

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def save_model(model, name="iceberg_model.h5", folder=None):
    """
    Saving model to file
    :param model: model to save
    :param name: model name
    :param folder: folder to save (if it needs)
    """

    if folder:
        path = os.path.join(os.getcwd(), folder)
    else:
        path = os.getcwd()

    model.save(os.path.join(path, name))

    print('Model', name, 'has successfully saved!')


def evaluate_model(model, eval_data):
    """
    Evaluate model on full data
    :param model: model to evaluate
    :param eval_data: full data generator
    """

    score = model.evaluate_generator(eval_data)
    print("Loss", score[0], "Accuracy", score[1])


#def predict_class(model, image_path):

