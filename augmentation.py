from keras.preprocessing.image import ImageDataGenerator
import os


def image_augmentation(data):
    """
    Image augmentation, create augmented images in folder with original images
    :rtype: DirectoryIterator
    :param data: name of folder with folders 0 and 1
    """

    # Creating image generator
    # We will allow horizontal flip, little rotation and shifts
    image_datagen = ImageDataGenerator(horizontal_flip=True,
                                       rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2)

    image_gen = image_datagen.flow_from_directory(data,
                                                  target_size=(128, 128),
                                                  batch_size=32,
                                                  class_mode='binary')

    return image_gen
