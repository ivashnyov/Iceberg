from keras.preprocessing import image
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot


def image_augmentation(data):
    """
    Image augmentation, create augmented images in folder with original images
    :rtype: DirectoryIterator
    :param data: name of folder with folders 0 and 1
    """

    # Change directory to folder 'data'
    parent_dir = os.getcwd()
    os.chdir(os.path.join(parent_dir, data))
    folders = os.listdir(os.getcwd())

    # Creating image generator
    # We will allow horizontal flip, little rotation and shifts
    image_gen = image.ImageDataGenerator(horizontal_flip=True,
                                         rotation_range=15,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1)

    train = []

    # Start augmenting for each class
    for folder in folders:
        # Avoiding hidden folders
        if not folder.startswith('.'):
            images = os.listdir(folder)
            for im in images:
                try:
                    img_path = os.path.join(folder, im)
                    img = Image.open(img_path)
                    train.append(np.array(img))
                except Exception as e:
                    print(e)

            # Converting list into numpy array and correct format
            train = np.array(train)
            train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 3)

            # Fitting image generator
            image_gen.fit(train)
            os.chdir(os.path.join(parent_dir, data))

            total_image_augmented = 0

            for X_batch in image_gen.flow(train, batch_size=9, save_to_dir=folder, save_prefix='aug',
                                          save_format='png'):
                total_image_augmented += 1

            print(total_image_augmented, 'images were augmented for class', folder)


image_augmentation('data')
