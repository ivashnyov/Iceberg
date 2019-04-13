from PIL import Image
import os


def image_resizing(data, size):
    """
    Resize all images in folder
    :param data: name of folder with folders 0 and 1
    :param size: size to resize to
    """

    # Control variables
    total_images_resized = 0
    total_errors = 0

    # Change directory to folder 'data'
    parent_dir = os.getcwd()
    os.chdir(os.path.join(parent_dir, data))
    folders = os.listdir(os.getcwd())

    for folder in folders:
        # Avoiding hidden folders
        if not folder.startswith('.'):
            images = os.listdir(folder)
            for image in images:
                try:
                    img_path = os.path.join(folder, image)
                    img = Image.open(img_path)

                    # I know that it is better to maintain aspect ratio
                    # but in this case I think it is not so important
                    img = img.resize(size, Image.ANTIALIAS)
                    img.save(img_path, 'PNG')
                    print('Image', img_path, 'was resized!')
                    total_images_resized += 1
                except Exception as e:
                    print(e)
                    total_errors += 1

    print(total_images_resized, 'images were resized with', total_errors, 'errors')
