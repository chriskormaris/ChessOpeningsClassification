import os
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split


def load_dataset(base_path='./dataset', grayscale=True, data_augmentation=False, normalization=True):
    ruy_lopez_dir = base_path + '/ruy_lopez'
    italian_game_dir = base_path + '/italian_game'
    queens_gambit_dir = base_path + '/queen\'s_gambit'
    sicilian_defense_dir = base_path + '/sicilian_defense'
    nimzo_indian_defense_dir = base_path + '/nimzo-indian_defense'

    X, X_real_images, y, y_real_images = get_images(
        ruy_lopez_dir,
        italian_game_dir,
        queens_gambit_dir,
        sicilian_defense_dir,
        nimzo_indian_defense_dir,
        grayscale=grayscale
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Concatenate real images only on validation and test datasets.
    X_val_real_images, X_test_real_images, y_val_real_images, y_test_real_images = train_test_split(
        X_real_images,
        y_real_images,
        stratify=y_real_images,
        test_size=0.5,
        random_state=42
    )
    X_val = np.concatenate((X_val, X_val_real_images), axis=0)
    y_val = np.concatenate((y_val, y_val_real_images), axis=0)
    X_test = np.concatenate((X_test, X_test_real_images), axis=0)
    y_test = np.concatenate((y_test, y_test_real_images), axis=0)

    if data_augmentation:
        for i, x in enumerate(X_train):
            im = Image.fromarray(x)
            enhancer = ImageEnhance.Brightness(im)
            # Image Brightening/Darkening
            # If e < 1, then the image becomes darker. If e > 1, then the image becomes brighter.
            for e in [0.5, 0.7, 0.9, 1.1, 1.2]:
                new_im = enhancer.enhance(e)
                if grayscale:
                    new_im = new_im.convert('L')
                else:
                    new_im = new_im.convert('RGB')
                # new_im.show()

                new_im_array = np.array(new_im)
                new_im_array = np.expand_dims(new_im_array, axis=0)

                X_train = np.concatenate((X_train, new_im_array), axis=0)
                y_train = np.append(y_train, y_train[i])
            if not grayscale:
                # Channel Shifting
                for mult in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                    Rmult, Gmult, Bmult = mult

                    matrix = (Rmult, 0, 0, 0,
                              0, Gmult, 0, 0,
                              0, 0, Bmult, 0)

                    new_im = im.convert('RGB', matrix)
                    # new_im.show()

                    new_im_array = np.array(new_im)
                    new_im_array = np.expand_dims(new_im_array, axis=0)

                    X_train = np.concatenate((X_train, new_im_array), axis=0)
                    y_train = np.append(y_train, y_train[i])

    if grayscale:
        X_train = np.expand_dims(X_train, axis=3)
        X_val = np.expand_dims(X_val, axis=3)
        X_test = np.expand_dims(X_test, axis=3)

    if normalization:
        X_train = X_train / 255.
        X_val = X_val / 255.
        X_test = X_test / 255.

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_images(
        ruy_lopez_dir,
        italian_game_dir,
        queens_gambit_dir,
        sicilian_defense_dir,
        nimzo_indian_defense_dir,
        grayscale=True
):
    X = None
    X_real_images = None
    y = np.array([])
    y_real_images = np.array([])

    dirs = [
        ruy_lopez_dir,
        italian_game_dir,
        queens_gambit_dir,
        sicilian_defense_dir,
        nimzo_indian_defense_dir
    ]

    for category, dir in enumerate(dirs):
        dir_list = os.listdir(dir)
        for i, file in enumerate(dir_list):
            print(file)
            im = Image.open(dir + '/' + file)
            im = im.resize((100, 100))

            if grayscale:
                im = im.convert('L')
            else:
                im = im.convert('RGB')

            im_array = np.array(im)
            im_array = np.expand_dims(im_array, axis=0)

            if file.startswith('IMG'):
                if X_real_images is None:
                    X_real_images = im_array
                else:
                    X_real_images = np.concatenate((X_real_images, im_array), axis=0)
                y_real_images = np.append(y_real_images, category)
            else:
                if X is None:
                    X = im_array
                else:
                    X = np.concatenate((X, im_array), axis=0)
                y = np.append(y, category)

    y = y.astype(np.int8)
    y_real_images = y_real_images.astype(np.int8)
    return X, X_real_images, y, y_real_images


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        base_path='./dataset',
        grayscale=True,
        data_augmentation=True
    )

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    print(Counter(y_train))
    print(Counter(y_val))
    print(Counter(y_test))
